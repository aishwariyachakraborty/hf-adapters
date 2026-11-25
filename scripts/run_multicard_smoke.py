# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the multicard smoke test and print a human-readable summary.

For real tensor-parallel multi-card runs, launch with torchrun (one process
per card).  For single-card, plain python works.

Requirements:
  - transformers==5.23.x  (uv pip install "transformers==5.23.0")

SPYRE_DEVICES tells the Flex runtime which card indices to use per rank.
Index-to-physical-card mapping is handled internally by Flex and is not
independently verifiable from this script.

1-card example::

    export SPYRE_DEVICES=0
    torchrun --nproc-per-node=1 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

2-card example (valid indices are node-specific — check yours first)::

    export SPYRE_DEVICES=0,1
    torchrun --nproc-per-node=2 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

4-card example::

    export SPYRE_DEVICES=0,1,2,3
    torchrun --nproc-per-node=4 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

Use ``--max-new-tokens`` to control output length.  By default the model
stops at EOS (typically just a few tokens).  Pass ``--min-new-tokens``
equal to ``--max-new-tokens`` to suppress EOS and force exactly N tokens
every run — useful for consistent latency benchmarking::

    export SPYRE_DEVICES=0
    torchrun --nproc-per-node=1 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16 \\
        --max-new-tokens 256 --min-new-tokens 256

Pass ``--with-profiling`` to record the timed generate runs with
``torch.profiler.profile`` (CPU + PrivateUse1 activities, shapes).  Model
load, torch.compile and the warmup generate run before the profiler starts,
so the trace covers steady-state generation only.  One profiler covers all
``--iterations`` runs, and each rank writes one Chrome trace,
``<trace-dir>/trace_rank<RANK>_pid<PID>.json``; view it at
https://ui.perfetto.dev.  Setting ``STA_TRACE_DIR`` turns profiling on and
uses that directory, which is what the spyre-trace-analyser expects of a
re-runnable workload.

With ``--batch 1`` each generate step is marked with a ``record_function``
span: ``p<RR>/ttft`` for prefill (up to the first token) and
``p<RR>/itl_<NNNN>`` for each decode step, where ``RR`` is the run index.
spyre-trace-analyser reads these as its steps, and needs at least 5 per
phase: use ``--iterations 5`` or more for prefill, and force a fixed length
with ``--min-new-tokens`` equal to ``--max-new-tokens`` for decode.  With
``--batch`` > 1 the steps are not marked, because generate only accepts a
streamer at batch size 1.

Set ``FLEX_TIMESTAMP_CALIBRATION=always`` so device timestamps are calibrated
rather than estimated.  Profiling adds overhead, so TTFT / ITL from a
profiled run are not comparable to an unprofiled one::

    export SPYRE_DEVICES=0,1
    export FLEX_TIMESTAMP_CALIBRATION=always
    torchrun --nproc-per-node=2 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16 \\
        --max-new-tokens 16 --min-new-tokens 16 --iterations 5 \\
        --with-profiling --trace-dir ./spyre_traces

The --model argument accepts any HuggingFace repo ID or local path
(default: ibm-granite/granite-3.3-8b-instruct).

The script exits with code 0 on PASS and code 1 on FAIL or ERROR. A FAIL is
reported by that exit code alone, not by an exception, so under torchrun the
only thing printed after the summary is the elastic agent's teardown:

    Sending process <pid> closing signal SIGTERM
    failed (exitcode: 1) local_rank: 0 ...
    torch.distributed.elastic.multiprocessing.errors.ChildFailedError

That SIGTERM goes to the ranks that were still running; torchrun stops the
whole gang once any rank exits nonzero. It is the result of the failure, never
the cause -- read the RESULTS SUMMARY above it for the real verdict.

Before exiting, the script calls ``dist.destroy_process_group()``. Once
torch-spyre#6009 lands, this calls ``SpyreCCLBackend::shutdown()`` and so
``spyre_comms::finalize_library()``; until then ``shutdown()`` is an empty
function and the call does not reach ``finalize_library()``. The intent is to
release the Spyre runtime in order instead of leaving it to static destructors
in unspecified order, which may have contributed to the ``corrupted
double-linked list`` / SIGABRT after the RESULTS SUMMARY. That has not been
proven, so if a SIGABRT (exit 134) follows ``Status: PASS``, treat it as a
teardown problem, not a model failure.

If the exit barrier fails (a peer rank is gone), the script skips both
``destroy_process_group()`` and interpreter teardown and exits via
``os._exit()`` with the verdict's exit code, since either could hang in
``finalize_library()``'s own barrier.

Set ``FLEX_CB_TIMELINE_FILE`` to record flex's per-control-block arrival dump
(``FLEX_CB_TIMELINE_MODE=cb|batch``). The script flushes it on every rank just
before exiting, because the ``os._exit()`` path below skips the exit-time
write. Each rank writes ``<path>.<pid>``; the directory must exist::

    mkdir -p /tmp/cb_timeline
    export FLEX_CB_TIMELINE_FILE=/tmp/cb_timeline/cb.csv
    torchrun --nproc-per-node=2 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.profiler import record_function
from transformers.generation.streamers import BaseStreamer

# Ensure the project root (parent of scripts/) is on sys.path so that
# tests.spyre.test_multicard_spyre can be imported when running directly
# from anywhere inside the repo.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.spyre.test_multicard_spyre import run_multicard_smoke_test  # noqa: E402

DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MODEL = "ibm-granite/granite-3.3-8b-instruct"
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_TRACE_DIR = "./spyre_traces"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model path or local dir (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Token generation budget (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help=(
            "Torch dtype to pass to AutoSpyreModelForCausalLM.from_pretrained "
            "(e.g. float16, bfloat16, float32).  Omit to let the model decide."
        ),
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=None,
        help=(
            "Suppress EOS until this many tokens have been generated.  "
            "Set equal to --max-new-tokens to force exactly N tokens every run "
            "(useful for latency benchmarking).  Omit to let the model stop naturally."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        metavar="N",
        help="Number of identical prompts to batch together (default: 1).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        metavar="N",
        help="Number of timed generate runs to perform (default: 1).",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=f"Read prompt from the file (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "--with-profiling",
        action="store_true",
        help=(
            "Profile the timed generate runs with torch.profiler "
            "(load, compile and warmup excluded).  Also on when "
            "STA_TRACE_DIR is set."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        default=DEFAULT_TRACE_DIR,
        help=(
            "Directory for the per-rank Chrome traces, used with "
            f"--with-profiling (default: {DEFAULT_TRACE_DIR}).  "
            "STA_TRACE_DIR overrides it."
        ),
    )
    return parser


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}{unit}"


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class _StepAnnotator(BaseStreamer):
    """Mark each generate step with a ``record_function`` span.

    generate() calls put() once with the prompt, then once per selected
    token, then end().  So the prefill span runs from the prompt to the first
    token, and decode span N from token N to token N+1.  No span is opened
    after the last token of the budget, so a forced-length run has no empty
    trailing step.
    """

    def __init__(self, prefix: str, max_new_tokens: int) -> None:
        self._prefix = prefix
        self._max_new_tokens = max_new_tokens
        self._tokens = -1  # the first put() is the prompt
        self._span: record_function | None = None

    def _close(self) -> None:
        if self._span is not None:
            self._span.__exit__(None, None, None)
            self._span = None

    def put(self, value: torch.Tensor) -> None:
        self._close()
        self._tokens += 1
        if self._tokens == 0:
            leaf = "ttft"
        elif self._tokens < self._max_new_tokens:
            leaf = f"itl_{self._tokens:04d}"
        else:
            return
        self._span = record_function(f"{self._prefix}/{leaf}")
        self._span.__enter__()

    def end(self) -> None:
        self._close()


def _flush_cb_dump(local_rank: int) -> None:
    """Write flex's CB dump (FLEX_CB_TIMELINE_FILE) now.

    The os._exit() path in main() (exit barrier failed) skips the dump's
    exit-time write; flushing explicitly also covers the normal exit path.
    """
    if not os.environ.get("FLEX_CB_TIMELINE_FILE"):
        return
    try:
        # No synchronize needed: flex records a CB just before its completion
        # callback, and the test already waited on the run's results.
        import torch_spyre

        rows = torch_spyre._C.flush_cb_dump()
        sys.stdout.write(f"[rank {local_rank}] CB dump: flushed {rows} rows\n")
    except Exception as e:  # pragma: no cover - diagnostics only, never fail the run
        sys.stdout.write(f"[rank {local_rank}] CB dump flush failed: {e}\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sta_trace_dir = os.environ.get("STA_TRACE_DIR")
    if sta_trace_dir:
        args.with_profiling = True
        args.trace_dir = sta_trace_dir
    if args.dtype is not None:
        if args.dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unknown --dtype {args.dtype!r}. "
                f"Choose from: {', '.join(_DTYPE_MAP)}"
            )
    dtype = _DTYPE_MAP.get(args.dtype) if args.dtype is not None else None

    aiu_ids = os.environ.get("AIU_IDS")
    spyre_devices = os.environ.get("SPYRE_DEVICES")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_rank0 = local_rank == 0

    # Trim AIU_IDS to WORLD_SIZE entries so a 4-address env var doesn't get
    # printed when only 1 or 2 ranks are actually launched.
    if aiu_ids is not None:
        _ids = [c.strip() for c in aiu_ids.split(",") if c.strip()]
        if len(_ids) > world_size:
            aiu_ids = ",".join(_ids[:world_size])

    if is_rank0:
        print()
        print("=" * 70)
        print("  run_multicard_smoke")
        print("=" * 70)
        print(f"  SPYRE_DEVICES    : {spyre_devices!r}")
        print(f"  AIU_IDS          : {aiu_ids!r}  (fallback; SPYRE_DEVICES is primary)")
        print(f"  Model            : {args.model}")
        print(f"  max_new_tokens   : {args.max_new_tokens}")
        print(f"  batch            : {args.batch}")
        print(f"  iterations       : {args.iterations}")
        print(f"  dtype            : {args.dtype or '(not set — model default)'}")
        print(f"  Prompt File      : {args.prompt_file}")
        print(
            f"  Profiling        : {args.trace_dir if args.with_profiling else 'off'}"
        )
        print("=" * 70)

    if args.prompt_file is not None:
        try:
            with open(args.prompt_file, "r") as _f:
                prompt = _f.read()
        except OSError as e:
            raise RuntimeError(
                f"Cannot read prompt file {args.prompt_file!r}: {e}"
            ) from e
    else:
        prompt = DEFAULT_PROMPT

    timed_runs_ctx = None
    streamer_factory = None
    if args.with_profiling:
        from torch.profiler import ProfilerActivity, profile

        rank = int(os.environ.get("RANK", local_rank))

        def _export_trace(prof) -> None:
            os.makedirs(args.trace_dir, exist_ok=True)
            path = os.path.join(
                args.trace_dir, f"trace_rank{rank}_pid{os.getpid()}.json"
            )
            prof.export_chrome_trace(path)
            sys.stdout.write(f"[rank {local_rank}] trace written: {path}\n")
            sys.stdout.flush()

        def timed_runs_ctx():
            # Exits (and on_trace_ready fires) inside run_multicard_smoke_test,
            # well before the process group is destroyed below.
            return profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                record_shapes=True,
                on_trace_ready=_export_trace,
            )

        if args.batch == 1:

            def streamer_factory(run_idx: int) -> _StepAnnotator:
                return _StepAnnotator(f"p{run_idx:02d}", args.max_new_tokens)

        if is_rank0:
            if args.batch != 1:
                print(
                    "  NOTE: --batch > 1, so generate steps are not annotated; "
                    "the trace analyser needs --batch 1 to find ttft/itl steps."
                )
            if os.environ.get("FLEX_TIMESTAMP_CALIBRATION") != "always":
                print(
                    "  NOTE: FLEX_TIMESTAMP_CALIBRATION is not 'always'; device "
                    "timestamps may be estimated and fail the analyser's checks."
                )

    result = run_multicard_smoke_test(
        args.model,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        dtype=dtype,
        batch_size=args.batch,
        prompt=prompt,
        num_iterations=args.iterations,
        timed_runs_ctx=timed_runs_ctx,
        streamer_factory=streamer_factory,
    )

    # ── Summary table ──────────────────────────────────────────────────────
    # Each rank builds its summary as one string and writes it atomically.
    # A brief sleep stagger keeps blocks from interleaving in the terminal.
    import time

    def _rank_summary(r: dict) -> str:
        lines = [
            "",
            "=" * 70,
            f"  RESULTS SUMMARY  [rank {r['local_rank']}/{r['world_size']}]",
            "=" * 70,
            f"  SPYRE_DEVICES : {r['spyre_devices_env']!r}",
        ]
        if r["resolved_cards"]:
            lines.append(
                f"  PCI hint      : {', '.join(r['resolved_cards'])}"
                f"  (AIU_WORLD_RANK_* guess, unconfirmed)"
            )
        lines += [
            f"  AIU_IDS       : {r['aiu_ids_env']!r}",
            f"  Model         : {r['model']}",
            f"  Batch      : {r['batch_size']}",
            f"  Status     : {r['status']}",
            f"  Load       : {_fmt(r['load_s'], 's')}",
            f"  Generate   : {_fmt(r['gen_s'], 's')}",
            f"  TTFT       : {_fmt(r['ttft_ms'], ' ms')}",
            f"  Decode avg : {_fmt(r['decode_ms'], ' ms')}",
            f"  Steady ITL : {_fmt(r['steady_itl_ms'], ' ms')}  (outliers excluded)",
            f"  Output     : {r['output']!r}",
        ]
        # gen_s is None when load or warmup failed, i.e. the profiler never ran.
        if args.with_profiling and r["gen_s"] is not None:
            lines.append(f"  Trace dir  : {args.trace_dir}")
        if r["error"]:
            lines += ["", "  ERROR DETAIL:"]
            lines += [f"    {line}" for line in r["error"].splitlines()]
        lines += ["=" * 70, ""]
        return "\n".join(lines)

    time.sleep(local_rank * 0.3)
    sys.stdout.write(_rank_summary(result))
    sys.stdout.flush()

    failed = result["status"] != "PASS"

    # A FAIL is reported by exiting nonzero, not by raising, so torchrun has no
    # traceback to show. Say so explicitly: otherwise the only thing in the log
    # is the agent's SIGTERM/ChildFailedError teardown, which reads like an
    # abort or a timeout rather than a verdict this script already reached.
    if failed and local_rank == 0:
        sys.stdout.write(
            "\n"
            f"multicard smoke FAILED (status={result['status']!r}); "
            "exiting rank 0 with code 1.\n"
            "The torchrun 'SIGTERM' / ChildFailedError below is the elastic "
            "agent tearing down the remaining ranks after this exit. It is the "
            "consequence of this failure, not its cause; the real reason is in "
            "the RESULTS SUMMARY above.\n"
        )
        sys.stdout.flush()

    # Ranks reach their verdict a few ms apart. Whichever exits first leaves its
    # siblings alive, and torchrun SIGTERMs them mid-print -- truncating their
    # summaries and making one rank's failure look like an unrelated signal
    # kill. Sync first so every rank prints fully and they exit together.
    barrier_ok = True
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception as e:  # pragma: no cover - teardown best effort
            barrier_ok = False
            sys.stdout.write(f"\n[rank {local_rank}] exit barrier skipped: {e}\n")
            sys.stdout.flush()

    _flush_cb_dump(local_rank)

    # If the exit barrier failed, a peer is likely gone. Both an explicit
    # destroy_process_group() and normal interpreter teardown (which releases the
    # backend and runs its destructor -> finalize_library(), with its own
    # barrier) could then hang until torchrun kills the rank. Flush and bypass
    # interpreter teardown entirely, preserving the verdict's exit code.
    if not barrier_ok:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if failed else 0)

    # destroy_process_group() -> SpyreCCLBackend::shutdown() ->
    # spyre_comms::finalize_library() (once torch-spyre#6009 lands). Must run
    # before interpreter teardown so the Spyre runtime is released in order.
    # Best effort: a teardown error must not turn the verdict into a traceback.
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:  # pragma: no cover - teardown best effort
            sys.stdout.write(
                f"\n[rank {local_rank}] destroy_process_group failed: {e}\n"
            )
            sys.stdout.flush()

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
