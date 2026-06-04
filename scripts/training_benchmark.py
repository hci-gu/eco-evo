#!/usr/bin/env python3
"""Time-boxed benchmark wrapper for the real Mareld training loop.

The script runs ``train.py`` as a child process with normal training arguments,
suppresses the verbose training log, counts completed training iterations, and
prints compact throughput stats. Arguments after ``--`` are passed through to
``train.py`` unchanged, except that safe defaults are added when missing.
"""

from __future__ import annotations

import argparse
import os
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path


ITER_RE = re.compile(r"^\s*Iter\s+(\d+)\s*/\s*(\d+)\s*\|")
GEN_RE = re.compile(r"^=+\s*Generation\s+(\d+)/([^\s]+)")
SUMMARY_INT_RE = {
    "n_eval_ticks": re.compile(r"^N Eval Ticks:\s*(\d+)"),
    "n_deltas": re.compile(r"^N Deltas:\s*(\d+)"),
}
ROLLOUTS_RE = re.compile(r"^Rollouts/Delta:\s*(\d+)")


def _split_args(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1:]


def _has_flag(args: list[str], names: tuple[str, ...]) -> bool:
    for arg in args:
        for name in names:
            if arg == name or arg.startswith(name + "="):
                return True
    return False


def _default_train_args(train_args: list[str]) -> list[str]:
    out = list(train_args)
    if not _has_flag(out, ("--project",)):
        out.extend(["--project", "mareld2.yaml"])
    if not _has_flag(out, ("--run-name", "--run_name")):
        out.extend(["--run-name", f"benchmark-{int(time.time())}"])
    if not _has_flag(out, ("--generations",)):
        out.extend(["--generations", "inf"])
    return out


def _terminate_like_ctrl_c(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    except Exception:
        proc.send_signal(signal.SIGINT)


def main(argv: list[str] | None = None) -> int:
    bench_argv, train_argv = _split_args(list(sys.argv[1:] if argv is None else argv))
    parser = argparse.ArgumentParser(
        description="Run train.py for a fixed wall-clock duration and report throughput."
    )
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Benchmark duration in seconds before sending SIGINT. Default: 60.")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="Seconds between progress reports. Default: 10.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used to launch train.py. Default: current Python.")
    parser.add_argument("--show-train-output", action="store_true",
                        help="Also echo suppressed train.py output while benchmarking.")
    args = parser.parse_args(bench_argv)

    repo_root = Path(__file__).resolve().parents[1]
    train_args = _default_train_args(train_argv)
    cmd = [args.python, "-u", "train.py", *train_args]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Local macOS/Anaconda environments may otherwise abort on duplicate OpenMP.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    print("==========================================")
    print("      MARELD TRAINING BENCHMARK           ")
    print("==========================================")
    print(f"Duration:  {args.duration:g}s")
    print(f"Command:   {' '.join(cmd)}")
    print("------------------------------------------", flush=True)

    started = time.monotonic()
    next_report = started + max(1.0, args.interval)
    iterations = 0
    generations_seen = 0
    last_iter = None
    last_gen = None
    n_eval_ticks = None
    n_deltas = None
    rollouts_per_delta = None
    last_lines: list[str] = []

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdout is not None
    if proc.stdin is not None:
        try:
            proc.stdin.write("\n")
            proc.stdin.flush()
            proc.stdin.close()
        except BrokenPipeError:
            pass

    try:
        while True:
            now = time.monotonic()
            ready, _, _ = select.select([proc.stdout], [], [], 0.1)
            if ready:
                line = proc.stdout.readline()
            else:
                line = ""

            if line:
                stripped = line.rstrip("\n")
                last_lines.append(stripped)
                del last_lines[:-20]
                if args.show_train_output:
                    print(stripped)

                gen_m = GEN_RE.match(stripped)
                if gen_m:
                    generations_seen += 1
                    last_gen = int(gen_m.group(1))

                iter_m = ITER_RE.match(stripped)
                if iter_m:
                    iterations += 1
                    last_iter = (int(iter_m.group(1)), int(iter_m.group(2)))

                for key, regex in SUMMARY_INT_RE.items():
                    m = regex.match(stripped)
                    if not m:
                        continue
                    if key == "n_eval_ticks":
                        n_eval_ticks = int(m.group(1))
                    elif key == "n_deltas":
                        n_deltas = int(m.group(1))
                m_roll = ROLLOUTS_RE.match(stripped)
                if m_roll:
                    rollouts_per_delta = int(m_roll.group(1))

            if now >= next_report:
                elapsed = max(now - started, 1e-9)
                iter_per_s = iterations / elapsed
                msg = f"[bench] {elapsed:7.1f}s  iterations={iterations}  iter/s={iter_per_s:.4f}"
                if last_gen is not None:
                    msg += f"  gen={last_gen}"
                if last_iter is not None:
                    msg += f"  iter={last_iter[0]}/{last_iter[1]}"
                print(msg, flush=True)
                next_report = now + max(1.0, args.interval)

            if now - started >= args.duration:
                _terminate_like_ctrl_c(proc)
                break

            if proc.poll() is not None:
                break

        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    finally:
        _terminate_like_ctrl_c(proc)

    elapsed = max(time.monotonic() - started, 1e-9)
    iter_per_s = iterations / elapsed
    print("------------------------------------------")
    print(f"Elapsed:             {elapsed:.2f}s")
    print(f"Completed iters:     {iterations}")
    print(f"Iterations/second:   {iter_per_s:.5f}")
    if n_eval_ticks and n_deltas and rollouts_per_delta:
        invisible_ticks_per_iter = 2 * n_deltas * rollouts_per_delta * n_eval_ticks
        visible_probe_ticks = n_eval_ticks
        total_model_ticks = iterations * (invisible_ticks_per_iter + visible_probe_ticks)
        print(f"Rollout ticks/iter:  {invisible_ticks_per_iter} train + {visible_probe_ticks} probe")
        print(f"Model ticks/second:  {total_model_ticks / elapsed:.1f}")
    if last_gen is not None:
        print(f"Last generation:     {last_gen}")
    if last_iter is not None:
        print(f"Last iter in gen:    {last_iter[0]}/{last_iter[1]}")
    print(f"Child exit code:     {proc.returncode}")
    if proc.returncode not in (0, -signal.SIGINT):
        print("------------------------------------------")
        print("Last child output lines:")
        for line in last_lines[-10:]:
            print(f"  {line}")
    print("==========================================")
    return 0 if iterations > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
