#!/usr/bin/env python3
"""Everything for the paper in one go, on a CUDA machine. No arguments.

    uv run tools/paper/run_all.py

1. Checks the machine: CUDA, the trained policies (results/<run_name>/
   of the scenario files), the wind farm map (made if missing).
2. A smoke run (one scenario, a few days, small grid) so a broken setup
   fails in a minute, not after hours.
3. Every scenario in scenarios/paper/ (tools/paper/run_scenarios.py) into
   results/paper/. Finished runs are skipped: if it is stopped, run the
   same command again and it carries on.
4. The figures: every notebook in notebooks/paper/ run headless with
   "Save figures" ticked -> figures/paper/*.pdf, *.png.

A log of everything printed goes to results/paper/run_all.log.
"""

import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

RESULTS = os.path.join("results", "paper")
SMOKE = os.path.join("results", "paper_smoke")
LOG = os.path.join(RESULTS, "run_all.log")
# Most important first, the long harvest sweeps last.
ORDER = ["current_fishing", "no_fishing", "windfarm_poseidon", "warming",
         "habitat_trawling", "eutrophication", "fishing_constant", "fishing_rebuild"]


def log(message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, env=None):
    """Run ``cmd``, echoing its output to the console and the log."""
    log("$ " + " ".join(cmd))
    with open(LOG, "a", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def fail(message):
    log("ERROR: " + message)
    sys.exit(1)


def scenario_files():
    files = sorted(glob.glob(os.path.join("scenarios", "paper", "*.yaml")))
    rank = {name: i for i, name in enumerate(ORDER)}
    return sorted(files, key=lambda p: rank.get(os.path.splitext(os.path.basename(p))[0], 99))


def check_machine(files):
    import torch
    from lib.scenarios.spec import load_scenario

    if not torch.cuda.is_available():
        fail("torch sees no CUDA device. Run this on the GPU machine "
             "(or tools/paper/run_scenarios.py --device cpu for a small test).")
    log(f"CUDA: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

    run_names = set()
    for path in files:
        for variant in load_scenario(path):
            run_names.add(variant.run_name)
    for name in sorted(run_names):
        policies = glob.glob(os.path.join("results", name, "policy_*.pth"))
        if not policies:
            fail(f"no trained policies in results/{name}/ - copy that run's directory "
                 f"(policy_*.pth, gpu_run.json) to this machine first")
        log(f"policies: results/{name}/ ({len(policies)} groups)")

    windfarm = os.path.join("maps", "poseidon_placeholder.npz")
    if not os.path.exists(windfarm):
        log("making the placeholder wind farm map")
        if run([sys.executable, os.path.join("tools", "paper", "make_windfarm_placeholder.py")]):
            fail("could not make the wind farm map")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    started = time.time()
    files = scenario_files()
    log(f"=== paper run: {len(files)} scenario files")
    check_machine(files)

    runner = [sys.executable, os.path.join("tools", "paper", "run_scenarios.py")]

    log("=== smoke run (a few days on a small grid)")
    if run(runner + [os.path.join("scenarios", "paper", "windfarm_poseidon.yaml"),
                     "--device", "cuda", "--grid", "20x20", "--days", "5",
                     "--warmup-days", "2", "--seeds", "0", "--only", "noise_ban",
                     "--out", SMOKE, "--force", "--quiet"]):
        fail("the smoke run failed (see above); nothing else was started")

    log("=== all scenarios -> " + RESULTS)
    code = run(runner + files + ["--device", "cuda", "--out", RESULTS])
    if code:
        log("some runs failed (see above); the figures use the runs that finished. "
            "Run this script again to retry them.")

    log("=== figures -> figures/paper/")
    env = dict(os.environ, PAPER_SAVE="1", PAPER_RESULTS=os.path.abspath(RESULTS),
               MPLBACKEND="Agg")
    failed = []
    for notebook in sorted(glob.glob(os.path.join("notebooks", "paper", "*.py"))):
        if run(["uv", "run", "--group", "paper", "python", notebook], env=env):
            failed.append(notebook)
    figures = sorted(glob.glob(os.path.join("figures", "paper", "*.pdf")))
    log(f"{len(figures)} figures in figures/paper/")
    if failed:
        log("notebooks that failed: " + ", ".join(failed))

    hours = (time.time() - started) / 3600
    log(f"=== done in {hours:.1f} h"
        + ("" if not (code or failed) else " (with failures, see above)"))
    return 1 if (code or failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
