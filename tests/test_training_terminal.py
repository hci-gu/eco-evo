"""Exercise CPU startup behind a launcher with a real controlling terminal."""

import errno
import os
import select
import signal
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# Like `uv run`, the launcher owns the terminal's foreground process group
# and waits for a Python child. Report job-control suspension immediately,
# rather than letting a broken trainer hang until the test timeout.
LAUNCHER = """
import os, subprocess, sys
child = subprocess.Popen(sys.argv[1:])
_, status = os.waitpid(child.pid, os.WUNTRACED)
if os.WIFSTOPPED(status):
    print('TRAINER_STOPPED_BY_SIGNAL=' + str(os.WSTOPSIG(status)), flush=True)
    child.kill()
    child.wait()
    sys.exit(99)
sys.exit(os.waitstatus_to_exitcode(status))
"""


def run_in_terminal(command, *, reply_to_prompt=None):
    import pty

    launcher, terminal = pty.fork()
    if launcher == 0:
        os.chdir(ROOT)
        os.execv(sys.executable, [sys.executable, "-c", LAUNCHER, *command])
    output = bytearray()
    status = None
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if select.select([terminal], [], [], 0.1)[0]:
                try:
                    chunk = os.read(terminal, 65536)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    chunk = b""
                output.extend(chunk)
                if reply_to_prompt is not None and b"Continue with these parameters?" in output:
                    os.write(terminal, reply_to_prompt)
                    reply_to_prompt = None
            finished, candidate = os.waitpid(launcher, os.WNOHANG)
            if finished:
                status = candidate
                break
        assert status is not None, "Trainer did not finish:\n" + output.decode(errors="replace")
        return os.waitstatus_to_exitcode(status), output.decode(errors="replace")
    finally:
        if status is None:
            os.killpg(launcher, signal.SIGKILL)
            os.waitpid(launcher, 0)
        os.close(terminal)


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX terminal job control")
@pytest.mark.parametrize("confirm", [True, False])
def test_cpu_training_starts_from_terminal_launcher(confirm, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    command = ([sys.executable, "train.py"] if confirm else
               [sys.executable, "-c", "from train import main; main(confirm=False)"])
    command += ["--project", "mareld2.yaml", "--run-name", str(tmp_path / "run"),
                "--grid", "5x6", "--n_deltas", "2", "--workers", "2",
                "--n_eval_ticks", "2", "--generations", "1", "--iter-per-gen", "1"]
    status, output = run_in_terminal(command, reply_to_prompt=b"n\n" if confirm else None)
    assert "TRAINER_STOPPED_BY_SIGNAL" not in output, output
    assert status == 0, output
    if confirm:
        assert "Aborted by user." in output
    else:
        assert "Continue with these parameters?" not in output
        assert (tmp_path / "run" / "policy_gadoids.pth").is_file()
