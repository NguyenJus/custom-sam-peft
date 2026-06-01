import subprocess
from pathlib import Path

SCRIPT = "scripts/check_gpu_evidence.sh"


def _run(args):
    return subprocess.run(  # noqa: S603
        ["bash", SCRIPT, *args],  # noqa: S607
        capture_output=True,
        text=True,
    )


def test_exit_zero_when_artifact_missing(tmp_path):
    r = _run([str(tmp_path / "nonexistent.md"), "deadbeef"])
    assert r.returncode == 0  # non-blocking even when missing


def test_exit_zero_when_artifact_stale(tmp_path):
    art = tmp_path / "evidence.md"
    art.write_text("evidence for commit OLDSHA\n")
    r = _run([str(art), "NEWSHA"])
    assert r.returncode == 0  # non-blocking even when stale
    assert "stale" in (r.stdout + r.stderr).lower()


def test_exit_zero_and_green_when_current(tmp_path):
    art = tmp_path / "evidence.md"
    art.write_text("evidence for commit GOODSHA\n")
    r = _run([str(art), "GOODSHA"])
    assert r.returncode == 0
    out = (r.stdout + r.stderr).lower()
    assert "ok" in out or "current" in out or "green" in out


def test_workflow_declares_job_non_required():
    wf = Path(".github/workflows")
    text = "\n".join(p.read_text() for p in wf.glob("*.yml"))
    assert "gpu-evidence" in text
    # crude guard: the evidence job/step must not be marked required/blocking
    assert (
        "required: true" not in text or "gpu-evidence" not in text.split("required: true")[0][-200:]
    )
