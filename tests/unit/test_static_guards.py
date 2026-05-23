# tests/unit/test_static_guards.py
"""Static guards enforce structural invariants from spec §3.

These are cheap regex-based checks. They land FIRST (failing) so that
Tasks 4 and 5 land the refactors that make them pass. After this PR,
they stay green as regression detectors.

Implemented in pure Python (stdlib re + pathlib) so no external binary
is required — works in CI runners that don't ship ripgrep.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "custom_sam_peft"


def _grep(pattern: str, *, in_dir: Path, suffix: str = ".py") -> list[str]:
    """Return `path:line:text` hits for *pattern* across *.suffix files under *in_dir*."""
    rx = re.compile(pattern)
    out: list[str] = []
    for path in in_dir.rglob(f"*{suffix}"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                out.append(f"{path}:{lineno}:{line}")
    return out


def test_no_peft_method_branches_outside_peft_adapters():
    """Spec §3 #2: no `if .*\\.method ==` in src/ outside peft_adapters/."""
    hits = _grep(r"\.method\s*==", in_dir=SRC)
    offenders = [h for h in hits if "/peft_adapters/" not in h and "test_" not in h]
    assert not offenders, (
        "PEFT method-string branches detected outside peft_adapters/.\n"
        "Move these behind the @register('peft', ...) factory and the\n"
        "PEFTMethod protocol. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_no_to_device_outside_collator_and_runtime():
    """Spec §3 #3: device-move sites collapse to data collator + runtime/.

    Allowed exceptions beyond runtime/ and data/collate.py:
      - /models/sam3.py (_build_geometric_prompt): model-internal dtype coercion
        of caller-supplied box hints to float32 on the model's own device.
        The device argument is derived from model allocations (not a dataset
        tensor move), and the call must also cast dtype — runtime.to_device
        only targets device placement, not dtype coercion. This is not a
        scattered dataset-to-device move; it belongs to model internals.
    """
    hits = _grep(r"\.to\(device", in_dir=SRC)
    allowed_substrings = ("/runtime/", "/data/collate.py", "/models/sam3.py")
    offenders = [h for h in hits if not any(allowed in h for allowed in allowed_substrings)]
    assert not offenders, (
        "`.to(device)` outside runtime/, data/collate.py, and models/sam3.py.\n"
        "Route all device moves through runtime.to_device. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_no_string_joined_checkpoint_paths_outside_paths_module():
    """Spec §3: no `runs/.../checkpoints/` string-joining outside paths/."""
    # Patterns: literal "checkpoints/" in a string, OR
    # f-strings / .format / + concat that build a checkpoints subpath.
    patterns = [
        r'"checkpoints/',  # literal
        r"'checkpoints/",  # literal single-quoted
        r'f".*checkpoints',  # f-string
        r"f'.*checkpoints",  # f-string single-quoted
    ]
    offenders: list[str] = []
    for pattern in patterns:
        hits = _grep(pattern, in_dir=SRC)
        for h in hits:
            if "/paths/" in h or "/_patches/" in h:
                continue  # paths/ owns the layout; _patches has no path code
            if "# noqa: paths-guard" in h:
                continue  # explicit opt-out (audit may surface legitimate cases)
            offenders.append(h)
    assert not offenders, (
        "String-joined checkpoint paths outside src/custom_sam_peft/paths/.\n"
        "Use paths.checkpoint_path(run_dir, step=N). Offenders:\n  " + "\n  ".join(offenders)
    )
