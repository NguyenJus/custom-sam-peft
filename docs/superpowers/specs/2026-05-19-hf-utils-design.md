# spec/hf-utils — HuggingFace Utils (token auth + auto download)

**Status:** Draft (2026-05-19)
**Tracking:** [#21](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/21) — *feat: HuggingFace utils — token auth + auto model download*
**Scope:** Ship a thin `esam3.utils.huggingface` module so non-technical users don't have to run `huggingface-cli download` to get SAM 3.1 weights. Two public functions (`resolve_hf_token`, `download_model`) plus wiring into three existing surfaces (`esam3 init`, `esam3 doctor`, `models/sam3.py::_resolve_checkpoint_path`). Add `huggingface-hub` as an explicit, non-optional project dependency.

**Builds on:** [`2026-05-15-esam3-architecture-design.md`](2026-05-15-esam3-architecture-design.md); [`2026-05-16-model-loading-design.md`](2026-05-16-model-loading-design.md) (for `_resolve_checkpoint_path` consumer); [`2026-05-18-cli-design.md`](2026-05-18-cli-design.md) (for `esam3 init` / `esam3 doctor` shapes).

---

## 1. Goals & v0 Scope

The repo `facebook/sam3.1` is gated. Today users must run `huggingface-cli login` and `huggingface-cli download facebook/sam3.1 --local-dir models/sam3.1` before anything works; if they forget, `_resolve_checkpoint_path` raises `FileNotFoundError` and points them at a CLI they may not have used before. This spec makes the happy path "install the package, run `esam3 init`, training works" — the same UX the architecture promises.

**In scope:**

| Item | Where |
| --- | --- |
| New package `src/esam3/utils/` (first util) with `__init__.py` and `huggingface.py` | `src/esam3/utils/` |
| Public `resolve_hf_token(token: str \| None) -> str \| None` | `utils/huggingface.py` |
| Public `download_model(repo_id, local_dir, *, token=None, revision=None, force=False) -> Path` | `utils/huggingface.py` |
| Auto-download on missing weights in `_resolve_checkpoint_path` | `models/sam3.py` |
| Prompt-on-miss UX in `esam3 init` (tri-state flag + `--yes`) | `cli/init_cmd.py` |
| Doctor reports HF auth status (local probe only — no network) | `diagnostics.py`, `cli/doctor_cmd.py` |
| `huggingface-hub` moved from transitive to an explicit hard dep | `pyproject.toml` |
| Unit tests covering both APIs, the three wiring points, and the rename | `tests/unit/test_utils_huggingface.py` + extensions to `test_cli_init.py`, `test_cli_doctor.py`, the existing `_resolve_checkpoint_path` test |

**Naming note (intentional rename from issue #21).** Issue #21 proposed `ensure_hf_login`. This spec uses `resolve_hf_token` because we do **NOT** call `huggingface_hub.login()` — there is no token persistence, no writing to `~/.cache/huggingface/token`. The function is a pure read-side probe (`explicit arg → env → cached creds`). The name change reflects the behavior.

**Out of scope (explicit deferral):**

- An `auto_download: bool` config knob. Auto-download on miss is unconditional in `_resolve_checkpoint_path`; the only knobs are at the CLI level (`esam3 init`'s tri-state flag).
- A new `esam3 hf` / `esam3 login` subcommand. `init` handles the bootstrap case; `doctor` reports status.
- Real-Hub integration tests. Every test mocks `huggingface_hub.snapshot_download` and the error classes. Network coverage is too flaky and the gated repo is too sensitive for CI.
- Uploads, pushing to Hub, custom HF endpoint, multi-repo / dataset downloads.
- Making `huggingface-hub` an optional extra. It becomes a direct hard dep — already pulled transitively (via `transformers` / `datasets` / `sam3`), so we are codifying reality rather than expanding the install footprint.

---

## 2. Module Layout

```text
src/esam3/utils/
  __init__.py           # NEW — empty, package marker
  huggingface.py        # NEW — resolve_hf_token + download_model

src/esam3/
  models/sam3.py        # CHANGED — _resolve_checkpoint_path auto-downloads on miss
  diagnostics.py        # CHANGED — DoctorReport gains hf_auth: HuggingFaceAuthInfo
  cli/init_cmd.py       # CHANGED — three new flags; prompt-on-miss flow
  cli/doctor_cmd.py     # CHANGED — _render_table adds "HuggingFace auth" sub-table

pyproject.toml          # CHANGED — add huggingface-hub to [project] dependencies

tests/unit/
  test_utils_huggingface.py   # NEW — direct tests of both functions
  test_cli_init.py            # EXTENDED — flag + prompt-on-miss + short-circuit
  test_cli_doctor.py          # EXTENDED — auth section + JSON round-trip
  test_sam3_wrapper.py (or _model_config.py)  # EXTENDED — auto-download path
                              # planner pins the file based on where the existing
                              # _resolve_checkpoint_path test lives (none exists
                              # today; the planner creates it in the file that
                              # already imports from esam3.models.sam3).
```

No deletions. No moves. The new package `esam3.utils` exists for future utils too; this spec ships the first one.

---

## 3. Public Surfaces

### 3.1 `src/esam3/utils/__init__.py`

Empty (package marker). No re-exports — callers import from `esam3.utils.huggingface` directly. This keeps the import graph cheap (the top-level `esam3.utils` doesn't pull in `huggingface_hub` unless the consumer asks for it).

### 3.2 `src/esam3/utils/huggingface.py`

```python
"""HuggingFace Hub helpers for token resolution and model download.

This module is a thin wrapper around ``huggingface_hub``: it never calls
``login()`` (no token persistence), and it never logs the resolved token.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_hf_token(token: str | None = None) -> str | None:
    """Resolve an HF token from explicit arg → ``HF_TOKEN`` env → cached creds.

    Returns the token string, or ``None`` if none is available. Never persists
    the token; never logs its value.
    """


def download_model(
    repo_id: str,
    local_dir: Path,
    *,
    token: str | None = None,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Snapshot-download ``repo_id`` into ``local_dir`` if not already present.

    Idempotent unless ``force=True``: when ``local_dir`` exists and is non-empty,
    returns immediately without contacting the Hub. Returns ``local_dir``.
    """
```

**`resolve_hf_token` contract.**

Probe order (returns the first non-empty token; never falls through after a hit):

1. The explicit `token` argument, if truthy.
2. `os.environ.get("HF_TOKEN")`, if truthy.
3. `huggingface_hub.get_token()`, which reads `~/.cache/huggingface/token` (the file written by `huggingface-cli login`).

Returns `None` when all three are empty. Never raises. Never calls `huggingface_hub.login()`. Never logs the token value.

> **Planner verification:** `huggingface_hub.get_token()` is the public API in v0.19+; older code uses `HfFolder.get_token()`. Confirm `get_token` is exposed at the top level in `huggingface_hub==1.15.0` (the version pinned by `uv.lock`).

**`download_model` contract.**

1. **Skip condition.** If `local_dir.exists()` AND it is non-empty (`any(local_dir.iterdir())`) AND `force is False`, return `local_dir` immediately. No log, no Hub call. The "non-empty" check is the cheap presence marker; the consumer that knows the expected checkpoint filename (see §4.3) does the file-level check after the call.
2. **Resolve token.** `resolved = resolve_hf_token(token)`. The resolved value is opaque from here on — it's passed to `snapshot_download(token=resolved)` and never embedded in any log line or error message.
3. **Announce.** One info-level log line: `logger.info("fetching %s → %s", repo_id, local_dir)`. The token must not appear in the message.
4. **Create the directory.** `local_dir.mkdir(parents=True, exist_ok=True)` before the download.
5. **Call the Hub.** `huggingface_hub.snapshot_download(repo_id=repo_id, local_dir=str(local_dir), revision=revision, token=resolved, <kwarg-to-materialize-real-files>=True)`.
6. **Return.** `return local_dir`.

> **Planner verification (load-bearing):** The kwarg name that makes `snapshot_download` materialize real files inside `local_dir` (rather than symlinks pointing at `~/.cache/huggingface/hub/`) differs across `huggingface_hub` major versions. In `0.x` it was `local_dir_use_symlinks=False`; in `1.x` the symlink option may be removed/replaced. Pin the exact kwarg against `huggingface_hub==1.15.0` and use it. The reason we care: when the user later moves their `models/sam3.1/` directory to another machine or a different POSIX filesystem (or NTFS over WSL), symlinks dangle. We want a self-contained `local_dir`. If the kwarg is gone in 1.x because real files are now the default, drop the kwarg and document the new behavior in the implementation plan — same outcome either way.

**Error mapping.**

The Hub raises a small zoo of exceptions. Map the three load-bearing ones to a single `RuntimeError`; let everything else propagate (including unforeseen `huggingface_hub.errors.*` subclasses, network timeouts that surface as `OSError`, etc. — the user will see a real traceback and we can decide later which to map).

| Caught | Re-raised as | Message (sketch) |
| --- | --- | --- |
| `GatedRepoError` | `RuntimeError` | `"could not download '<repo_id>': the repo is gated. Accept the license at https://huggingface.co/<repo_id> and then `export HF_TOKEN=<your-token>`."` |
| `RepositoryNotFoundError` | `RuntimeError` | `"could not download '<repo_id>': repo not found, or your token lacks access. Check the repo id and verify your token."` |
| `HfHubHTTPError` (generic) | `RuntimeError` | `"could not download '<repo_id>': Hub request failed (<status code if available>). Check network and `export HF_TOKEN=...` if the repo is private/gated."` |

The resolved token MUST NOT appear in any of these messages — verified in tests (§5.1). Other exception types propagate unchanged.

> **Planner verification:** `huggingface_hub.errors` import path in `huggingface_hub==1.15.0`. Older versions exposed `GatedRepoError` / `RepositoryNotFoundError` / `HfHubHTTPError` at `huggingface_hub.utils` or `huggingface_hub.errors`. Pin the correct module.

**Logging.** Module-level `logger = logging.getLogger(__name__)`. No Typer, no rich — these surface in CLI layers that wrap this module. The "fetching" line is the only chatter; success returns silently.

---

## 4. Wiring — three surfaces

### 4.1 `esam3 init` — prompt-on-miss UX

`init_cmd.py` today writes the rendered template and exits. The change: after writing the config, load it via `esam3.config.loader.load_config`, then probe whether the checkpoint already exists on disk. Drive a small decision matrix.

**New Typer options.**

```python
def init(
    template: str = ...,
    output: Path = ...,
    force: bool = ...,
    download_weights: bool | None = typer.Option(
        None,
        "--download-weights/--no-download-weights",
        help="Download SAM 3.1 weights after writing the config. "
             "Default: prompt interactively when stdout is a TTY.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Skip the interactive prompt; assume yes. Implies --download-weights "
             "when --no-download-weights is not also passed.",
    ),
) -> None:
```

`download_weights` is tri-state: `True` / `False` / `None` (Typer's behavior for `Optional[bool]` with a paired `--x/--no-x` switch).

**Decision matrix** (executed *after* `output.write_text(body)`):

```text
cfg = load_config(output)
ckpt = Path(cfg.model.local_dir) / cfg.model.checkpoint_file

if ckpt.exists():
    rprint(f"[dim]weights already present at {ckpt}; skipping download[/dim]")
    return

if download_weights is False:
    rprint(f"[dim]skipping weights download; weights will be fetched on first "
           f"`esam3 train`. Re-run `esam3 init --download-weights` (or "
           f"`huggingface-cli download {cfg.model.name} --local-dir {cfg.model.local_dir}`) "
           f"to fetch them now.[/dim]")
    return

if download_weights is True or yes:
    proceed = True
elif sys.stdin.isatty():
    proceed = typer.confirm(
        f"Download {cfg.model.name} weights into {cfg.model.local_dir}? "
        f"(this can be several GB)",
        default=True,
    )
else:
    rprint(f"[dim]non-interactive shell and no --download-weights flag; skipping. "
           f"Weights will be fetched on first `esam3 train`; re-run with "
           f"`--download-weights` to fetch them now.[/dim]")
    return

if proceed:
    download_model(cfg.model.name, Path(cfg.model.local_dir), revision=cfg.model.revision)
```

Behavior summary in a table:

| `--download-weights` | `--no-download-weights` | `--yes` | tty? | weights present? | Action |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | yes | print "already present; skipping" |
| — | set | — | — | no | print hint, skip |
| set | — | — | — | no | download silently (no prompt) |
| — | — | set | — | no | download silently (no prompt) |
| — | — | — | yes | no | `typer.confirm(default=True)` |
| — | — | — | no | no | print hint pointing at `--download-weights`, skip |

If `cfg.model.local_dir is None` (a user edited the template), skip with a hint — there is no directory to download into.

**Imports.** `init_cmd.py` adds `from esam3.utils.huggingface import download_model` and `from esam3.config.loader import load_config` and `import sys`. Library calls only — no logic.

**Failure mode.** If `download_model` raises `RuntimeError` (gated/not-found/HTTP), the CLI surface should translate it to `typer.Exit(1)` with `rich.print` of the message, matching the cross-cutting error model from `2026-05-18-cli-design.md` §7. The init has already succeeded at writing the config; the download is a follow-on convenience that can fail without invalidating the written file.

### 4.2 `esam3 doctor` — local auth status (no network)

The doctor's "cheap, no network" contract (`2026-05-18-cli-design.md` §5.3) is preserved. We only report what the local environment says, not what the Hub thinks of our token.

**New dataclass** added to `src/esam3/diagnostics.py`:

```python
from typing import Literal

@dataclass(frozen=True)
class HuggingFaceAuthInfo:
    token_source: Literal["env", "cache", "none"]
    has_token: bool
```

**`DoctorReport` extension.** A new field:

```python
@dataclass(frozen=True)
class DoctorReport:
    # ... existing fields ...
    hf_auth: HuggingFaceAuthInfo
    issues: list[str] = field(default_factory=list)
```

Field ordering: `hf_auth` immediately before `issues` so JSON output groups it with the other structured "what we found" blocks before the human-readable issue list.

**Population helper** in `diagnostics.py`:

```python
def _hf_auth_info() -> HuggingFaceAuthInfo:
    """Local-only probe of the HF token sources. Mirrors resolve_hf_token's
    order but reports the source, not the token value. Never calls login(),
    never hits the network."""
    if os.environ.get("HF_TOKEN"):
        return HuggingFaceAuthInfo(token_source="env", has_token=True)
    import huggingface_hub
    if huggingface_hub.get_token():
        return HuggingFaceAuthInfo(token_source="cache", has_token=True)
    return HuggingFaceAuthInfo(token_source="none", has_token=False)
```

We deliberately do not call `resolve_hf_token` from `_hf_auth_info` — we need the *source*, not the token. The two functions share probe order but report different things. Pull out a shared private helper later only if a third caller appears.

**Issues integration.** In `run_doctor`, after populating `hf_auth`, append to `issues` when `hf_auth.token_source == "none"`:

```text
"no HuggingFace token found; gated repos like facebook/sam3.1 will not download (set HF_TOKEN or run `huggingface-cli login`)"
```

**Renderer change** in `cli/doctor_cmd.py::_render_table`. Add a "HuggingFace auth" sub-table immediately after the "SAM 3.1 weights" sub-table (so all model-fetching context is co-located):

```python
hf = report.hf_auth
auth = Table(title="HuggingFace auth", show_header=False, box=None)
auth.add_row("token source", hf.token_source)
auth.add_row("has token", str(hf.has_token))
console.print(auth)
```

`--json` output picks up the new field automatically via `dataclasses.asdict`.

### 4.3 `models/sam3.py::_resolve_checkpoint_path` — auto-download on miss

Current body (`src/esam3/models/sam3.py:166-179`) raises `FileNotFoundError` when the checkpoint file isn't on disk. Replace the bare raise with an auto-download then re-check. Sketch (final exact code is the planner's):

```python
def _resolve_checkpoint_path(cfg: ModelConfig) -> Path:
    if cfg.local_dir is None:
        raise FileNotFoundError(
            f"ModelConfig.local_dir is None. Set it to a directory for "
            f"{cfg.checkpoint_file}, or run `esam3 init` to scaffold one."
        )
    local_dir = Path(cfg.local_dir)
    path = local_dir / cfg.checkpoint_file
    if path.exists():
        return path
    logger.info(
        "SAM 3.1 checkpoint missing at %s; auto-downloading from %s",
        path,
        cfg.name,
    )
    download_model(cfg.name, local_dir, revision=cfg.revision)
    if not path.exists():
        raise FileNotFoundError(
            f"Downloaded {cfg.name} into {local_dir} but {cfg.checkpoint_file} "
            f"is still missing. Check that the repo contains that file."
        )
    return path
```

**Why this shape.**

- **No new config flag.** Auto-download on miss is unconditional. If the user opts out, they do so at the `esam3 init` level (the prompt) or by setting `cfg.model.local_dir` to a directory that already has the file. Adding an `auto_download` knob would surface a fourth way to fail ("I set the flag and forgot why") without buying anything — the file-presence check is already the gate.
- **`ModelConfig.revision` is the knob.** It already exists in the schema (`schema.py:42`); pass it through to `snapshot_download(revision=...)`. Users who want a pinned revision set it in their config.
- **The strict re-check is the consumer's contract.** `download_model`'s skip condition (`local_dir` non-empty) is intentionally weak — it allows the directory to be populated by some other process. The consumer who knows the expected filename (`cfg.checkpoint_file`) does the file-level check after the call and raises a precise error if the file is still absent post-download (e.g., the repo's contents changed, the user pinned a revision without the file).
- **Idempotent.** A second call with the file present short-circuits at `path.exists()`. A second call with the directory non-empty but the file missing falls through to `download_model`, which short-circuits on its own skip condition, and then we raise the precise "downloaded but file still missing" error — no infinite re-download loop.

---

## 5. Tests

All new tests are unit-tier and CPU-only. Real-Hub integration tests are out of scope. Coverage gate stays at 80% (`pyproject.toml` `[tool.pytest.ini_options].addopts`); new code under `src/esam3/utils/` and the wired surfaces should land well above that based on the test plan below.

### 5.1 `tests/unit/test_utils_huggingface.py` (new)

Direct tests of both functions.

**`resolve_hf_token`:**

| Case | Mechanism |
| --- | --- |
| Explicit arg wins | Pass `token="explicit"`, monkeypatch `HF_TOKEN=env-token`, monkeypatch `huggingface_hub.get_token` → `"cache-token"`. Assert `== "explicit"`. |
| Env wins over cache | No arg, `HF_TOKEN=env-token`, cache returns `"cache-token"`. Assert `== "env-token"`. |
| Cache used when no arg/env | No arg, `HF_TOKEN` unset (`monkeypatch.delenv("HF_TOKEN", raising=False)`), cache returns `"cache-token"`. Assert `== "cache-token"`. |
| Returns `None` when nothing | No arg, no env, cache returns `None`. Assert `is None`. |

**`download_model`:**

| Case | Mechanism |
| --- | --- |
| Skip when `local_dir` non-empty | Pre-create `tmp_path / "models" / "sentinel.txt"`. Call `download_model("repo", tmp_path / "models")`. Assert `snapshot_download` mock NOT called. |
| Calls snapshot_download when missing | Empty `tmp_path / "models"` (or non-existent). Assert mock called with `repo_id="repo"`, `local_dir=str(tmp_path / "models")`, and the planner-verified "real files" kwarg. |
| Honors `force=True` | Pre-populate dir, pass `force=True`. Assert mock IS called. |
| Honors `revision` | Pass `revision="v1.0"`. Assert mock called with `revision="v1.0"`. |
| Token passed through | Monkeypatch `HF_TOKEN="env-tok"`. Assert mock called with `token="env-tok"`. |
| Token never appears in raised message | Force `snapshot_download` to raise `GatedRepoError`. Catch the wrapped `RuntimeError`; assert `"env-tok"` NOT in `str(exc)`. Repeat for `RepositoryNotFoundError` and `HfHubHTTPError`. |
| Maps `GatedRepoError` → `RuntimeError` with remediation | Assert `"gated"` or `"accept the license"` substring, and the repo id, and the `HF_TOKEN` hint, all present. |
| Maps `RepositoryNotFoundError` → `RuntimeError` | Assert repo id and "not found" / "access" substring. |
| Maps generic `HfHubHTTPError` → `RuntimeError` | Assert repo id and "Hub request failed" substring. |
| Other exception types propagate | Force `snapshot_download` to raise `OSError("network down")`. Assert `OSError` reaches the caller (not wrapped). |
| Logs the fetch line without token | Capture log; assert `"fetching repo →"` appears, `"env-tok"` does not. |

All HF calls are mocked: `monkeypatch.setattr("esam3.utils.huggingface.huggingface_hub.snapshot_download", fake)` (or whatever import shape the implementation uses; the planner pins the patch target).

### 5.2 `tests/unit/test_cli_init.py` (extend)

Existing tests (template-writes, force, unknown-template) stay unchanged. New tests:

| Test | Mechanism |
| --- | --- |
| `test_init_skips_when_no_download_weights_flag` | Pass `--no-download-weights`; monkeypatch `esam3.cli.init_cmd.download_model`; assert mock NOT called and the hint text appears in stdout. |
| `test_init_short_circuits_when_weights_present` | Touch `<cfg.model.local_dir>/<cfg.model.checkpoint_file>` before running init; assert `"skipping download"` in stdout and `download_model` mock NOT called. (The template's `local_dir` is `models/sam3.1` by default; the test overrides via `--output` written to `tmp_path` and then materializes the resolved checkpoint path under `tmp_path` — planner pins how, likely by `monkeypatch.chdir(tmp_path)`.) |
| `test_init_non_tty_skips_with_hint` | Force `sys.stdin.isatty` to return `False`; no flag passed; assert `download_model` NOT called and hint in stdout. |
| `test_init_download_weights_yes_triggers_download` | Pass `--download-weights --yes`; monkeypatch `download_model`; assert called once with `(cfg.model.name, Path(cfg.model.local_dir), revision=cfg.model.revision)`. |
| `test_init_download_failure_propagates_as_exit_1` | Monkeypatch `download_model` to raise `RuntimeError("gated")`; pass `--download-weights --yes`; assert exit code != 0 and `"gated"` in stderr. |

### 5.3 `tests/unit/test_cli_doctor.py` (extend)

| Test | Mechanism |
| --- | --- |
| `test_doctor_table_includes_hf_auth_section` | Run `esam3 doctor`; assert `"HuggingFace auth"` or `"token source"` in plaintext output. |
| `test_doctor_reports_env_token_source` | `monkeypatch.setenv("HF_TOKEN", "env-tok")`; run `esam3 doctor --json`; assert `blob["hf_auth"]["token_source"] == "env"` and `has_token is True`. |
| `test_doctor_reports_cache_token_source` | `monkeypatch.delenv("HF_TOKEN", raising=False)`; monkeypatch `huggingface_hub.get_token` → `"cache-tok"`; assert `token_source == "cache"`. |
| `test_doctor_reports_no_token` | No env, monkeypatch `huggingface_hub.get_token` → `None`; assert `token_source == "none"` AND the `"no HuggingFace token found"` issue appears in `blob["issues"]`. |
| `test_doctor_json_includes_hf_auth_field` | Bare `esam3 doctor --json`; `json.loads(stdout)` has `"hf_auth"` key with `token_source` and `has_token` sub-keys. |

Doctor must never call any networked HF API in these tests; the patches are local-only.

### 5.4 `_resolve_checkpoint_path` (new tests)

There is no existing test for `_resolve_checkpoint_path` (verified by grep against `tests/`). The planner creates one in the file that already imports `esam3.models.sam3` — most likely `tests/unit/test_sam3_wrapper.py`, or a new `tests/unit/test_sam3_checkpoint_resolve.py` if the wrapper test file is already large. Three cases:

| Test | Mechanism |
| --- | --- |
| `test_resolve_checkpoint_auto_downloads_on_miss` | `cfg = ModelConfig(name="facebook/sam3.1", local_dir=str(tmp_path), checkpoint_file="ckpt.pt")`. Monkeypatch `esam3.models.sam3.download_model` to side-effect-touch `tmp_path / "ckpt.pt"`. Call `_resolve_checkpoint_path(cfg)`; assert mock called once with `("facebook/sam3.1", Path(str(tmp_path)), revision=None)`; assert returned path == `tmp_path / "ckpt.pt"`. |
| `test_resolve_checkpoint_raises_when_download_leaves_file_missing` | Same setup, but `download_model` mock does NOT create the file. Assert `FileNotFoundError` with `"still missing"` substring. |
| `test_resolve_checkpoint_local_dir_none` | `cfg = ModelConfig(local_dir=None)`; assert `FileNotFoundError` with `"local_dir is None"` and the new `"esam3 init"` hint. |

No real `huggingface_hub` calls; `download_model` is the mock boundary. (The minimum the planner spec-asserts is that `download_model` is patched at the import-into-`sam3` site, i.e. `esam3.models.sam3.download_model`, not at the source `esam3.utils.huggingface.download_model` — `_resolve_checkpoint_path` will do `from esam3.utils.huggingface import download_model` at module top, binding the name in the consumer's namespace.)

### 5.5 Coverage

The 80% project gate is unchanged. The new module `src/esam3/utils/huggingface.py` is small (≈80 lines) and entirely test-covered by §5.1; the three wired surfaces add coverage in the existing files. No coverage exemption needed.

---

## 6. Dependency Change

`pyproject.toml` change:

```toml
[project]
dependencies = [
  # ... existing entries ...
  "huggingface-hub>=0.30",  # planner pins floor against uv.lock (currently 1.15.0)
]
```

**Floor rationale.** `uv.lock` resolves `huggingface-hub==1.15.0` today (verified). The functions we use (`get_token`, `snapshot_download` with the materialize-real-files semantics, the `errors` module) have been stable since well before 1.x — but to keep the floor honest, the planner picks a sensible lower bound that:

- Includes `huggingface_hub.get_token()` at the top level (v0.19+).
- Includes `snapshot_download` with the modern kwarg surface the implementation actually uses.

A pragmatic floor is `>=0.30` (lines up with the floors `transformers` and `datasets` already enforce transitively), but the planner verifies against `1.15.0` and picks the lowest floor that doesn't constrain our actual API surface. Optional: pin tighter (`>=1.0`) if 1.x renamed something we rely on; the planner decides during implementation.

**Why not an optional extra.** `huggingface-hub` is already pulled transitively by `transformers`, `datasets`, and `sam3`. Making it explicit costs nothing at install time and lets us import it unconditionally from `utils/huggingface.py` and `diagnostics.py` without an `ImportError` guard. The two existing optional extras (`wandb`, `qlora`, `tensorboard`) gate things that *aren't* already transitive; HF Hub is not in that category.

**`[tool.mypy.overrides]` change.** Not needed — `huggingface_hub` ships type stubs in modern releases. If `mypy --strict` complains, the planner adds it to `ignore_missing_imports` alongside `peft.*`, `datasets.*`, etc.

---

## 7. Risks & Open Questions

| Risk | Mitigation / planner action |
| --- | --- |
| `snapshot_download` symlink-vs-real-file kwarg differs between hub 0.x and 1.x. | Planner verifies the exact kwarg name in `huggingface_hub==1.15.0` and uses it (or drops it if real files are now the default). Land a comment in the source citing the version checked against. |
| `huggingface_hub.errors` module path may have moved between minors. | Planner verifies and pins the correct import. If it has moved, prefer importing from the top-level `huggingface_hub` if those names are re-exported there. |
| `huggingface_hub.get_token()` rename/removal. | Planner verifies it exists in 1.15.0. If only `HfFolder.get_token()` is available in older releases, the floor we set keeps us safely on the modern API. |
| HF Hub becomes a hard dep — bigger install surface for users who only use a local cache. | Accepted. It's already transitive via `transformers`/`datasets`/`sam3`; codifying it doesn't grow the wheel set. |
| Full snapshot of `facebook/sam3.1` is multi-GB; users hit disk quotas. | Surfaced in the `esam3 init` confirmation prompt: `"Download ... weights into {local_dir}? (this can be several GB)"`. No download-size estimate (the Hub API doesn't cheaply provide one); the prompt text is the user's warning. |
| User's HF token leaks via a logger or stack trace. | Verified by §5.1: token is never embedded in the "fetching" log line, never embedded in any of the three mapped `RuntimeError` messages. Other exception types pass through unchanged — if a future Hub release adds a class we don't catch and its `__str__` happens to include the token, the test suite won't catch it. Documented as a residual risk; a stronger guard (e.g., a `try/except BaseException` wrapper that redacts) is deferred. |
| `esam3 init` runs `load_config` on the freshly-written template. Templates have placeholder data paths (`data/train.json`, etc.) that don't exist on disk. | `load_config` does not stat data paths (verified: `DataSplit` enforces only `min_length=1`); this works today and is leveraged by the existing init tests. |
| Mocking `huggingface_hub.snapshot_download` at the right import path. | The implementation imports `huggingface_hub` at module top of `utils/huggingface.py`; tests should `monkeypatch.setattr("esam3.utils.huggingface.huggingface_hub.snapshot_download", ...)`. Planner pins the exact patch target. |

---

## 8. Out of Scope (deferred)

- `auto_download: bool` config knob (unconditional in `_resolve_checkpoint_path`).
- `esam3 hf` / `esam3 login` CLI subcommand.
- Real-Hub integration tests (everything mocked).
- Upload / push-to-Hub.
- Custom HF endpoint (`HF_ENDPOINT` env var) support — `huggingface_hub` honors it natively; we don't surface it.
- Multi-repo or dataset downloads.
- A token-redaction wrapper around arbitrary exception messages (current scope only redacts the three mapped error classes).
- Download-size estimate in the `esam3 init` prompt (no cheap API).

---

## 9. Exit Criteria

- `src/esam3/utils/__init__.py` and `src/esam3/utils/huggingface.py` exist; `resolve_hf_token` and `download_model` match the contracts in §3.2.
- `src/esam3/models/sam3.py::_resolve_checkpoint_path` auto-downloads on miss and re-checks (§4.3).
- `src/esam3/cli/init_cmd.py` exposes `--download-weights/--no-download-weights` and `--yes`; prompt-on-miss flow matches §4.1.
- `src/esam3/diagnostics.py` `DoctorReport` has `hf_auth: HuggingFaceAuthInfo`; `run_doctor` populates it without network calls; `_render_table` renders a "HuggingFace auth" sub-table.
- `pyproject.toml` `[project] dependencies` lists `huggingface-hub` with a verified floor.
- All new and extended tests pass; existing tests continue to pass.
- `ruff check`, `mypy --strict`, `pytest` green at coverage ≥ 80%.
- No real-Hub network calls in any test (CI-safe).
