# Minimal GPU Testing on a GTX 1080 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **AMENDED 2026-05-24 — gradient checkpointing abandoned.** The Phase-0 diagnostic on the real GTX 1080 proved the `CheckpointError` is a structural save-count divergence INSIDE sam3's `multi_head_attention_forward` recompute, unfixable without editing forbidden sam3 source (Fix A/A+/B all fail; Fix C corrupts the backward). See `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` ("Phase-0 trace + fix classification"). **Consequences for this plan:** Task **C-2** (Phase-0 diagnostic + Phase-1 GC fix) is **abandoned**; Task **C-4** (Phase-3 verification + 8 GB calibration) and Task **C-3**'s deliverables (`gpu_smoke_qlora_8gb.yaml`, `tests/gpu/test_real_train_qlora_8gb.py`) are **deferred to #137**. Task **C-1** (`use_double_quant`) is GC-independent and **kept**. PR #127 closed unmerged; #89/#60 not-planned; `origin/main` already ships GC OFF (nothing to revert); the GC-free rebuild of this branch removes `vit_act_checkpoint.py` and `tests/gpu/test_grad_checkpointing.py`. The corrected GPU-tagged inventory is **12** files (see the File map / D-1). The `gpu_local` **training** tier is **PROVISIONAL pending #137**; its baseline members are forward-only / inspection tests.

**Goal:** Make the dev box's live GTX 1080 (sm_61, ~7 GB effective VRAM) a first-class **GPU-test target** — provision its sm_61 kernels, enable CC-6.0/float16 code paths — and rationalize the GPU test suite into a three-tier hardware taxonomy — all in one PR. (Gradient checkpointing is abandoned; see the amendment note above.)

**Architecture:** An opt-in `gpu-pascal` uv extra resolves a cu118 torch + bitsandbytes (isolated from the default cu130 install via uv explicit-index + conflicting-extras). A device-aware dtype-coercion helper widens `bfloat16 → float16` on CC<8.0 hardware at two seams (autocast + QLoRA compute dtype). Three mutually-exclusive pytest hardware-tier markers (`gpu_local`, `gpu_t4`, `gpu_xl`) replace the retired cost markers (`gpu`, `gpu_inspection`); a conftest autoskip gates each tier against live hardware. The GC fix originally planned for the flag-flip-only `vit_act_checkpoint` patch is **abandoned** (the Phase-0 diagnostic proved it infeasible without editing forbidden sam3 source); the 8 GB QLoRA *training* recipe is **deferred to #137**. The GC-independent `use_double_quant` field (C-1) lands. The #117 audit reclassifies every surviving GPU test into a tier (calibrated on the 1080) and moves CPU-testable cases to CPU; the #116 notebook matrix closes coverage gaps.

**Tech Stack:** Python 3.12, PyTorch (`torch.utils.checkpoint`, `torch.autocast`, `torch.cuda.*`), bitsandbytes ≥ 0.43 (`Linear4bit` NF4), Meta `sam3` (external, NOT importable on the dev box — never edited), `pytest` (`-m gpu_local|gpu_t4|gpu_xl`), `ruff`, `mypy`, Pydantic (`TrainConfig`), `uv` (cu118 explicit index + conflicting extras).

**Reference spec (source of truth):** `docs/superpowers/specs/2026-05-24-min-gpu-testing-gtx1080-design.md`
**GC abandonment evidence:** `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` ("Phase-0 trace + fix classification"). The GC fix-taxonomy spec (`docs/superpowers/specs/2026-05-23-gradient-checkpointing-t4-design.md`) is now historical context only — the fix it describes was attempted on the 1080 and abandoned.

---

## Execution environment (read before Task 1)

This plan executes **on the dev box** — the WSL2 machine that holds the live GTX 1080. Verified at plan time:

- `nvidia-smi` reports `NVIDIA GeForce GTX 1080, 8192 MiB, driver 582.28, compute_cap 6.1`.
- The default `uv sync` resolves `torch 2.12.0+cu130` (no sm_61 cubin) — `torch.cuda.is_available()` is `True` but **no kernel runs on the 1080 today**.

**Guiding principle (spec §1):** *any testing runnable on this branch gets run here.* After `uv sync --extra gpu-pascal`, real 1080 runs are in-session. The §4.3 milestone and the `gpu_local` tier classification (forward-only / inspection) are **real executable steps with captured output**, not deferred handoffs. (The Phase-0 GC diagnostic was likewise run in-session — it is what established the GC abandonment; the 8 GB *training* calibration is deferred to #137.)

**The worktree root** is `/home/justin/projects/custom-sam-peft/.claude/worktrees/feat+min-gpu-testing`. All paths below are absolute against it; the constant `$WT` is used as shorthand in commands:

```bash
WT=/home/justin/projects/custom-sam-peft/.claude/worktrees/feat+min-gpu-testing
```

**Test env note (from project memory):** the worktree venv needs `uv sync --extra dev` before tests run; the 80% coverage gate only passes on the **FULL** pytest suite (`addopts` carries `--cov-fail-under=80`), not `tests/unit` alone. Per-file/subset pytest runs in this plan therefore pass `--no-cov` to avoid a spurious coverage-gate failure on a partial collection.

---

## Pre-flight checks

Run once before Task 1. STOP and re-derive anchors if any fails.

> **AMENDMENT NOTE:** the GC-specific anchors below describe the original
> pre-implementation tree and **no longer hold** after the GC abandonment / the
> GC-free rebuild: `vit_act_checkpoint.py` is removed (it has no anchor); and
> `use_double_quant` HAS landed (C-1) with `_QLORA_META_VERSION = 2`. The GC
> assertions are commented out below. Re-derive against the actual rebuilt tree
> before re-running any task.

```bash
WT=/home/justin/projects/custom-sam-peft/.claude/worktrees/feat+min-gpu-testing
# 1. In the worktree, clean tree.
git -C "$WT" rev-parse --show-toplevel    # Expected: $WT
git -C "$WT" status --porcelain           # Expected: no output

# 2. Dev deps present (memory: worktree venv needs --extra dev).
cd "$WT" && uv sync --extra dev

# 3. sam3 NOT importable on the dev box (fix is monkeypatch-only).
cd "$WT" && uv run python -c "import sam3" 2>&1 | tail -1
# Expected: ModuleNotFoundError / ImportError.

# 4. CPU baseline green (record counts + the coverage % — Task D-3 compares against it).
cd "$WT" && uv run pytest -q 2>&1 | tail -8
# Expected: all pass; "Required test coverage of 80% reached" (or higher).

# 5. The 1080 is live and torch is cu130 today.
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader   # Expected: NVIDIA GeForce GTX 1080, 6.1
cd "$WT" && uv run python -c "import torch; print(torch.__version__)"   # Expected: 2.12.0+cu130

# 6. Confirm the cited live-code anchors still match the spec.
cd "$WT" && uv run python - <<'PY'
import pathlib
def line(p, n): return pathlib.Path(p).read_text().splitlines()[n-1]
# GC ABANDONED: vit_act_checkpoint.py is removed by the GC-free rebuild — no anchor.
conf = pathlib.Path("tests/conftest.py").read_text()
assert "(7, 5)" in conf and "gpu_inspection" in conf
loop = pathlib.Path("src/custom_sam_peft/train/loop.py").read_text()
assert "def _autocast_ctx" in loop and "disables_outer_autocast" in loop
pres = pathlib.Path("src/custom_sam_peft/presets.py").read_text().splitlines()
assert 'dtype: Literal["bfloat16"]' in pres[82]      # line 83
assert 'dtype="bfloat16"' in pres[334]               # line 335
assert "bf16 —" in pres[119] or "bf16" in pres[119]  # line 120 label token
schema = pathlib.Path("src/custom_sam_peft/config/schema.py").read_text()
assert 'Dtype = Literal["bfloat16", "float16"]' in schema   # Dtype already widened
assert 'Optimizer = Literal["adamw", "adamw8bit", "auto"]' in schema
assert "class QLoRAConfig" in schema
# C-1 LANDED: use_double_quant now present; metadata version bumped to 2.
ql = pathlib.Path("src/custom_sam_peft/peft_adapters/qlora.py").read_text()
assert "bnb.nn.Linear4bit(" in ql
print("anchors OK")
PY
# Expected: anchors OK
```

---

## File map (what gets touched)

| File | Action | Owning task | Workstream |
| --- | --- | --- | --- |
| `pyproject.toml` | Modify — add `gpu-pascal` extra + `[[tool.uv.index]]` (cu118, explicit) + `[tool.uv.sources]` + conflicting-extras; register 3 tier markers; retire `gpu`/`gpu_inspection` markers | A-1, B-4 | A, B |
| `docs/testing/local-pascal-gpu-testing.md` | Create — how to provision `gpu-pascal`, run `gpu_local`, float16 caveat | A-1, A-2 | A, Docs |
| `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` | Create — §4.3 milestone evidence (PASS) + the Phase-0 trace+classification that established the **GC abandonment**. (No Phase-3 GC calibration — deferred to #137.) | A-2 | A, Docs |
| `src/custom_sam_peft/runtime/_runtime.py` | Modify — add device-aware `coerce_dtype_for_capability` helper (or new module — see B-2) | B-2 | B |
| `src/custom_sam_peft/train/loop.py` | Modify — `_autocast_ctx` routes through the coercion helper (line ~191) | B-2 | B |
| `src/custom_sam_peft/peft_adapters/qlora.py` | Modify — `_replace_with_bnb_linear4bit` coerces compute dtype on CC<8.0 (line ~142); honor `use_double_quant` in the `Linear4bit` ctor (line ~146); bump metadata version if shape changes | B-2, C-1 | B, C |
| `src/custom_sam_peft/presets.py` | Modify — widen `PresetDecision.dtype` Literal (line 83); CC-aware dtype in `decide_preset` (lines 295/335); `label()` renders real token (line 120); byte-comment (line 142) | B-3 | B |
| `tests/conftest.py` | Modify — lower CC floor to (6,0); rewrite skip reasons; register 3 tier markers in `pytest_configure`; per-tier hardware autoskip in `pytest_collection_modifyitems` | B-1, B-4 | B |
| `scripts/run_gpu_tests.sh` | Modify — accept `{local,t4,xl}`; add `tests/predict/`; fix header counts; preserve `--deselect`/CI guard | B-4 | B, E |
| `src/custom_sam_peft/config/schema.py` | Modify — add `QLoRAConfig.use_double_quant: bool = False` (line ~488) | C-1 | C |
| ~~`src/custom_sam_peft/models/_patches/vit_act_checkpoint.py`~~ | **GC abandoned (C-2 dropped); removed by the GC-free rebuild** | — | — |
| ~~`configs/examples/gpu_smoke_qlora_8gb.yaml`~~ | **8 GB training recipe deferred to #137 — not created on this branch** | — | — |
| ~~`tests/gpu/test_real_train_qlora_8gb.py`~~ | **8 GB training smoke deferred to #137 — not created on this branch** | — | — |
| ~~`tests/gpu/test_grad_checkpointing.py`~~ | **GC-branch test; removed by the GC-free rebuild** | — | — |
| `tests/unit/test_qlora.py` (or sibling) | Modify/Create — CPU test for `use_double_quant` wiring + metadata round-trip | C-1 | C |
| `tests/unit/test_dtype_coercion.py` | Create — CPU TDD for the coercion helper | B-2 | B |
| `tests/unit/test_marker_autoskip.py` | Create — CPU TDD for tier marker registration + collection autoskip | B-4 | B |
| `tests/unit/test_run_gpu_tests_script.py` | Create — CPU TDD for runner tier→marker/path parsing | B-4 | B, E |
| `tests/gpu/*.py`, `tests/integration/test_*_real.py`, `tests/predict/test_gpu_predict.py` | Modify — swap legacy `gpu`/`gpu_inspection` markers for the assigned tier marker | D-2 | D |
| `docs/testing/gpu-audit-2026-05-24.md` | Create — the #117 per-test tier+decision+rationale audit (or a section in gpu-test-policy.md — D-1 decides) | D-1, D-2 | D |
| `docs/testing/gpu-test-policy.md` | Modify — CC 6.0 floor; float16-on-Pascal; three-tier taxonomy (cost demoted to guidance); refresh inventory to the **12** GC-free GPU-tagged tests; fix stale "12 tests"/per-tier counts | D-1, E-1 | D/E, Docs |
| `notebooks/colab_gpu_tests.ipynb` | Modify — tier cells (`local`/`t4`/`xl`); add cells for the 6 unreferenced tests; **drop the GC Phase-0/Phase-3 cells** (abandoned work); coverage-matrix markdown | E-1 | E |

**CPU-testable (TDD-able without the 1080):** B-1 (CC-floor logic via injected capability), B-2 (dtype-coercion helper), B-3 (preset dtype widening + `label()` token, capability passed as a value), B-4 (marker registration/collection autoskip logic, runner-script tier parsing), C-1 (`use_double_quant` field + metadata round-trip). Apply `superpowers:test-driven-development` (failing test first) for all of these.

**REQUIRE the live 1080:** A-2 (the §4.3 hard-gate milestone — sm_61 kernel + bnb `Linear4bit` forward), and the D-1 tier classification of every `gpu_local` candidate (a test is `gpu_local` only if its forward-only / inspection run fits within ~7 GB on the real card). (C-2/C-4 — the GC diagnostic and 8 GB calibration — are no longer plan tasks: GC abandoned, 8 GB training deferred to #137. The Phase-0 diagnostic that established the abandonment is already recorded in the manual-pass doc.)

**Parallelizable (file-disjoint, no shared state) for an orchestrator:** within D, the CPU-move refactors of disjoint test files can run in parallel once D-1's audit fixes the decisions; E's per-test notebook-cell authoring is disjoint from D's source edits. **Serialize** anything touching `tests/conftest.py`, `scripts/run_gpu_tests.sh`, `docs/testing/gpu-test-policy.md`, or `pyproject.toml` (B-1/B-4/D-2/E-1 all touch one or more of these — run them in workstream order).

**Lint/format gate** — run before EVERY commit that lands on the ready PR (exempt: none here, this is a feature PR):

```bash
cd "$WT" && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

Per `superpowers:verification-before-completion`: a step is "done" only when you have SHOWN the command output, not asserted it.

---

# Workstream A — Pascal environment + the hard-gate milestone

> Spec §4. **A-2 is the HARD GATE: nothing in B/C/D-`gpu_local`-calibration proceeds until A-2 passes (or its §4.3 fallback is invoked and documented).**

## Task A-1: Add the `gpu-pascal` uv extra (cu118 + bitsandbytes), isolated from cu130

**Workstream A. CPU-testable (resolution-only) — no 1080 kernel run here.** Adds the opt-in extra + uv index/sources/conflict tables. Spec §4.2.

**Difficulty:** Medium (uv table shapes). **Subagent:** implementer (sonnet/high).

**Files:**
- Modify: `pyproject.toml`
- Create (stub now, fill in A-2): `docs/testing/local-pascal-gpu-testing.md`

**Context:** `pyproject.toml` today has `torch>=2.4` (line 10, no CUDA index), `[project.optional-dependencies]` with `wandb/qlora/tensorboard/jupyter/dev` (lines 30-42), and **no `[tool.uv]`/`[[tool.uv.index]]`/`[tool.uv.sources]` blocks** (only `[dependency-groups]` at line 146). The contract (spec §4.2): bare `uv sync` → cu130 (unchanged); `uv sync --extra gpu-pascal` → cu118 torch + bitsandbytes; `uv sync --extra dev` → unchanged.

- [ ] **Step 1: Add the `gpu-pascal` extra.**

In `pyproject.toml` `[project.optional-dependencies]` (after the `qlora` line 32), add:

```toml
gpu-pascal = ["torch", "bitsandbytes>=0.43"]
```

(Listing bare `torch` here lets `[tool.uv.sources]` route it to the cu118 index under this extra.)

- [ ] **Step 2: Add the uv index + sources + conflict tables.**

Append a `[tool.uv]` section group (place it after the existing `[dependency-groups]` block, ~line 149):

```toml
[[tool.uv.index]]
# PyTorch cu118 wheels — covers sm_60..sm_90; PTX from compute_60 JIT-compiles
# to the GTX 1080's sm_61 at first kernel launch (research #79). Explicit so it
# is consulted ONLY for packages routed to it (the default cu130 resolution is
# untouched).
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
# Route torch to cu118 ONLY under the gpu-pascal extra. Outside it, torch
# resolves from PyPI (cu130) as today.
torch = [{ index = "pytorch-cu118", extra = "gpu-pascal" }]

[tool.uv]
# gpu-pascal (cu118) and the default cu130 torch must never co-resolve.
conflicts = [[{ extra = "gpu-pascal" }]]
```

> **Implementer note:** the exact uv table shapes are an implementation detail (spec §4.2). uv's conflicting-extras syntax is `conflicts = [[ {extra = "a"}, {extra = "b"} ]]` for mutually-exclusive *pairs*; for a single extra that must not co-resolve with the default torch, the cleanest expression may instead be the `[tool.uv.sources]` `extra=` routing alone (which already scopes cu118 to the extra). If `uv` rejects a single-element conflict set, drop the `[tool.uv]` conflicts block and rely on the source-routing scope — the **contract in Step 3 is the gate**, not the table shape. Iterate the tables until Step 3 passes.

- [ ] **Step 3: Verify the resolution contract (no kernel run yet).**

```bash
cd "$WT"
# Default install still cu130.
uv sync 2>&1 | tail -3
uv run python -c "import torch; print(torch.__version__)"   # Expected: 2.12.0+cu130 (a +cu130 wheel)
# Dev path unchanged.
uv sync --extra dev 2>&1 | tail -2
# gpu-pascal resolves cu118 + bitsandbytes.
uv sync --extra gpu-pascal 2>&1 | tail -5
uv run python -c "import torch; print(torch.__version__)"   # Expected: a +cu118 wheel
uv run python -c "import bitsandbytes; print('bnb', bitsandbytes.__version__)"
```

Expected: bare/dev resolve `+cu130`; `--extra gpu-pascal` resolves a `+cu118` torch and installs bitsandbytes. **Capture this output for the manual-pass record (A-2).**

- [ ] **Step 4: Stub the Pascal testing doc.**

Create `docs/testing/local-pascal-gpu-testing.md` with the provisioning + run instructions (the milestone evidence section is filled in A-2):

```markdown
# Local Pascal (GTX 1080) GPU Testing

The dev box holds a GTX 1080 (compute capability 6.1 / sm_61, 8 GB VRAM, ~7 GB
effective after WSL/Xwayland overhead). It is a **GPU-test target**, not a
training/inference platform: it exercises real GPU code paths (sm_61 kernels,
bitsandbytes 4-bit, the gradient-checkpointing fix, float16 dtype handling).

## Provision

The default `uv sync` installs cu130 torch (no sm_61 cubin). To reach the 1080:

    uv sync --extra gpu-pascal   # cu118 torch (sm_60..sm_90 + PTX) + bitsandbytes

This extra is isolated via a uv explicit index + extra-scoped source routing, so
the bare `uv sync` and `uv sync --extra dev` paths are unchanged (still cu130).

## Run the gpu_local tier

    bash scripts/run_gpu_tests.sh local

Or directly: `uv run pytest -m gpu_local tests/gpu/ tests/integration/ tests/predict/`.

## float16 caveat (Pascal has no fast bf16)

bf16 is **emulated** below compute capability 8.0, so the 1080 trains/runs in
**float16**. A `bfloat16` request is coerced to `float16` with a one-time
warning (see `coerce_dtype_for_capability`). This means numerics validated on
the 1080 do NOT certify the bf16 T4 release path — that confirmation is a
follow-up (gpu_t4 tier).

## Milestone evidence

<!-- A-2 fills this in: the §4.3 sm_61-kernel + bnb-Linear4bit proofs. -->
```

- [ ] **Step 5: Lint gate + commit.**

```bash
cd "$WT" && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
git add pyproject.toml docs/testing/local-pascal-gpu-testing.md
git commit -m "feat(deps): add opt-in gpu-pascal extra (cu118 torch + bitsandbytes) isolated from cu130"
```

**Completion criteria:** bare/dev `uv sync` → cu130; `--extra gpu-pascal` → cu118 + bitsandbytes; lint gate green. Resolution output captured.

---

## Task A-2: HARD-GATE MILESTONE — prove sm_61 + bnb `Linear4bit` on the real 1080

**Workstream A. REQUIRES THE LIVE 1080. This task GATES all of B/C and D's `gpu_local` calibration (spec §4.3, §4.5).** Run in-session on the dev box after `uv sync --extra gpu-pascal`. Its evidence (commands + output) is recorded in the manual-pass record. Spec §4.3.

**Difficulty:** Hard (real-hardware proof; branch-on-failure). **Subagent:** implementer (sonnet/high) — runs the proofs in-session and records output; escalate to the §4.3 fallback path if either proof fails.

**Files:**
- Create: `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` (mirror the structure of `docs/testing/manual-gpu-pass-2026-05-19.md`)
- Modify: `docs/testing/local-pascal-gpu-testing.md` (fill the "Milestone evidence" section)

- [ ] **Step 1 (RUN ON THE 1080): Prove a real torch CUDA kernel executes on sm_61 under cu118.**

```bash
cd "$WT" && uv sync --extra gpu-pascal
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
c = a @ b                      # forces a CUDA matmul kernel launch -> PTX JIT compute_60 -> sm_61
torch.cuda.synchronize()
ref = (a.cpu() @ b.cpu())
err = (c.cpu() - ref).abs().max().item()
print("matmul max abs err", err)
assert err < 1e-2, "sm_61 matmul produced wrong results"
print("SM_61 KERNEL OK")
PY
```

Expected: `cc (6, 1)`, a small `max abs err`, and `SM_61 KERNEL OK` — **no** `CUDA error: no kernel image is available for execution on the device`.

- [ ] **Step 2 (RUN ON THE 1080): Prove a bnb `Linear4bit` (NF4) forward runs on the 1080 under float16.**

```bash
cd "$WT" && uv run python - <<'PY'
import torch, bitsandbytes as bnb
print("bnb", bnb.__version__)
lin = bnb.nn.Linear4bit(256, 128, bias=False, quant_type="nf4", compute_dtype=torch.float16)
lin = lin.to("cuda")           # quantization fires on .to(cuda)
x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
y = lin(x)                     # NF4 4-bit kernel forward on sm_61
torch.cuda.synchronize()
print("out", tuple(y.shape), y.dtype, "finite", bool(torch.isfinite(y).all()))
assert y.shape == (4, 128) and torch.isfinite(y).all()
print("BNB LINEAR4BIT OK")
PY
```

Expected: `out (4, 128) torch.float16 finite True` and `BNB LINEAR4BIT OK` — no kernel-image / unsupported-CC error from bitsandbytes.

- [ ] **Step 3: Record the evidence in the manual-pass record.**

Create `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` mirroring `manual-gpu-pass-2026-05-19.md`'s structure (How to run / Test checklist / Session log). Include a "§4.3 milestone" section pasting the captured commands + output from Steps 1-2 and the A-1 resolution log. Also paste the Step 1-2 evidence into the "Milestone evidence" section of `local-pascal-gpu-testing.md`. (The doc also records the Phase-0 GC diagnostic that established the GC abandonment — see the amendment note; there is no Phase-3 GC calibration, that is #137's.)

- [ ] **Step 4 (DECISION GATE — branch):**

- **If both proofs pass** → proceed to Workstream B. Commit:

  ```bash
  cd "$WT" && uv run ruff format --check docs/ 2>/dev/null; git add docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md docs/testing/local-pascal-gpu-testing.md
  git commit -m "docs(testing): record §4.3 GTX 1080 sm_61 + bnb Linear4bit milestone (PASS)"
  ```

> **Historical outcome:** both proofs **PASSED** on the real 1080 (evidence recorded in the manual-pass doc); the §4.3 fallback below was not invoked.

- **If EITHER proof fails (§4.3 FALLBACK):** STOP the Pascal track. Then:
  1. Record the **negative** result (the exact error) in the manual-pass record and in `gpu-test-policy.md`.
  2. Ship the **`gpu_local` tier empty** (defined + wired in B-4, but with **no member tests** — every GPU test classifies `gpu_t4`/`gpu_xl`).
  3. Workstreams B (markers/dtype/preset, all CPU-acceptance-testable), D, and E **still complete** on the tiers that have hardware.
  4. **File the Pascal-blocked follow-ups** (the close-out's follow-up task lists these). The PR remains shippable.

**Completion criteria:** either both milestone proofs PASS (evidence recorded; proceed to B) **or** the §4.3 fallback is invoked, documented, and the follow-ups are queued.

---

# Workstream B — Code enablement (CC 6.0 floor, dtype coercion, preset fidelity, tier markers)

> Spec §5. Depends on A-2 for the on-1080 acceptance checks; the *logic* is CPU-TDD-able (capability passed as a value).

## Task B-1: Lower the GPU-compatibility floor to CC 6.0 and rewrite skip reasons

**Workstream B. CPU-testable (logic) + 1080 acceptance.** Spec §5.1. Closes the #79 follow-up "Tighten CC-7.5 skip message in `tests/conftest.py`".

**Difficulty:** Easy. **Subagent:** implementer (sonnet/high).

**Files:**
- Modify: `tests/conftest.py` (`_has_compatible_gpu` line 39-46; skip reasons lines 24-27, 51-52)

**Context:** `_has_compatible_gpu` currently returns `(major, minor) >= (7, 5)` (line 46). The `requires_compatible_gpu` marker reason (lines 24-27) and the `skip_no_gpu` reason (line 52) both say `CC >= 7.5`.

- [ ] **Step 1: Write the failing test.**

The autoskip logic is CPU-testable by injecting capability. Add to a new `tests/unit/test_marker_autoskip.py` (B-4 extends this file; create it here for the CC-floor case). Refactor `_has_compatible_gpu` to accept an optional capability so it is testable without a GPU — OR test via monkeypatching `torch.cuda.get_device_capability`. Use the monkeypatch form (no production signature change):

```python
"""CPU tests for conftest GPU-compat floor and tier autoskip logic."""

from __future__ import annotations

import importlib

import pytest


def _conftest():
    return importlib.import_module("tests.conftest")


@pytest.mark.parametrize(
    ("cap", "expected"),
    [((6, 0), True), ((6, 1), True), ((7, 5), True), ((8, 0), True), ((5, 0), False)],
)
def test_has_compatible_gpu_floor_is_cc60(monkeypatch, cap, expected) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: cap)
    assert _conftest()._has_compatible_gpu() is expected
```

- [ ] **Step 2: Run it; verify the `(6, 0)`/`(6, 1)` cases FAIL** (current floor is `(7, 5)`):

```bash
cd "$WT" && uv run pytest tests/unit/test_marker_autoskip.py -v --no-cov
```

Expected: `(6, 0)` and `(6, 1)` cases FAIL (return `False` under the old floor).

- [ ] **Step 3: Lower the floor and rewrite skip reasons.**

In `tests/conftest.py`, change line 46:

```python
    return (major, minor) >= (6, 0)
```

Rewrite the `requires_compatible_gpu` marker reason (lines 24-27):

```python
    config.addinivalue_line(
        "markers",
        "requires_compatible_gpu: skip unless a CUDA device with compute "
        "capability >= 6.0 is available (NF4 QLoRA + LoRA work from CC 6.0 / "
        "Pascal; only LLM.int8() needs CC 7.5 and is unused here)",
    )
```

Rewrite the `skip_no_gpu` reason (line 52):

```python
    skip_no_gpu = pytest.mark.skip(
        reason="requires a CUDA GPU with CC >= 6.0 (NF4 QLoRA + LoRA; "
        "LLM.int8() would need CC 7.5 but is unused here)"
    )
```

- [ ] **Step 4: Run the test; verify all cases pass.**

```bash
cd "$WT" && uv run pytest tests/unit/test_marker_autoskip.py -v --no-cov
```

Expected: all parametrized cases pass.

- [ ] **Step 5 (1080 acceptance, after A-2): confirm the floor on real hardware.**

```bash
cd "$WT" && uv run python -c "from tests.conftest import _has_compatible_gpu; print('compatible', _has_compatible_gpu())"
```

Expected: `compatible True` (the 1080 is CC 6.1). On a CPU-only shell it returns `False`.

- [ ] **Step 6: Lint gate + commit.**

```bash
cd "$WT" && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
git add tests/conftest.py tests/unit/test_marker_autoskip.py
git commit -m "feat(tests): lower GPU-compat floor to CC 6.0; clarify NF4-vs-LLM.int8 skip reasons (#79)"
```

**Completion criteria:** floor is `(6,0)`; skip reasons distinguish NF4/CC-6.0 from unused-LLM.int8; on the 1080 `_has_compatible_gpu()` is `True`; lint gate green.

---

## Task B-2: Device-aware dtype-coercion helper (bf16→fp16 on CC<8.0), wired at both seams

**Workstream B. CPU-testable (helper logic) + 1080 acceptance.** Spec §5.2. The helper coerces `bfloat16 → float16` when the target device has CC < 8.0, with **at most one warning per process**.

**Difficulty:** Medium. **Subagent:** implementer (sonnet/high).

**Files:**
- Modify: `src/custom_sam_peft/runtime/_runtime.py` (add `coerce_dtype_for_capability`)
- Modify: `src/custom_sam_peft/train/loop.py` (`_autocast_ctx`, line ~191)
- Modify: `src/custom_sam_peft/peft_adapters/qlora.py` (`_replace_with_bnb_linear4bit`, line ~142)
- Create: `tests/unit/test_dtype_coercion.py`

**Context:** `runtime/_runtime.py` already owns device+dtype truth (the `Runtime` value object + `_DTYPE_MAP`), so the helper's home is there (spec §5.2 leaves placement to the planner). `_autocast_ctx` (loop.py:186-192) maps `cfg.model.dtype == "bfloat16"` → `torch.bfloat16` else `torch.float16` at line 191; it returns `nullcontext()` for QLoRA (line 187-188), so the autocast seam is only hit on the LoRA path. `_replace_with_bnb_linear4bit` (qlora.py:139-155) builds `compute_dtype = _torch_dtype(qcfg.compute_dtype)` at line 142 and passes it to `bnb.nn.Linear4bit(... compute_dtype=compute_dtype)` at line 146-151.

- [ ] **Step 1: Write the failing CPU test.**

Create `tests/unit/test_dtype_coercion.py`:

```python
"""CPU tests for device-aware bf16->fp16 coercion on CC<8.0 hardware."""

from __future__ import annotations

import logging

import torch

from custom_sam_peft.runtime._runtime import coerce_dtype_for_capability


def test_bf16_coerced_to_fp16_below_cc80() -> None:
    assert coerce_dtype_for_capability(torch.bfloat16, capability=(6, 1)) is torch.float16
    assert coerce_dtype_for_capability(torch.bfloat16, capability=(7, 5)) is torch.float16


def test_bf16_preserved_at_cc80_and_above() -> None:
    assert coerce_dtype_for_capability(torch.bfloat16, capability=(8, 0)) is torch.bfloat16
    assert coerce_dtype_for_capability(torch.bfloat16, capability=(9, 0)) is torch.bfloat16


def test_non_bf16_never_coerced() -> None:
    assert coerce_dtype_for_capability(torch.float16, capability=(6, 1)) is torch.float16
    assert coerce_dtype_for_capability(torch.float32, capability=(6, 1)) is torch.float32


def test_warns_at_most_once_per_process(caplog) -> None:
    # Reset the module's one-shot flag so the test is order-independent.
    import custom_sam_peft.runtime._runtime as rt

    rt._dtype_coercion_warned = False
    with caplog.at_level(logging.WARNING, logger="custom_sam_peft.runtime._runtime"):
        coerce_dtype_for_capability(torch.bfloat16, capability=(6, 1))
        coerce_dtype_for_capability(torch.bfloat16, capability=(6, 1))
    warnings = [r for r in caplog.records if "bfloat16" in r.message.lower()]
    assert len(warnings) == 1, [r.message for r in caplog.records]
```

- [ ] **Step 2: Run it; verify it fails** (`ImportError: cannot import name 'coerce_dtype_for_capability'`):

```bash
cd "$WT" && uv run pytest tests/unit/test_dtype_coercion.py -v --no-cov
```

- [ ] **Step 3: Implement the helper.**

In `src/custom_sam_peft/runtime/_runtime.py`, add a module-level flag and the helper:

```python
import logging

logger = logging.getLogger(__name__)

_dtype_coercion_warned = False


def coerce_dtype_for_capability(
    dtype: torch.dtype,
    *,
    capability: tuple[int, int] | None = None,
    device: torch.device | None = None,
) -> torch.dtype:
    """Coerce bfloat16 -> float16 on hardware below compute capability 8.0.

    bf16 is emulated below CC 8.0 (Pascal/Turing/Volta), so we run those cards
    in float16. Only bfloat16 is touched; float16/float32 pass through. Emits a
    one-time warning per process when a coercion happens.

    Pass ``capability`` directly (CPU-testable) or a ``device`` to read it from.
    """
    global _dtype_coercion_warned
    if dtype is not torch.bfloat16:
        return dtype
    if capability is None:
        if device is not None and device.type == "cuda":
            capability = torch.cuda.get_device_capability(device)
        else:
            return dtype  # CPU / unknown: leave bf16 alone (autocast path handles CPU)
    if capability >= (8, 0):
        return dtype
    if not _dtype_coercion_warned:
        logger.warning(
            "Requested bfloat16 on a device with compute capability %s (< 8.0, "
            "where bf16 is emulated); coercing to float16. This is expected on "
            "Pascal (GTX 1080).",
            capability,
        )
        _dtype_coercion_warned = True
    return torch.float16
```

- [ ] **Step 4: Run the test; verify it passes.**

```bash
cd "$WT" && uv run pytest tests/unit/test_dtype_coercion.py -v --no-cov
```

Expected: 4 passed.

- [ ] **Step 5: Wire the autocast seam (LoRA path).**

In `src/custom_sam_peft/train/loop.py`, replace the dtype selection at line 191:

```python
    requested = torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16
    dtype = coerce_dtype_for_capability(
        requested, device=torch.device("cuda", torch.cuda.current_device())
    )
    return torch.autocast(device_type="cuda", dtype=dtype)
```

Add the import at the top of `loop.py` (with the other `custom_sam_peft.runtime` imports):

```python
from custom_sam_peft.runtime._runtime import coerce_dtype_for_capability
```

> The `nullcontext()` early-returns at lines 187-190 stay above this — they are reached first for QLoRA and CPU, so the coercion only runs on the LoRA+CUDA autocast path. (Spec §5.4 note.)

- [ ] **Step 6: Wire the QLoRA compute-dtype seam.**

In `src/custom_sam_peft/peft_adapters/qlora.py`, change `_replace_with_bnb_linear4bit` (line 142) so the compute dtype is coerced against the **target device** before constructing `Linear4bit`. The device is `old.weight.device` (the parameter is moved to cuda at line 154); read it before the loop body:

```python
    compute_dtype = _torch_dtype(qcfg.compute_dtype)
    for name in names:
        parent, attr = _resolve_parent(base, name)
        old = cast(nn.Linear, getattr(parent, attr))
        block_dtype = coerce_dtype_for_capability(compute_dtype, device=old.weight.device)
        new = bnb.nn.Linear4bit(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            quant_type=qcfg.quant_type,
            compute_dtype=block_dtype,
            ...
```

Add the import at the top of `qlora.py`:

```python
from custom_sam_peft.runtime._runtime import coerce_dtype_for_capability
```

> If `old.weight.device` is CPU at construction time (model not yet moved to cuda), the helper returns bf16 unchanged — but the QLoRA recipe loads the base on cuda first. **Implementer: confirm the device at the `_replace_with_bnb_linear4bit` call site is already cuda on the 1080 path** (it is, since quantization fires on `.to(cuda)` at line 154 and the base is constructed on-device). If it is CPU, pass the model's target device explicitly from `apply_qlora` instead.

- [ ] **Step 7: CPU regression — full unit suite (the coercion must not change CPU behavior).**

```bash
cd "$WT" && uv run pytest tests/unit -q --no-cov 2>&1 | tail -5
```

Expected: all pass (CPU autocast path is unaffected because `_autocast_ctx` returns `nullcontext()` when `not torch.cuda.is_available()`).

- [ ] **Step 8 (1080 acceptance, after A-2): one coercion warning per process on a bf16 QLoRA construct.**

```bash
cd "$WT" && uv run python - <<'PY'
import logging, torch
logging.basicConfig(level=logging.WARNING)
from custom_sam_peft.runtime._runtime import coerce_dtype_for_capability
d = torch.device("cuda", 0)
print("coerced", coerce_dtype_for_capability(torch.bfloat16, device=d))  # -> float16 on 1080
print("again  ", coerce_dtype_for_capability(torch.bfloat16, device=d))  # no 2nd warning
PY
```

Expected: prints `float16` twice; the bf16-coercion WARNING appears exactly once.

- [ ] **Step 9: Lint gate + commit.**

```bash
cd "$WT" && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
git add src/custom_sam_peft/runtime/_runtime.py src/custom_sam_peft/train/loop.py src/custom_sam_peft/peft_adapters/qlora.py tests/unit/test_dtype_coercion.py
git commit -m "feat(runtime): device-aware bf16->fp16 coercion on CC<8.0 at autocast + QLoRA seams"
```

**Completion criteria:** helper coerces bf16→fp16 below CC 8.0, warns once; both seams route through it; CPU suite unaffected; on the 1080 a bf16 request yields float16 with one warning; lint gate green.

---

## Task B-3: Preset decision dtype fidelity (float16 on CC<8.0)

**Workstream B. CPU-testable + 1080 acceptance.** Spec §5.3.

**Difficulty:** Medium. **Subagent:** implementer (sonnet/high).

**Files:**
- Modify: `src/custom_sam_peft/presets.py` (Literal line 83; `decide_preset` lines 295/335; `label()` line 120; comment line 142)
- Modify/extend: `tests/unit/test_presets.py`

**Context:** `PresetDecision.dtype` is `Literal["bfloat16"]` (line 83), constructed with `dtype="bfloat16"` (line 335). `decide_preset` reads `torch.cuda.get_device_properties(0)` for total memory at line 295 (`props`). `label()` hardcodes `bf16` in the returned string at line 120. The byte-count `_bytes_per_param_for_method` comment says `# bf16 vs NF4` (line 142). The schema's `Dtype` is already `Literal["bfloat16", "float16"]`. The existing round-trip test (`test_preset_decision_to_json_round_trip`) and label tests do NOT assert the `bf16` token (they check `"LoRA r=32"`, `"calibrated"`), so widening is safe.

- [ ] **Step 1: Write the failing CPU tests.**

Add to `tests/unit/test_presets.py`:

```python
def test_preset_decision_label_renders_fp16_token() -> None:
    d = _make_decision()
    object.__setattr__(d, "dtype", "float16")  # PresetDecision is a frozen dataclass
    assert "fp16" in d.label()
    assert "bf16" not in d.label()


def test_preset_decision_label_renders_bf16_token() -> None:
    d = _make_decision()  # default dtype="bfloat16"
    assert "bf16" in d.label()


def test_preset_decision_float16_round_trips() -> None:
    d = _make_decision()
    object.__setattr__(d, "dtype", "float16")
    d2 = PresetDecision.from_json(d.to_json())
    assert d2.dtype == "float16"
    assert d == d2
```

> If `PresetDecision` is not frozen, drop the `object.__setattr__` and pass `dtype` into `_make_decision` (add a `dtype` kwarg to the helper). Implementer adapts to the actual dataclass mutability.

- [ ] **Step 2: Run; verify failures** (label still emits `bf16` for a float16 decision; `Literal["bfloat16"]` rejects `"float16"` on from_json validation if validated):

```bash
cd "$WT" && uv run pytest tests/unit/test_presets.py -k "fp16 or bf16 or float16_round" -v --no-cov
```

- [ ] **Step 3: Widen the Literal + render the real token + select CC-aware dtype.**

In `src/custom_sam_peft/presets.py`:

1. Line 83 — widen the annotation:

```python
    dtype: Literal["bfloat16", "float16"]
```

2. Line 120 (inside `label()`) — render the real token. Replace the hardcoded `bf16` in the f-string:

```python
        dtype_token = "fp16" if self.dtype == "float16" else "bf16"
        return (
            f"auto: {method} r={self.r} batch={self.batch_size} "
            f"grad_accum={self.grad_accum_steps} ckpt={ckpt} {dtype_token} — "
            f"fits in {used_gib:.1f}/{total_gib:.1f} GiB on {self.gpu_name} {suffix}"
        )
```

3. `decide_preset` (lines 295, 335) — read capability and choose the dtype:

```python
    props = torch.cuda.get_device_properties(0)   # line 295 (unchanged)
    total = int(props.total_memory)
    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    decided_dtype = "float16" if cc < (8, 0) else "bfloat16"
```

Then at line 335 use the variable:

```python
        dtype=decided_dtype,
```

4. Line 142 — update the comment to acknowledge float16 (the 2.0 B/param value is unchanged):

```python
    return 2.0 if method == "lora" else 0.5  # bf16/fp16 (2.0) vs NF4 (0.5)
```

> `config_patch` already emits `self.dtype` into `model.dtype` (spec §5.3), so a float16 decision flows into the generated config automatically.

- [ ] **Step 4: Run the new + existing preset tests.**

```bash
cd "$WT" && uv run pytest tests/unit/test_presets.py -v --no-cov 2>&1 | tail -10
```

Expected: all pass (new fp16/bf16/round-trip cases + the pre-existing label/round-trip cases).

- [ ] **Step 5 (1080 acceptance, after A-2): `decide_preset` returns float16 on the 1080.**

```bash
cd "$WT" && uv run python - <<'PY'
from custom_sam_peft.presets import decide_preset
d = decide_preset(image_size=1008)
print("dtype", d.dtype)            # Expected: float16 (1080 is CC 6.1)
print("label", d.label())          # Expected: contains "fp16"
print("patch dtype", d.config_patch()["model"]["dtype"])  # Expected: float16
PY
```

> NOTE: on a tight ~7 GB card `decide_preset` may raise the "needs a larger GPU" RuntimeError (line 319) if no candidate fits — that is acceptable; what we verify is *when it does decide*, the dtype is `float16`. If it raises, confirm via a forced-capability unit assertion instead and note it in the manual-pass record.

- [ ] **Step 6: Lint gate + commit.**

```bash
cd "$WT" && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
git add src/custom_sam_peft/presets.py tests/unit/test_presets.py
git commit -m "feat(presets): float16 decision + fp16 label token on CC<8.0 hardware"
```

**Completion criteria:** `dtype` Literal widened; `decide_preset` picks float16 when CC<8.0; `label()` renders `fp16`/`bf16`; round-trip preserves float16; all preset CPU tests pass; lint gate green.

---

## Task B-4: Tier-marker wiring — register markers, autoskip, runner script

**Workstream B. CPU-testable (registration/collection/parsing) + 1080 acceptance.** Spec §3.1, §5.4. **Serialize: touches `pyproject.toml`, `tests/conftest.py`, `scripts/run_gpu_tests.sh`.**

**Difficulty:** Medium. **Subagent:** implementer (sonnet/high).

**Files:**
- Modify: `pyproject.toml` (`[tool.pytest.ini_options].markers`, lines 126-132)
- Modify: `tests/conftest.py` (`pytest_configure` add 3 markers; `pytest_collection_modifyitems` per-tier autoskip)
- Modify: `scripts/run_gpu_tests.sh`
- Create: `tests/unit/test_marker_autoskip.py` (extend the B-1 file), `tests/unit/test_run_gpu_tests_script.py`

**Context:** Today `pyproject.toml` markers (lines 126-132) register `integration`, `gpu`, `gpu_inspection`, `requires_checkpoint`, `requires_compatible_gpu`. `tests/conftest.py::pytest_configure` registers `gpu_inspection` (lines 32-36) but NOT `gpu`/`integration` (those live only in pyproject). `pytest_collection_modifyitems` (lines 49-58) skips on missing checkpoint / missing GPU. `scripts/run_gpu_tests.sh` accepts `inspection|release|all` mapping to `gpu_inspection`/`gpu` markers and `tests/integration/`/`tests/gpu/` paths — it **omits `tests/predict/`** and the header counts are stale ("12 tests").

> **Marker reclassification ordering:** this task REGISTERS the three new markers and the autoskip, and updates the runner, but the per-test marker SWAP (replacing `pytest.mark.gpu`/`gpu_inspection` on each test file) happens in **D-2** after the D-1 audit assigns tiers. Until D-2 lands, keep the legacy `gpu`/`gpu_inspection` markers registered ALONGSIDE the new ones so collection does not break. Drop the legacy registrations in D-2's final step.

- [ ] **Step 1: Write the failing CPU tests.**

Extend `tests/unit/test_marker_autoskip.py` (created in B-1) with the per-tier autoskip logic. The autoskip is testable by constructing fake pytest items with tier keywords and asserting the skip marker added. Add:

```python
class _FakeItem:
    def __init__(self, *keywords: str) -> None:
        self.keywords = set(keywords)
        self.markers: list[object] = []

    def add_marker(self, marker: object) -> None:
        self.markers.append(marker)


def _run_modify(items, *, have_local, have_t4, have_xl, ckpt_exists):
    import importlib

    conftest = importlib.reload(importlib.import_module("tests.conftest"))
    return conftest, items


def test_gpu_t4_skipped_on_1080(monkeypatch) -> None:
    """A gpu_t4 test is skipped when only the local (1080) tier is present."""
    import tests.conftest as conftest

    monkeypatch.setattr(conftest, "_has_compatible_gpu", lambda: True)
    monkeypatch.setattr(conftest, "_current_tier", lambda: "gpu_local")
    item = _FakeItem("gpu_t4", "requires_compatible_gpu")
    conftest.pytest_collection_modifyitems(config=None, items=[item])  # type: ignore[arg-type]
    assert item.markers, "gpu_t4 test was not skipped on the local tier"


def test_gpu_local_runs_on_1080(monkeypatch) -> None:
    import tests.conftest as conftest

    monkeypatch.setattr(conftest, "_has_compatible_gpu", lambda: True)
    monkeypatch.setattr(conftest, "_current_tier", lambda: "gpu_local")
    item = _FakeItem("gpu_local", "requires_compatible_gpu")
    conftest.pytest_collection_modifyitems(config=None, items=[item])  # type: ignore[arg-type]
    assert not item.markers, "gpu_local test should not be skipped on the 1080"


def test_gpu_xl_skip_reason_names_124(monkeypatch) -> None:
    import tests.conftest as conftest

    monkeypatch.setattr(conftest, "_has_compatible_gpu", lambda: True)
    monkeypatch.setattr(conftest, "_current_tier", lambda: "gpu_local")
    item = _FakeItem("gpu_xl", "requires_compatible_gpu")
    conftest.pytest_collection_modifyitems(config=None, items=[item])  # type: ignore[arg-type]
    assert item.markers, "gpu_xl test not skipped"
    reason = getattr(item.markers[0], "kwargs", {}).get("reason", "")
    assert "#124" in reason, reason
```

> Implementer: adapt the fake-item/skip-marker introspection to whatever `pytest.mark.skip(...)` returns (`MarkDecorator` with `.kwargs["reason"]`). The key behaviors to assert: (a) a higher-tier test is skipped on a lower-tier runner, (b) the matching-tier test is NOT skipped, (c) `gpu_xl` skip reason contains `#124`.

- [ ] **Step 2: Run; verify failures** (no `_current_tier`, no per-tier autoskip):

```bash
cd "$WT" && uv run pytest tests/unit/test_marker_autoskip.py -v --no-cov
```

- [ ] **Step 3: Register the three markers (both places) + implement the tier autoskip.**

In `pyproject.toml` `[tool.pytest.ini_options].markers` (lines 126-132), ADD (keep legacy until D-2):

```toml
  "gpu_local: GPU test that fits the GTX 1080 (<=~7 GB, CC 6.0+, NF4+float16); run via run_gpu_tests.sh local",
  "gpu_t4: GPU test needing >8 GB and <=16 GB, or bf16-representative numerics; Colab T4",
  "gpu_xl: GPU test beyond a T4 (>16 GB / larger arch); cloud auto-provision (needs #124)",
```

In `tests/conftest.py::pytest_configure`, add the same three `config.addinivalue_line("markers", ...)` entries (mirror the `gpu_inspection` block at lines 32-36).

Add a tier-detection helper + the per-tier autoskip in `tests/conftest.py`:

```python
_TIER_ORDER = {"gpu_local": 0, "gpu_t4": 1, "gpu_xl": 2}


def _current_tier() -> str | None:
    """The highest tier the current runner's live hardware can satisfy.

    The 1080 (and any CC>=6.0 card <=~7 GB) is the local tier. We cannot
    auto-detect >8 GB / bf16-faithful capability reliably here, so the runner
    that targets t4/xl asserts its own tier; on the dev box this returns
    'gpu_local'. Returns None on CPU-only (no compatible GPU)."""
    if not _has_compatible_gpu():
        return None
    return "gpu_local"
```

Extend `pytest_collection_modifyitems` (after the existing GPU/ckpt loop, lines 54-58):

```python
    runner_tier = _current_tier()
    for item in items:
        item_tier = next((t for t in _TIER_ORDER if t in item.keywords), None)
        if item_tier is None:
            continue
        if runner_tier is None:
            # CPU-only CI: already skipped via requires_compatible_gpu above.
            continue
        if _TIER_ORDER[item_tier] > _TIER_ORDER[runner_tier]:
            if item_tier == "gpu_xl":
                reason = (
                    "gpu_xl tier needs a >16 GB / larger-arch runner via cloud "
                    "auto-provision (#124); not available on this runner"
                )
            else:
                reason = f"{item_tier} tier needs hardware beyond this runner ({runner_tier})"
            item.add_marker(pytest.mark.skip(reason=reason))
```

- [ ] **Step 4: Run the autoskip tests; verify pass.**

```bash
cd "$WT" && uv run pytest tests/unit/test_marker_autoskip.py -v --no-cov
```

Expected: all pass.

- [ ] **Step 5: Write the failing runner-script parsing test.**

Create `tests/unit/test_run_gpu_tests_script.py` — verify the script maps each tier to its marker + path set, accepts only `{local,t4,xl}`, and collects `tests/predict/`. Drive the script in `--collect-only`/dry mode via env, or grep its case arms:

```python
"""CPU test for run_gpu_tests.sh tier parsing (no GPU needed)."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_gpu_tests.sh"


def _arms() -> str:
    return SCRIPT.read_text()


def test_accepts_three_tiers_and_rejects_legacy() -> None:
    src = _arms()
    assert "local)" in src and "t4)" in src and "xl)" in src
    assert "inspection)" not in src and "release)" not in src


def test_local_maps_to_gpu_local_marker() -> None:
    src = _arms()
    assert "gpu_local" in src


def test_collects_predict_path() -> None:
    src = _arms()
    assert "tests/predict/" in src


def test_rejects_unknown_tier() -> None:
    res = subprocess.run(  # noqa: S603
        ["bash", str(SCRIPT), "bogus"],
        capture_output=True,
        text=True,
        env={"PYTHON": "true", "PATH": "/usr/bin:/bin"},
    )
    assert res.returncode != 0
    assert "usage" in (res.stderr + res.stdout).lower()
```

- [ ] **Step 6: Run; verify failures** (script still uses `inspection|release|all`):

```bash
cd "$WT" && uv run pytest tests/unit/test_run_gpu_tests_script.py -v --no-cov
```

- [ ] **Step 7: Rewrite the runner script.**

Replace `scripts/run_gpu_tests.sh` lines 6-43 so the usage, tiers, and case arms select by `{local,t4,xl}`, add `tests/predict/`, and fix the header. Preserve the `--deselect` convention + `gpu-deselect-check` guard (lines 14-25 commentary, unchanged):

```bash
# Usage:
#   scripts/run_gpu_tests.sh [local|t4|xl]
#
# Hardware tiers (see docs/testing/gpu-test-policy.md):
#   local — fits the GTX 1080 (<=~7 GB, CC 6.0+, NF4 + float16). Dev box via
#           `uv sync --extra gpu-pascal`. Marker: gpu_local.
#   t4    — needs >8 GB and <=16 GB, or bf16-representative numerics. Colab T4.
#           Marker: gpu_t4.
#   xl    — beyond a T4 (>16 GB / larger arch). Cloud auto-provision (#124).
#           Marker: gpu_xl. Likely near-empty initially.
#
# (Test counts per tier are documented in gpu-test-policy.md, not hardcoded here.)
...
set -euo pipefail
TIER="${1:-local}"

case "$TIER" in
  local) MARKER_EXPR="gpu_local" ;;
  t4)    MARKER_EXPR="gpu_t4" ;;
  xl)    MARKER_EXPR="gpu_xl" ;;
  *) echo "usage: $0 [local|t4|xl]" >&2; exit 2 ;;
esac

PATHS="tests/gpu/ tests/integration/ tests/predict/"
# shellcheck disable=SC2086
"${PYTHON:-python}" -m pytest -v --tb=short \
  -m "$MARKER_EXPR" --no-cov $PATHS
```

> Each tier now collects across all three path roots and filters by marker (a `gpu_local` test in `tests/predict/` is collected; a `gpu_t4` test in `tests/gpu/` is filtered out for the `local` tier). This is simpler and correct because the marker is the single selection axis (spec §3 canonical partition).

- [ ] **Step 8: Run the script test + confirm the CI guard still greps clean.**

```bash
cd "$WT" && uv run pytest tests/unit/test_run_gpu_tests_script.py -v --no-cov
grep -nE -- '(^|[[:space:]])--deselect([[:space:]]|=)' scripts/run_gpu_tests.sh | grep -v '^[0-9]*:#' || echo "no stray --deselect"
```

Expected: script tests pass; `no stray --deselect`.

- [ ] **Step 9 (1080 acceptance, after A-2): the local tier collects on the 1080.**

```bash
cd "$WT" && uv run pytest -m gpu_local tests/gpu/ tests/integration/ tests/predict/ --collect-only -q 2>&1 | tail -15
```

Expected: collection succeeds (it lists `gpu_local` tests once D-2 has assigned them; before D-2 it lists none, which is fine — the autoskip/collection mechanism is what we verify here). On a CPU-only shell, `gpu_local` tests collect-and-skip.

- [ ] **Step 10: Lint gate + commit.**

```bash
cd "$WT" && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
git add pyproject.toml tests/conftest.py scripts/run_gpu_tests.sh tests/unit/test_marker_autoskip.py tests/unit/test_run_gpu_tests_script.py
git commit -m "feat(tests): register gpu_local/t4/xl tier markers + per-tier autoskip + tier-aware runner"
```

**Completion criteria:** three tier markers registered (pyproject + conftest); per-tier hardware autoskip works (CPU TDD green); runner accepts `{local,t4,xl}`, collects `tests/predict/`, rejects legacy names; CI deselect-guard clean; lint gate green.

---

# Workstream C — `use_double_quant` (GC abandoned; 8 GB training → #137)

> Spec §6. Depends on A-2 (milestone) and B (CC floor, dtype coercion). **Reduced scope:** only **C-1** (`use_double_quant`) is in scope and it has landed. **C-2** (Phase-0 diagnostic + Phase-1 GC fix) is **abandoned** — the Phase-0 diagnostic on the 1080 proved the GC `CheckpointError` unfixable without editing forbidden sam3 source (see the amendment note + `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md`). **C-3** (8 GB recipe config + training smoke) and **C-4** (Phase-3 verification + 8 GB calibration) are **deferred to #137**.

## Task C-1: Add `use_double_quant` to `QLoRAConfig` + CPU wiring tests (TDD)

**Workstream C. CPU-testable.** Spec §6.2. Adds the schema field and honors it at the `Linear4bit` constructor; persists in metadata if the round-trip needs it.

**Difficulty:** Medium. **Subagent:** implementer (sonnet/high).

**Files:**
- Modify: `src/custom_sam_peft/config/schema.py` (`QLoRAConfig`, line 486-488)
- Modify: `src/custom_sam_peft/peft_adapters/qlora.py` (`_replace_with_bnb_linear4bit` ctor line ~146; metadata write/read lines 354-359 / 380-383; version line 39 + docstring line 16)
- Modify/Create: `tests/unit/test_qlora.py` (or the existing QLoRA CPU test file — implementer locates it)

**Context:** `QLoRAConfig` (schema.py:486-488) has only `quant_type` and `compute_dtype`. The `Linear4bit` is built directly in `_replace_with_bnb_linear4bit` (qlora.py:139-155) — NO `BitsAndBytesConfig`. Double-quant is `bnb.nn.Linear4bit(..., compress_statistics=True)` (the bnb double-quant flag). The QLoRA metadata file `custom_sam_peft_qlora.json` (v1, `_QLORA_META_VERSION = 1` at line 39) carries `format_version/quant_type/compute_dtype` (lines 354-359); `load_qlora` rejects a version mismatch (line 375-379).

- [ ] **Step 1: Write the failing CPU tests.**

Add to the QLoRA CPU test file (find via `grep -rln "QLoRAConfig" tests/unit`):

```python
def test_qlora_config_double_quant_defaults_false() -> None:
    from custom_sam_peft.config.schema import QLoRAConfig

    assert QLoRAConfig().use_double_quant is False  # preserve current behavior


def test_qlora_config_double_quant_roundtrips() -> None:
    from custom_sam_peft.config.schema import QLoRAConfig

    cfg = QLoRAConfig(use_double_quant=True)
    assert cfg.model_dump()["use_double_quant"] is True
```

If the metadata shape is extended (Step 3 decision), also add a metadata round-trip test asserting `use_double_quant` survives `save_qlora`/`load_qlora` and the `format_version` bumps to 2.

- [ ] **Step 2: Run; verify failure** (`use_double_quant` is not a field — pydantic `_Strict` rejects it):

```bash
cd "$WT" && uv run pytest tests/unit -k double_quant -v --no-cov
```

- [ ] **Step 3: Add the field + honor it at the constructor.**

In `src/custom_sam_peft/config/schema.py` (line 488), extend `QLoRAConfig`:

```python
class QLoRAConfig(_Strict):
    quant_type: QuantType = "nf4"
    compute_dtype: Dtype = "bfloat16"
    use_double_quant: bool = False  # bnb nested quantization of the quant constants
```

In `src/custom_sam_peft/peft_adapters/qlora.py` `_replace_with_bnb_linear4bit`, pass the flag to the constructor (line ~146):

```python
        new = bnb.nn.Linear4bit(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            quant_type=qcfg.quant_type,
            compute_dtype=block_dtype,            # from B-2
            compress_statistics=qcfg.use_double_quant,
        )
```

**Metadata decision (spec §6.2):** persist `use_double_quant` so a reload reconstructs an identical quantization. Bump the version: in `qlora.py` set `_QLORA_META_VERSION = 2` (line 39), update the docstring example (line 16) to `format_version: 2` with the new field, add `"use_double_quant": <inferred>` to the `meta` dict in `save_qlora` (line 354-358), and read it in `load_qlora` into the reconstructed `QLoRAConfig` (line 380-383). Add an `_infer_double_quant_from_wrapper` helper alongside the existing `_infer_*_from_wrapper` functions (read `module.weight.quant_state` / `compress_statistics` from the first `Linear4bit`).

> If inferring from the live module proves unreliable on CPU (no real bnb module), persist the value from the config at save time instead of inferring — the planner's allowance in §6.2. Either way bump the version.

- [ ] **Step 4: Run the CPU tests; verify pass.**

```bash
cd "$WT" && uv run pytest tests/unit -k "double_quant or qlora" -v --no-cov 2>&1 | tail -10
```

Expected: new + existing QLoRA CPU tests pass.

- [ ] **Step 5: Lint gate + commit.**

```bash
cd "$WT" && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
git add src/custom_sam_peft/config/schema.py src/custom_sam_peft/peft_adapters/qlora.py tests/unit/
git commit -m "feat(qlora): add use_double_quant config field wired to Linear4bit (metadata v2)"
```

**Completion criteria:** `use_double_quant` exists, defaults `False` (preserves existing configs), honored at the `Linear4bit` ctor, metadata version bumped + round-trips; CPU tests pass; lint gate green.

---

## Task C-2: GC fix — ABANDONED (not implemented)

**Workstream C. ABANDONED — no work product.** The original C-2 ran the Phase-0 diagnostic on the real 1080 and was to implement the justified Phase-1 GC fix in `vit_act_checkpoint.py`.

**Outcome:** the Phase-0 diagnostic (recorded in `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md`, "Phase-0 trace + fix classification") proved the `CheckpointError` is a **structural save-count divergence INSIDE sam3's `multi_head_attention_forward` recompute** — a CPU int64 scalar materializing at a non-deterministic autograd-save position. Fix A (deterministic-autocast wrap), Fix A+ (SDPA-MATH pinning), and Fix B (owning the checkpoint with a pinned `context_fn`) were all attempted on real hardware and none resolve it; Fix C (`determinism_check="none"`) yields a corrupted backward (proving the divergence is not benign). The only resolution would edit `sam3/model/model_misc.py`, which is **FORBIDDEN** (sam3 is external; we monkeypatch only via `_patches/`).

**No fix ships.** PR #127 closed unmerged; #89/#60 closed not-planned; `origin/main` already ships GC OFF (config default `false` + a no-op guard, templates `false`) — nothing to revert. The GC-free rebuild of this branch **removes** `vit_act_checkpoint.py` and `tests/gpu/test_grad_checkpointing.py`. The independent fp16 cross-attention `attn_mask` MHA-hook fix (Blocker 1 in the evidence doc) is **not** GC work and **landed** regardless.

There is **no implementer task here** — the diagnostic already ran (its evidence is the load-bearing justification for the abandonment) and the fix was found infeasible.

---

## Task C-3: 8 GB QLoRA recipe + training smoke — DEFERRED to #137

**Workstream C. DEFERRED — no work product on this branch.** The original C-3 created a calibrated ~7 GB QLoRA *training* recipe (`configs/examples/gpu_smoke_qlora_8gb.yaml`) and a `gpu_local`-tagged training smoke (`tests/gpu/test_real_train_qlora_8gb.py`).

**Why deferred:** the recipe was **contingent on gradient checkpointing being the dominant activation-memory lever** (spec §6.x original). With GC abandoned (C-2), fitting a 1008² QLoRA *training* step (a full backward graph over the ViT-Det activations) within ~7 GB is an **open question**. `data.image_size` is **fixed at 1008** by SAM 3.1 (`sam3.py:749,753`; `schema.py:390`) and is not a lever, so the easy escape hatch is unavailable.

**Owner: issue #137** ("Investigate non-checkpointing approaches to fit QLoRA training in an 8 GB VRAM budget", OPEN). #137 owns the config, the training smoke, the calibration, and the GC-independent starting levers (NF4 + `use_double_quant`, `adamw8bit`, low LoRA rank/scope, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `batch_size: 1` + grad-accum, the `paged_adamw8bit` escape hatch). **Neither `gpu_smoke_qlora_8gb.yaml` nor `test_real_train_qlora_8gb.py` is created on this branch.**

> The GC-independent QLoRA improvement (`use_double_quant`) is **C-1**, which landed — it is not part of this deferral.

---

## Task C-4: Phase-3 verification + 8 GB calibration — DEFERRED to #137

**Workstream C. DEFERRED — no work product on this branch.** The original C-4 ran the 8 GB smoke on the 1080, calibrated the VRAM ceiling, and (per the former spec §6.8) decided a graceful-degradation branch. Both the GC verification it depended on (C-2) and the recipe it calibrated (C-3) are out of scope — GC abandoned, training recipe deferred. **#137 owns the on-hardware calibration** (on a 1080 or, if the step exceeds ~7 GB, a `gpu_t4` runner). The `gpu_local` **training** tier is **PROVISIONAL pending #137**; until then `gpu_local` holds forward-only / inspection tests only (classified in D-1).

---

# Workstream D — #117 full (CPU/GPU split audit)

> Spec §7. Depends on B-4 (tier markers); the `gpu_local` calibration depends on A-2. **Serialize the marker SWAP (D-2) — it shares files with B-4 and the test files. The CPU-move refactors of disjoint test files within D can parallelize once D-1 fixes the decisions.**

## Task D-1: Inventory + classify every GPU test into a tier (calibrated on the 1080)

**Workstream D. Audit doc (CPU-authored); the `gpu_local` classification requires the 1080.** Spec §7.1, §7.2.

**Difficulty:** Medium (analysis + calibration). **Subagent:** implementer (sonnet/high).

**Files:**
- Create: `docs/testing/gpu-audit-2026-05-24.md` (or a dedicated section of `gpu-test-policy.md` — D-1 decides placement per §7.2)

**Context — the 12 GC-free GPU-tagged test files (verified against the tree).** `test_grad_checkpointing.py` is **removed** by the GC-free rebuild and `test_real_train_qlora_8gb.py` was **never created** (deferred to #137), so neither is in the audit:
- `tests/gpu/` (8): `test_calibrate_real.py`, `test_channel_adapter_gpu.py`, `test_multiplex_vram.py`, `test_predict_nchannel_gpu.py`, `test_real_train_overfits.py`, `test_real_train_qlora.py`, `test_real_train_qlora_resume.py`, `test_run_end_to_end_gpu.py`.
- `tests/integration/` (3): `test_load_sam31_real.py`, `test_peft_lora_real.py`, `test_peft_qlora_real.py`.
- `tests/predict/` (1): `test_gpu_predict.py`.

> Verify the count before classifying: `grep -rl "mark\\.gpu" tests/ --include="*.py" | grep -v grad_checkpointing` → expect **12** files (the rebuild must already have dropped `test_grad_checkpointing.py`; if it is still present, the rebuild is incomplete — stop and resolve).

Current markers (verified): `tests/integration/test_{load_sam31,peft_lora,peft_qlora}_real.py` carry `gpu_inspection`; everything in `tests/gpu/` and `tests/predict/test_gpu_predict.py` carry `gpu`.

- [ ] **Step 1: Build the audit table.** For EVERY GPU-gated test (file::test granularity), record: assigned **tier** (`gpu_local`/`gpu_t4`/`gpu_xl`), a **keep-GPU / move-to-CPU / delete** decision + rationale, and (for move-to-CPU) the replacement mechanism (`TinySam3Stub` via the `stub_model` fixture — `tests/fixtures/tiny_sam3_stub.py`; synthetic tensors; or mocks).

  Tier-classification rules (spec §3, §7.2): a test is `gpu_local` **only if it actually runs within ~7 GB on the real 1080** (calibrate, don't estimate). Calibration runs **one test-file per process on the 1080** — the ~7 GB card OOMs when test files batch together. Forward-only / structural-inspection tests (the `tests/integration/*_real.py` inspection trio, channel-adapter/multiplex/predict introspection) are strong `gpu_local` candidates (no backward graph). Heavy training smokes (`test_real_train_overfits` LoRA, `test_real_train_qlora` 50-step) go to `gpu_t4` if they exceed ~7 GB. **The `gpu_local` training tier is PROVISIONAL pending #137** — with GC abandoned, there is no QLoRA training smoke in `gpu_local`; its baseline members are forward-only / inspection tests only.

- [ ] **Step 2 (RUN ON THE 1080): Calibrate the `gpu_local` candidates.** For each candidate tier-`gpu_local` test, run it on the 1080 and confirm it fits ~7 GB:

```bash
cd "$WT" && uv sync --extra gpu-pascal
# Example: run a forward/inspection candidate and observe peak VRAM.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run pytest tests/integration/test_load_sam31_real.py -v --no-cov 2>&1 | tail -20
```

Record the measured peak (or OOM) per candidate. A candidate that OOMs on the 1080 is reclassified `gpu_t4`.

- [ ] **Step 3: Identify the move-to-CPU candidates.** Any GPU test whose assertion is purely structural (shape checks, marker presence, config plumbing) and does not require real SAM3.1 weights/kernels can move to CPU via `TinySam3Stub`. List them with the exact replacement plan. (Do NOT move tests whose signal is real-kernel/real-weight behavior — those stay GPU.)

- [ ] **Step 4: Commit the audit doc.**

```bash
cd "$WT" && uv run ruff format --check docs/ 2>/dev/null; git add docs/testing/gpu-audit-2026-05-24.md
git commit -m "docs(testing): #117 GPU test tier-classification audit (calibrated on GTX 1080)"
```

**Completion criteria:** all **12** GPU-tagged tests appear with tier + decision + rationale; `gpu_local` assignments calibrated on the 1080 (one file per process); the training tier noted PROVISIONAL pending #137; move-to-CPU candidates have a concrete replacement plan.

---

## Task D-2: Apply the marker swap + perform the CPU moves + report coverage delta

**Workstream D. Marker swap touches every GPU test file + retires legacy markers (shares conftest/pyproject with B-4 → serialize). CPU moves are TDD/CPU-runnable.** Spec §7.2, §7.3. **The 80% coverage gate is on the FULL suite — D's CPU moves must not regress it.**

**Difficulty:** Medium. **Subagent:** implementer (sonnet/high); the disjoint CPU-move refactors may be dispatched in parallel per `superpowers:dispatching-parallel-agents` (file-disjoint test files, no shared state).

**Files:**
- Modify: every GPU test file (swap `pytest.mark.gpu`/`gpu_inspection` → the D-1-assigned tier marker)
- Modify: `tests/conftest.py` + `pyproject.toml` (retire the legacy `gpu`/`gpu_inspection` marker registrations once no test carries them)
- Create/move: the CPU-test files for the move-to-CPU decisions
- Delete: any test D-1 marked `delete` (with rationale)

- [ ] **Step 1: Capture the BEFORE coverage number.**

```bash
cd "$WT" && uv run pytest -q 2>&1 | tee /tmp/cov_before.txt | tail -5
```

Record the `TOTAL` coverage % from the report (this is the FULL-suite number the 80% gate enforces).

- [ ] **Step 2: Swap the tier markers on every GPU test file.** For each file, replace the legacy `pytestmark` (e.g. `pytest.mark.gpu` or `pytest.mark.gpu_inspection`) with the assigned tier marker from D-1. Each test must carry **exactly one** tier marker and NO legacy `gpu`/`gpu_inspection` marker. Keep the orthogonal capability gates (`requires_compatible_gpu`, `requires_checkpoint`, `requires_bnb`).

  Example (`tests/integration/test_load_sam31_real.py`): `pytest.mark.gpu_inspection` → `pytest.mark.gpu_local` (if D-1 classified it local).

- [ ] **Step 3: Perform the move-to-CPU refactors** D-1 called for. For each, write the CPU test (TDD: failing first via `TinySam3Stub`/synthetic tensors), confirm it passes on CPU, and remove the GPU original (or downgrade it). Each moved test must assert equivalent structural behavior.

- [ ] **Step 4: Retire the legacy marker registrations.** Once `grep -rn "mark.gpu\b\|mark.gpu_inspection" tests/` returns nothing, remove the `gpu` and `gpu_inspection` lines from `pyproject.toml` markers (lines 128-129) and the `gpu_inspection` block from `tests/conftest.py::pytest_configure` (lines 32-36).

```bash
cd "$WT" && grep -rn "mark\.gpu\b\|mark\.gpu_inspection\|gpu_inspection" tests/ src/ | grep -v "gpu_local\|gpu_t4\|gpu_xl" || echo "no legacy markers remain"
```

Expected: `no legacy markers remain` (then it is safe to drop the registrations). Confirm `--strict-markers` still passes (no unknown-marker errors).

- [ ] **Step 5: Capture the AFTER coverage number and confirm no regression below 80%.**

```bash
cd "$WT" && uv run pytest -q 2>&1 | tee /tmp/cov_after.txt | tail -5
```

Expected: full-suite green, coverage ≥ 80% AND ≥ the BEFORE number (CPU moves should hold or improve coverage). If coverage dropped below 80%, the move removed real coverage — add the CPU assertion back or keep the test GPU.

- [ ] **Step 6: Lint gate + commit.**

```bash
cd "$WT" && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
git add -A
git commit -m "refactor(tests): #117 tier-marker swap + CPU moves; retire legacy gpu markers"
```

**Completion criteria:** every GPU test carries exactly one tier marker, no legacy marker; CPU moves land and pass; legacy registrations retired; `--strict-markers` clean; BEFORE/AFTER coverage reported and ≥ 80% with no regression; lint gate green.

---

# Workstream E — #116 full (notebook coverage)

> Spec §8. Depends on B-4 (runner) and D (final tier assignments). **Notebook-cell authoring is disjoint from D's source edits (parallelizable); `gpu-test-policy.md` and `run_gpu_tests.sh` edits serialize with B-4/D.**

## Task E-1: Notebook coverage matrix + cells for unreferenced tests + policy refresh

**Workstream E. CPU-editable.** Spec §8.2, §8.3, §9.

**Difficulty:** Medium (notebook JSON + policy doc). **Subagent:** implementer (sonnet/high — notebook JSON warrants care).

**Files:**
- Modify: `notebooks/colab_gpu_tests.ipynb` (tier cells + cells/exclusions for the unreferenced tests + coverage-matrix markdown; **drop the GC Phase-0/Phase-3 cells**)
- Modify: `docs/testing/gpu-test-policy.md` (three-tier taxonomy; CC 6.0 floor; float16-on-Pascal; refresh inventory to the **12** GC-free tests, fix stale "12 tests" + per-tier counts)

**Context:** the notebook (cells 9-13) runs tiers via `%%bash scripts/run_gpu_tests.sh` (currently `inspection`/`release`). Tests NOT referenced by any cell (spec §8.1): `test_real_train_qlora_resume.py`, `test_channel_adapter_gpu.py`, `test_multiplex_vram.py`, `test_predict_nchannel_gpu.py`, `test_calibrate_real.py`, `tests/predict/test_gpu_predict.py`. The notebook has Phase-0/Phase-3 grad-ckpt cells (14-17) from the merged #127 — **these reference abandoned GC work and must be DROPPED** (GC is abandoned; `test_grad_checkpointing.py` is removed). Use the `NotebookEdit` tool (load its schema via ToolSearch) for cell edits, not hand-JSON.

- [ ] **Step 1: Update the tier-runner cells + drop the GC cells.** Change the `%%bash scripts/run_gpu_tests.sh inspection|release|all` invocations to `local|t4|xl`. Local cells are informational on Colab (the 1080 is not the Colab runtime) — annotate which tier each cell targets. **Delete the GC Phase-0/Phase-3 grad-ckpt cells (14-17)** — they reference the abandoned GC fix and the removed `test_grad_checkpointing.py`.

- [ ] **Step 2: Add a coverage-matrix markdown cell** mapping every GPU test ↔ its notebook cell ↔ its hardware tier (from D-1), over the corrected **12-test** inventory. For each of the 6 unreferenced tests, EITHER add a cell that runs it under its tier OR document an intentional exclusion (with reason) in the matrix. No gaps: every one of the 12 tests is cell-referenced or explicitly excluded.

- [ ] **Step 3: Confirm the notebook still parses.**

```bash
cd "$WT" && uv run --extra jupyter python -c "import nbformat; nbformat.read('notebooks/colab_gpu_tests.ipynb', as_version=4); print('notebook OK')" 2>/dev/null \
  || uv run python -c "import json; json.load(open('notebooks/colab_gpu_tests.ipynb')); print('notebook JSON OK')"
```

Expected: `notebook OK` (or `notebook JSON OK`).

- [ ] **Step 4: Refresh `gpu-test-policy.md`.** State the CC 6.0 floor; float16-on-Pascal; replace/augment the cost tiers with the three-tier hardware taxonomy (cost/cadence demoted to guidance, not a selection mechanism); refresh the inventory to the **12** GC-free tests and correct the stale "12 tests" claim (line ~22) and the stale per-tier counts (the §2 tier definitions naming "nine ... in tests/integration/" and "release tier (3 tests)"). Note that GC is abandoned (the policy must not reference grad-checkpointing GPU tests).

- [ ] **Step 5: Commit.**

```bash
cd "$WT" && uv run ruff format --check . ; git add notebooks/colab_gpu_tests.ipynb docs/testing/gpu-test-policy.md
git commit -m "docs(#116): notebook tier cells + coverage matrix; refresh gpu-test-policy to 3-tier taxonomy"
```

**Completion criteria:** coverage matrix covers all **12** GPU tests with no gaps (cell-referenced or excluded-with-reason); the GC Phase-0/Phase-3 cells are dropped; notebook selects by `{local,t4,xl}` and parses; `gpu-test-policy.md` states CC 6.0 floor + float16-on-Pascal + three-tier taxonomy + the corrected 12-test inventory (no GC references).

---

# Cross-cutting docs (spec §9)

Documentation is updated **alongside the workstream that produces each fact** (not a separate pass), per spec §13:
- `docs/testing/local-pascal-gpu-testing.md` — created in A-1, milestone evidence filled in A-2.
- `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` — §4.3 milestone (PASS) + the Phase-0 GC diagnostic that established the GC abandonment (already recorded). No Phase-3 GC calibration (deferred to #137).
- `docs/testing/gpu-test-policy.md` — refreshed in E-1 (and the §4.3-fallback negative result, if invoked in A-2).
- `docs/testing/gpu-audit-2026-05-24.md` — the #117 audit, created in D-1.

No standalone docs task; each fact lands with its workstream.

---

# Final verification + PR + follow-ups

## Task F-1: All-green verification (per superpowers:verification-before-completion)

- [ ] **Step 1: CPU full suite green, coverage ≥ 80%.**

```bash
cd "$WT" && uv sync --extra dev && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q 2>&1 | tail -8
```

Expected: all pass; "Required test coverage of 80% reached" (or higher); no `--strict-markers` errors.

- [ ] **Step 2 (RUN ON THE 1080): the `--deselect`-stripped `gpu_local` tier green on the 1080.**

```bash
cd "$WT" && uv sync --extra gpu-pascal
grep -nE -- '(^|[[:space:]])--deselect([[:space:]]|=)' scripts/run_gpu_tests.sh | grep -v '^[0-9]*:#' && echo "STRAY DESELECT — strip before merge" || echo "no stray --deselect"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/run_gpu_tests.sh local 2>&1 | tail -25
```

Expected: no stray `--deselect`; the `gpu_local` tier runs green on the 1080 (its forward-only / inspection members classified local in D-1; one file per process to avoid OOM). There is no QLoRA training smoke in `gpu_local` — the training tier is PROVISIONAL pending #137.

- [ ] **Step 3: Confirm the resolution contract is still intact** (cu130 default untouched):

```bash
cd "$WT" && uv sync && uv run python -c "import torch; print(torch.__version__)"   # Expected: +cu130
```

**Completion criteria:** CPU full suite green at ≥80% coverage; `gpu_local` tier green on the 1080 with no stray `--deselect`; default install still cu130.

## Task F-2: Open the PR

- [ ] Open a ready PR via `gh pr create --assignee @me --label <existing-or-new>` linking the spec (`docs/superpowers/specs/2026-05-24-min-gpu-testing-gtx1080-design.md`) and this plan. The PR body summarizes: the §4.3 milestone result (PASS), the **GC abandonment** (Phase-0 diagnostic finding; PR #127 closed, #89/#60 not-planned; 8 GB training deferred to **#137**), the `use_double_quant` landing (C-1), the #117 coverage before/after over the corrected 12-test inventory, and the three-tier taxonomy with the `gpu_local` training tier marked PROVISIONAL pending #137.

## Task F-3: File the follow-up issues (FILE, do not implement) — spec §12

- [ ] **#137 (already OPEN) — "Investigate non-checkpointing approaches to fit QLoRA training in an 8 GB VRAM budget."** Owns the deferred 8 GB QLoRA *training* recipe (`gpu_smoke_qlora_8gb.yaml`, the training smoke, on-1080 calibration) now that GC is abandoned. The `gpu_local` training tier is PROVISIONAL until #137 resolves. Cross-link it from the PR (no new issue needed — it exists).
- [ ] **"Operationalize the `gpu_t4` tier on Colab"** — `gh issue create --assignee @me --label <label>`: the 14 GB / 10 GB release-ceiling gate + the **bf16 confirmation** that cannot run on the 8 GB Pascal card (Risk 4's deferred bf16-vs-float16 validation; the T4 ceilings this PR does not touch).
- [ ] **`gpu_xl` tier population** — a follow-up (or a comment/cross-link) referencing the existing **#124** (cloud auto-provision) as the `gpu_xl` runner. The tier ships near-empty now, populated when #124 lands.

> The §4.3 milestone **PASSED**, so the Pascal-track-blocked fallback follow-ups are moot. The #127 GC fix (#89/#60) is **closed not-planned** — no follow-up; the diagnostic proved it unfixable without editing forbidden sam3 source (see the manual-pass evidence doc).

---

## Self-Review (run by the planner)

**Spec coverage (reduced — GC abandoned, 8 GB training deferred to #137):**
- §3 / §3.1 (three-tier taxonomy, marker mechanics) → B-4.
- §4 (Pascal env, §4.2 uv tables, §4.3 hard-gate milestone — PASSED, §4.4 acceptance) → A-1, A-2.
- §5.1 (CC 6.0 floor + skip reasons) → B-1; §5.2 (dtype coercion, both seams) → B-2; §5.3 (preset fidelity) → B-3; §5.4 (tier wiring, runner, `tests/predict/`) → B-4.
- §6.2 (`use_double_quant`) → **C-1** (landed). §6.1 (GC fix) → **C-2 ABANDONED** (Phase-0 diagnostic proved it infeasible without editing forbidden sam3 source). §6.3 (8 GB training recipe) → **C-3 DEFERRED to #137**; §6.4 (`gpu_local` training tier PROVISIONAL) + Phase-3 calibration → **C-4 DEFERRED to #137**. §6.5 (float16 stability) → no separate task (no GradScaler; carried by C-1/B-2). There is no longer a §6.x or §6.8 in the spec.
- §7 (#117 audit + CPU moves + coverage delta, **12-test** inventory) → D-1, D-2.
- §8 (#116 notebook coverage matrix, 12 tests, GC cells dropped) → E-1.
- §9 (docs) → folded into A/C/D/E as noted in "Cross-cutting docs."
- §12 (follow-ups: #137, gpu_t4, gpu_xl/#124; GC fix closed not-planned) → F-3. §13 (sequencing A→gate→B→{C(C-1 only),D,E after D}) → workstream order + the A-2 hard gate.

**Task disposition after the amendment:** C-1 KEEP (landed); **C-2 ABANDONED** (no work product — the diagnostic that justified abandonment is already recorded in the manual-pass doc); **C-3 / C-4 DEFERRED to #137** (`gpu_smoke_qlora_8gb.yaml` and `test_real_train_qlora_8gb.py` never created). The GC-free rebuild removes `vit_act_checkpoint.py` and `tests/gpu/test_grad_checkpointing.py`.

**Corrected GPU-test-file count (verified against the tree):** `grep -rl "mark\\.gpu" tests/ --include="*.py"` reports 13 today including `test_grad_checkpointing.py`; the GC-free rebuild drops `test_grad_checkpointing.py` and the 8 GB test was never added, leaving **12** GPU-tagged files (8 in `tests/gpu/` + 3 in `tests/integration/` + 1 in `tests/predict/`). The File map, D-1 inventory, and E-1 matrix all state 12.

**Verified anchors (confirmed against the tree):** GTX 1080 sm_61 live + cu130 torch; `tests/conftest.py` CC floor `(7,5)` + `gpu_inspection`-only registration; `presets.py` lines 83/120/142/295/335; `loop.py:186-192` `_autocast_ctx`; `qlora.py:139-155` direct `Linear4bit` + `_QLORA_META_VERSION=1`; `schema.py` `Dtype` already `["bfloat16","float16"]`, `Optimizer` literal line 97, `QLoRAConfig` line 486-488; `pyproject.toml` markers + 80% gate; `run_gpu_tests.sh` `inspection|release|all` omitting `tests/predict/`; 12 GC-free GPU test files (8 gpu/ + 3 integration/ + 1 predict/); `test_presets.py` label/round-trip tests do not assert the dtype token (safe to widen).

**Could not verify:** none material to the reduced scope.
