# QLoRA 8 GB Fit Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a pre-registered, serialized GPU protocol on the real GTX 1080 (sm_61, fp16) to decide whether one QLoRA fwd+bwd+optim step of SAM 3.1 multiplex fits in ~7 GB usable using non-offload levers, and ship a findings report (plus a conditionally-committed Pascal config and a manual-pass placeholder link).

**Architecture:** This is a *measurement investigation*, not a feature build. There is no production-code TDD: each "test" is a GPU measurement, and each task's success criterion is a *recorded number/observation*, not a passing unit test. Four GPU runs (A–D) execute strictly one-at-a-time, each in its own process/file, with the ~3.3 GB checkpoint released between files. Per-Run measurement scripts are **ephemeral** (throwaway, under a gitignored scratch path, results captured into the report) — see "Scripts: ephemeral vs committed" below. The only committed deliverables are the findings report, a conditional config (only on FIT), and a placeholder edit in the manual-pass doc. Branching control flow (Run A early-exit; Run C FIT/NO-FIT gate) is explicit in the task ordering.

**Tech Stack:** Python, PyTorch 2.7.1+cu118 (sm_61 via PTX JIT), bitsandbytes 0.49.2, `uv` with the `gpu-pascal` extra, the project's `run_training(cfg)` seam and `load_sam31(cfg)` loader, `torch.cuda.reset_peak_memory_stats()` + `torch.cuda.max_memory_allocated()` for VRAM (no `nvidia-smi`).

---

## Source of truth

Spec: `docs/superpowers/specs/2026-05-24-qlora-8gb-fit-investigation-design.md`. The plan covers exactly its scope — Runs A–D, the early-exit rule (§4.1), the decision gate (§4.3), the ablation (§4.4), the §4.5 hypothesis, and the three deliverables (§7). Nothing in §2.2 / §9 (out of scope) gets a task.

## Hard constraints (apply to every GPU task)

- **One GPU run at a time, serialized.** Never run two GPU scripts concurrently. Each Run is its own process/file; let the process fully exit (releasing the ~3.3 GB checkpoint) before starting the next.
- **No `nvidia-smi`.** Peak VRAM is measured *only* with `torch.cuda.reset_peak_memory_stats()` then `torch.cuda.max_memory_allocated() / 1e9` (GB). This mirrors `tests/gpu/test_real_train_qlora.py:60-62` and `tests/gpu/test_multiplex_vram.py:40-43`.
- **Allocator config.** Every GPU script must run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the environment.
- **Env.** Reach the GPU *only* via `uv run --extra gpu-pascal python <script>` (torch 2.7.1+cu118 + bitsandbytes 0.49.2). The default cu130 torch has no sm_61 cubin and cannot run on this card.
- **Ceiling.** Target is **~7.0 GB usable**, not the 8 GB nameplate (WSL holds ~1 GB).
- **fp16 only.** Pascal has no bf16 hardware; `coerce_dtype_for_capability` (`src/custom_sam_peft/runtime/_runtime.py:61`) coerces a `bfloat16` config to `float16` on sm_61. Do not add a GradScaler; the QLoRA path runs fp16 directly at the `Linear4bit` compute dtype and relies on the loop's existing NaN-skip / `nan_abort_after` guards.
- **Do not edit `src/`, test markers, or `gpu-test-policy.md`.** Do not re-enable/fix gradient checkpointing. Do not change `image_size` (fixed 1008). Do not edit the external sam3 package.
- **Checkpoint.** SAM 3.1 checkpoint (`models/sam3.1/sam3.1_multiplex.pt`) is reached via the worktree's symlink to the main repo's `models/sam3.1/`. `load_sam31(cfg)` (`src/custom_sam_peft/models/sam3.py:744`) loads it.

## Scripts: ephemeral vs committed (decision, stated explicitly)

**Decision: the per-Run measurement scripts are EPHEMERAL, not committed.** Rationale: the spec's §7 names exactly three committed deliverables (report, conditional config, placeholder edit); the scripts are *means* to those ends, and §9 explicitly defers any new committed `gpu_local` test to a follow-up. Committing throwaway probe scripts would (a) imply a test surface the spec excludes and (b) require a marker/policy decision that is out of scope. Therefore:

- Write each Run's script under a gitignored scratch directory: `runs/_investigation_137/` (the repo already gitignores `runs/` — verify in Task 0). If `runs/` is *not* gitignored, write scripts under `/tmp/qlora_137/` instead.
- Capture every script's stdout into a log file under that same scratch dir (e.g. `runs/_investigation_137/runA.log`), and transcribe the recorded numbers into the report (Task R). Do **not** `git add` any script or log.
- Keep scripts minimal — a single `if __name__ == "__main__":` block, no helper packages, no new fixtures. Reuse `load_config`, `run_training`, `load_sam31`, and the `tests/gpu/conftest.py` `_RecordingTracker` / `_bnb_available` helpers by import.

## Known scaffolding gaps the scripts must work around (do NOT fix in `src/`)

- **`paged_adamw_8bit` is not a wired optimizer.** `Optimizer = Literal["adamw", "adamw8bit", "auto"]` (`src/custom_sam_peft/config/schema.py:97`) and `_build_optimizer` (`src/custom_sam_peft/train/trainer.py:49-64`) only handle `adamw`/`adamw8bit`. To exercise the paged-optimizer lever for Runs C/D, the ephemeral script monkeypatches `_build_optimizer` (or patches the optimizer at the `custom_sam_peft.train.trainer` namespace) to return `bitsandbytes.optim.PagedAdamW8bit(params, lr=lr)`. This is a *measurement-only* patch inside the throwaway script — **not** a `src/` change.
- **No "decoder-only" scope literal.** `LoraScope = Literal["vision", "vision_decoder", "all"]` (`src/custom_sam_peft/config/schema.py:101`). `vision_decoder` (`src/custom_sam_peft/peft_adapters/lora.py:52-56`) includes the ViT-trunk `qkv|proj` pattern. To get the "narrowest available trainable scope" (decoder-only) for Run C, set `peft.target_modules` to *just* the two decoder patterns (dropping the trunk pattern) — `target_modules` overrides `scope` per `lora.py:68`:
  - `transformer\.decoder\.layers\.\d+\.(self_attn|cross_attn|ca_text)\.out_proj$`
  - `transformer\.decoder\.layers\.\d+\.linear[12]$`

---

## Task 0: Provision the GPU environment and scratch dir

**Files:**

- No source files. Environment + scratch setup only.

- [ ] **Step 1: Provision the gpu-pascal env**

Run: `uv sync --extra gpu-pascal`
Expected: resolves torch 2.7.1+cu118 and bitsandbytes 0.49.2 without error.

- [ ] **Step 2: Confirm the GPU is reachable and is sm_61 (no nvidia-smi)**

Run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --extra gpu-pascal python -c \
"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Expected: `True ... NVIDIA GeForce GTX 1080 (6, 1)`. If `(6, 1)` is not reported, halt and report — the wrong torch/extra is active.

- [ ] **Step 3: Confirm the checkpoint symlink resolves**

Run: `ls -lL models/sam3.1/sam3.1_multiplex.pt`
Expected: the file resolves (follows the symlink to the main repo) and is ~3.3 GB.

- [ ] **Step 4: Create the scratch dir and confirm it is gitignored**

Run: `mkdir -p runs/_investigation_137 && git check-ignore runs/_investigation_137 && echo IGNORED || echo NOT_IGNORED`
Expected: `IGNORED`. If `NOT_IGNORED`, use `/tmp/qlora_137/` for all scripts/logs in subsequent tasks instead. **Record which path is used** — it is referenced by every later task as `<SCRATCH>`.

**Success criterion (recorded observation):** env provisions, GPU reports capability `(6, 1)`, checkpoint resolves, and `<SCRATCH>` is chosen and confirmed gitignored.

---

## Task A: Run A — Stage 0 cheap probes (SDPA backend + static floor)

> Single GPU process. Sub-few-minutes. No training loop. Triggers the **early-exit branch** (Task A-EXIT) if the static floor alone exceeds ~7 GB.

**Files:**

- Create (ephemeral): `<SCRATCH>/runA.py`
- Log (ephemeral): `<SCRATCH>/runA.log`

- [ ] **Step 1: Write the Run A probe script**

The script performs three measurements in one process and prints clearly-labelled lines. Use synthetic tensors for the SDPA probe where possible to keep it sub-minute; load the 4-bit base for the static/forward floor.

```python
# <SCRATCH>/runA.py  — EPHEMERAL, do not commit
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

# --- A(a): SDPA backend probe (synthetic, representative shapes) ---
# Representative shapes (Blocker-2 trace):
#   ViT-trunk self-attention: 5184 image tokens, multi-head.
#   Decoder cross-attention: 34 decoder queries x 5184 image tokens, 8 heads.
dev = "cuda"
dt = torch.float16  # Pascal fp16

def try_backend(name, backend, q, k, v):
    try:
        with sdpa_kernel([backend]):
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        return f"{name}: RAN ok, out {tuple(o.shape)}"
    except Exception as e:  # noqa: BLE001 - probe wants the failure text
        return f"{name}: RAISED {type(e).__name__}: {e}"

# decoder cross-attention shape: (batch, heads, q_len, head_dim)
q = torch.randn(1, 8, 34, 64, device=dev, dtype=dt)
k = torch.randn(1, 8, 5184, 64, device=dev, dtype=dt)
v = torch.randn(1, 8, 5184, 64, device=dev, dtype=dt)
print("A(a) cross-attn EFFICIENT:", try_backend("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION, q, k, v))
print("A(a) cross-attn MATH:", try_backend("MATH", SDPBackend.MATH, q, k, v))
print("A(a) cross-attn FLASH:", try_backend("FLASH", SDPBackend.FLASH_ATTENTION, q, k, v))

# ViT-trunk self-attention shape (5184 self-attention)
qs = torch.randn(1, 8, 5184, 64, device=dev, dtype=dt)
print("A(a) trunk-self EFFICIENT:", try_backend("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION, qs, qs, qs))
print("A(a) trunk-self MATH:", try_backend("MATH", SDPBackend.MATH, qs, qs, qs))

# Default (unpinned) dispatch — which backend does the real forward take?
# Note that PyTorch picks at runtime; record what runs without a pinned context.
try:
    o = torch.nn.functional.scaled_dot_product_attention(qs, qs, qs)
    torch.cuda.synchronize()
    print("A(a) trunk-self DEFAULT(unpinned): RAN ok, out", tuple(o.shape))
except Exception as e:  # noqa: BLE001
    print("A(a) trunk-self DEFAULT(unpinned): RAISED", type(e).__name__, e)

# --- A(b): static post-load floor + forward-only peak ---
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.peft_adapters.qlora import apply_qlora  # adjust import to actual entry if needed
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.config.schema import ModelConfig

# Static floor: load 4-bit base + LoRA, NO forward.
torch.cuda.reset_peak_memory_stats()
# Build the model exactly as run_training would for the smoke config (QLoRA, fp16 coerced).
# Reuse the smoke config so the wrapped model matches Run B/C topology.
cfg = load_config(
    "configs/examples/gpu_smoke_qlora.yaml",
    overrides=["model.device=cuda"],
)
# Load + apply QLoRA via the same code path run_training uses (model load + apply_qlora).
# (If apply_qlora's exact signature differs, mirror what train/runner.py does to build the peft model.)
wrapper = load_sam31(ModelConfig(device="cuda", gradient_checkpointing=False, dtype=cfg.model.dtype))
peft_model = apply_qlora(wrapper, cfg.peft)  # 4-bit base + LoRA, on GPU
torch.cuda.synchronize()
static_floor_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"A(b) static_post_load_floor_GB: {static_floor_gb:.3f}")

# Forward-only peak: one no-grad forward at representative input.
from custom_sam_peft.data.base import TextPrompts
images = torch.zeros(1, 3, 1008, 1008, dtype=torch.float16, device="cuda")
prompts = [TextPrompts(classes=["class_0"])]
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = wrapper(images, prompts)
torch.cuda.synchronize()
fwd_only_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"A(b) forward_only_peak_GB: {fwd_only_gb:.3f}")
```

Note for the implementer: the exact `apply_qlora` / peft-build call must mirror what `src/custom_sam_peft/train/runner.py` does between loading the model and `Trainer.fit` (read that file to copy the precise build sequence). The probe must NOT run a training loop.

- [ ] **Step 2: Run Run A on the GPU and capture the log**

Run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --extra gpu-pascal python <SCRATCH>/runA.py 2>&1 | tee <SCRATCH>/runA.log
```

Expected: the process prints the `A(a)` backend lines (RAN/RAISED for EFFICIENT / MATH / FLASH on both shapes, plus the unpinned default), then `A(b) static_post_load_floor_GB:` and `A(b) forward_only_peak_GB:`, then exits cleanly (releasing the checkpoint).

- [ ] **Step 3: Record the Run A observations**

Record into a scratch notes file (or the orchestrator's running notes — these feed Task R):

- For each SDPA backend on each representative shape: did it RUN or RAISE (and the exception type/message)?
- Which backend the **default unpinned** dispatch took.
- The **`<EFFICIENT togglable?>`** conclusion: is EFFICIENT_ATTENTION usable on sm_61 (RAN ok), or is MATH effectively forced? This decides Run D's ablated lever (Task D).
- `static_post_load_floor_GB` (the irreducible floor).
- `forward_only_peak_GB`.

**Success criterion (recorded observation):** the three Run A numbers/answers above are recorded, AND a determination of whether `static_post_load_floor_GB > ~7.0`.

- [ ] **Step 4: Evaluate the early-exit branch**

If `static_post_load_floor_GB` **> ~7.0 GB** → the verdict is **NO-FIT on the static floor alone**. Go to **Task A-EXIT** and then skip Tasks B, C, D entirely (Run D is explicitly skipped when Run A early-exits). Otherwise continue to **Task B**.

---

## Task A-EXIT: Early-exit branch (only if Run A static floor > ~7 GB)

> Reached ONLY when Task A Step 4 fires the early-exit. If not fired, skip this task.

**Files:**

- No new files here. This task records the branch decision; the report itself is Task R.

- [ ] **Step 1: Record the early-exit verdict**

Record: "NO-FIT — static post-load floor alone (`<static_post_load_floor_GB>` GB) exceeds the ~7.0 GB ceiling. Runs B–D not run (would only add training state atop an already-overflowing floor)." Carry `static_post_load_floor_GB` and `forward_only_peak_GB` and the Run A(a) SDPA finding forward to Task R.

- [ ] **Step 2: Proceed**

Skip Tasks B, C, D. Go directly to **Task R** (report). The conditional-config task (Task C-CONFIG) resolves to the NO-FIT (appendix-only) branch.

**Success criterion (recorded observation):** the NO-FIT-on-floor verdict and the floor number are recorded, and Tasks B/C/D are marked not-run.

---

## Task B: Run B — Stage 1 baseline (as-is smoke peak)

> Single GPU process. Only runs if Run A did NOT early-exit. A baseline OOM is a recorded datum, NOT an early-exit — continue to Run C regardless.

**Files:**

- Create (ephemeral): `<SCRATCH>/runB.py`
- Log (ephemeral): `<SCRATCH>/runB.log`

- [ ] **Step 1: Write the Run B baseline script**

Reuse the `tests/gpu/test_real_train_qlora.py::test_qlora_smoke_fast` mechanics (~3 steps via `train.epochs=2`, `train.log_every=1`, Evaluator no-op) at the **as-is** smoke config — fp16 (coerced from `bfloat16`), scope `vision_decoder`, optimizer `adamw8bit`, **no** double-quant. Only fixture/output + fast-smoke overrides; no lever changes.

```python
# <SCRATCH>/runB.py  — EPHEMERAL, do not commit
import math, torch
from pathlib import Path
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.train.runner import run_training
from custom_sam_peft.eval.metrics import MetricsReport
import custom_sam_peft.train.trainer as trainer_mod
from tests.gpu.conftest import _RecordingTracker

FIX = Path("tests/fixtures/tiny_coco")  # confirm path; tiny_coco_dir fixture -> FIXTURES/"tiny_coco"
OUT = Path("<SCRATCH>/runB_out")
cfg = load_config(
    "configs/examples/gpu_smoke_qlora.yaml",
    overrides=[
        f"data.train.annotations={FIX/'annotations.json'}",
        f"data.train.images={FIX/'images'}",
        f"data.val.annotations={FIX/'annotations.json'}",
        f"data.val.images={FIX/'images'}",
        f"run.output_dir={OUT}",
        "train.epochs=2",
        "train.log_every=1",
    ],
)

tracker = _RecordingTracker()
import custom_sam_peft.train.runner as runner_mod
runner_mod.build_tracker = lambda *a, **k: tracker  # patch tracker

class _SkipEval:
    def __init__(self, *a, **k): pass
    def evaluate(self, *a, **k): return MetricsReport()
trainer_mod.Evaluator = _SkipEval  # no-op eval

torch.cuda.reset_peak_memory_stats()
status = "completed"
try:
    run_training(cfg)
except torch.cuda.OutOfMemoryError as e:
    status = f"OOM: {e}"
peak_gb = torch.cuda.max_memory_allocated() / 1e9
finite = all(math.isfinite(v) for _, s in tracker.scalars for v in s.values()) if tracker.scalars else None
print(f"B status: {status}")
print(f"B peak_GB: {peak_gb:.3f}")
print(f"B loss_finite: {finite}  (scalars logged: {len(tracker.scalars)})")
```

Implementer note: confirm the tracker-patch and Evaluator-patch target the same namespaces the fast smoke uses (`custom_sam_peft.train.runner.build_tracker`, `custom_sam_peft.train.trainer.Evaluator`). Confirm the fixture path via `tests/conftest.py:146-148`.

- [ ] **Step 2: Run Run B on the GPU and capture the log**

Run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --extra gpu-pascal python <SCRATCH>/runB.py 2>&1 | tee <SCRATCH>/runB.log
```

Expected: prints `B status:` (completed / OOM), `B peak_GB:`, `B loss_finite:`. Expected outcome per spec is ~10 GB or OOM at the ~7 GB ceiling. Process exits, releasing the checkpoint.

- [ ] **Step 3: Record the Run B observations**

Record: baseline `peak_GB` (or "OOM at ~7 GB ceiling" if it OOM'd — note the exact peak is then a lower bound only), whether the step completed/OOM'd, and whether loss stayed finite.

**Success criterion (recorded observation):** baseline peak (or OOM datum) + completion status + finiteness recorded. **Always continue to Task C** (baseline OOM is not an early-exit).

---

## Task C: Run C — Stage 2 all non-offload levers stacked (the decisive run)

> Single GPU process. 2–3 consecutive steps; take the MAX peak across them (fragmentation guard). Triggers the **FIT / NO-FIT decision gate** (Task C-CONFIG).

**Files:**

- Create (ephemeral): `<SCRATCH>/runC.py`
- Log (ephemeral): `<SCRATCH>/runC.log`

- [ ] **Step 1: Write the Run C stacked-levers script**

Same `run_training` + fast-smoke mechanics as Run B, but with **every non-offload lever ON** (spec §5):

- fp16 model + `compute_dtype` (coerced from `bfloat16`; Pascal-required).
- `peft.qlora.use_double_quant=true` (NF4 nested quant).
- Optimizer: `paged_adamw_8bit` — **wired via the in-script monkeypatch** described in "Known scaffolding gaps". Patch `_build_optimizer` in the `custom_sam_peft.train.trainer` namespace so the `adamw8bit` branch returns `bitsandbytes.optim.PagedAdamW8bit(params, lr=lr)`; set `train.optimizer=adamw8bit` in config so the patched branch is hit.
- Narrowest trainable scope (decoder-only): set `peft.target_modules` to the two decoder patterns (drop the trunk pattern), per "Known scaffolding gaps".
- Smaller LoRA rank and fewer target modules: e.g. `peft.r=8` (down from 16) and the reduced `target_modules` above.
- `train.batch_size=1`; grad-accum unused (VRAM-neutral at batch 1).
- 2–3 steps via `train.epochs=2`, `train.log_every=1`, Evaluator no-op.

```python
# <SCRATCH>/runC.py  — EPHEMERAL, do not commit
import math, torch
from pathlib import Path
import bitsandbytes as bnb
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.train.runner import run_training
from custom_sam_peft.eval.metrics import MetricsReport
import custom_sam_peft.train.trainer as trainer_mod
import custom_sam_peft.train.runner as runner_mod
from tests.gpu.conftest import _RecordingTracker

# --- monkeypatch paged optimizer (measurement-only; NOT a src change) ---
_orig_build = trainer_mod._build_optimizer
def _paged_build(name, params, lr):
    if name == "adamw8bit":
        return bnb.optim.PagedAdamW8bit(params, lr=lr)
    return _orig_build(name, params, lr)
trainer_mod._build_optimizer = _paged_build

FIX = Path("tests/fixtures/tiny_coco")
OUT = Path("<SCRATCH>/runC_out")
DECODER_ONLY = [
    r"transformer\.decoder\.layers\.\d+\.(self_attn|cross_attn|ca_text)\.out_proj$",
    r"transformer\.decoder\.layers\.\d+\.linear[12]$",
]
cfg = load_config(
    "configs/examples/gpu_smoke_qlora.yaml",
    overrides=[
        f"data.train.annotations={FIX/'annotations.json'}",
        f"data.train.images={FIX/'images'}",
        f"data.val.annotations={FIX/'annotations.json'}",
        f"data.val.images={FIX/'images'}",
        f"run.output_dir={OUT}",
        "train.epochs=2",
        "train.log_every=1",
        "train.optimizer=adamw8bit",      # -> patched to PagedAdamW8bit
        "peft.qlora.use_double_quant=true",
        "peft.r=8",
        # target_modules override (decoder-only) — pass via a programmatic edit if
        # the override CLI cannot express a list; see note below.
    ],
)
# If load_config overrides cannot set a list, set it on the parsed object:
cfg.peft.target_modules = DECODER_ONLY

tracker = _RecordingTracker()
runner_mod.build_tracker = lambda *a, **k: tracker
class _SkipEval:
    def __init__(self, *a, **k): pass
    def evaluate(self, *a, **k): return MetricsReport()
trainer_mod.Evaluator = _SkipEval

torch.cuda.reset_peak_memory_stats()
status = "completed"
try:
    run_training(cfg)
except torch.cuda.OutOfMemoryError as e:
    status = f"OOM: {e}"
peak_gb = torch.cuda.max_memory_allocated() / 1e9  # max across all steps
finite = all(math.isfinite(v) for _, s in tracker.scalars for v in s.values()) if tracker.scalars else None
print(f"C status: {status}")
print(f"C peak_GB(max across steps): {peak_gb:.3f}")
print(f"C loss_finite: {finite}  (scalars logged: {len(tracker.scalars)})")
```

Implementer note: `max_memory_allocated()` is already the max over the whole `run_training` call, so it inherently captures the peak across the 2–3 steps (the fragmentation guard). If you also want to *see* a rising-vs-plateau trend, log per-step peaks by resetting between steps — but the gate uses the single max. Verify whether the `load_config` override syntax supports list values for `peft.target_modules`; if not, set it on the parsed `cfg` object as shown.

- [ ] **Step 2: Run Run C on the GPU and capture the log**

Run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --extra gpu-pascal python <SCRATCH>/runC.py 2>&1 | tee <SCRATCH>/runC.log
```

Expected: prints `C status:`, `C peak_GB(max across steps):`, `C loss_finite:`, then exits.

- [ ] **Step 3: Record the Run C observations and apply the decision gate**

Record: `peak_GB` (the minimum achieved peak under stacked levers), completion/OOM status, loss finiteness. Then apply the **pre-registered decision gate**:

- **FIT** ⇔ `peak_GB ≤ ~7.0` **AND** no OOM **AND** finite loss across the 2–3 steps.
- **Otherwise NO-FIT** — record the **floor** (minimum achieved peak) and what limited it: `OOM` | `>7 GB peak` | `non-finite loss`.

Record the verdict. This drives Task C-CONFIG (FIT → commit config; NO-FIT → appendix-only).

**Success criterion (recorded observation):** Run C peak, status, finiteness, and the FIT/NO-FIT verdict (with floor + limiting cause if NO-FIT) recorded.

---

## Task D: Run D — Stage 3 single-lever ablation

> Single GPU process. Only runs if Run A did NOT early-exit (Run D is skipped on early-exit). The ablated lever is decided by Run A(a).

**Files:**

- Create (ephemeral): `<SCRATCH>/runD.py`
- Log (ephemeral): `<SCRATCH>/runD.log`

- [ ] **Step 1: Choose the ablated lever from Run A(a) (pre-registered rule)**

- **If A(a) found EFFICIENT_ATTENTION runs on sm_61 and is togglable** → ablate the **SDPA backend**: re-run the Run C config but force MATH (wrap the forward in `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])`, e.g. via a context patch around the model forward), or force EFFICIENT if Run C ran under MATH — i.e. flip whichever backend Run C used. A large peak swing ⇒ **activation-bound**.
- **Else** (MATH effectively forced on sm_61) → ablate the **optimizer-state lever**: revert the paged-optimizer monkeypatch so the `adamw8bit` branch returns the plain `bnb.optim.AdamW8bit` (or a 32-bit `torch.optim.AdamW`), keeping all other Run C levers. A large peak swing ⇒ **optimizer/grad-bound**.

Record the chosen lever name before running.

- [ ] **Step 2: Write the Run D script**

Copy `<SCRATCH>/runC.py` to `<SCRATCH>/runD.py` and toggle exactly the one chosen lever from Step 1 (everything else identical to Run C). Keep the same 2–3-step measurement and the same printed labels (`D status:`, `D peak_GB(max across steps):`, `D loss_finite:`).

For the SDPA-backend ablation, wrap the training step's forward in the chosen `sdpa_kernel([...])` context. If a clean wrap point in `run_training` is not exposed, the simplest faithful approach is to enter the `sdpa_kernel` context around the entire `run_training(cfg)` call (the context propagates into the forward). Verify the context actually pins the backend by reusing the Run A(a) probe assertion if uncertain.

- [ ] **Step 3: Run Run D on the GPU and capture the log**

Run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --extra gpu-pascal python <SCRATCH>/runD.py 2>&1 | tee <SCRATCH>/runD.log
```

Expected: prints `D status:`, `D peak_GB(max across steps):`, `D loss_finite:`, then exits.

- [ ] **Step 4: Record the Run D observations and the attribution**

Record: the ablated lever (named), Run D `peak_GB`, the **delta** `Run C peak − Run D peak`, and the attribution conclusion: a large swing on the SDPA lever ⇒ **activation-bound**; a large swing on the optimizer lever ⇒ **optimizer/grad-bound**. State whether the §4.5 hypothesis (small retained decoder-side footprint + a large transient trunk peak under MATH) is **confirmed or refuted** by Run A(a) + Run D together.

**Success criterion (recorded observation):** ablated lever, Run D peak, delta vs Run C, and the activation-bound vs optimizer/grad-bound attribution recorded, plus the §4.5 adjudication.

---

## Task R: Write the findings report (committed deliverable 1)

> No GPU. Transcribes the recorded numbers from Tasks A–D into the committed report.

**Files:**

- Create: `docs/research/2026-05-24-issue-137-qlora-8gb-feasibility.md`

- [ ] **Step 1: Write the report**

The report MUST contain (spec §7.1), drawing only from the recorded observations of Tasks A–D:

1. **Background / constraints** — restate the hard facts from spec §3 (hardware sm_61 / ~7 GB usable; gpu-pascal env / torch 2.7.1+cu118 / bnb 0.49.2; fp16-only / no bf16; `image_size` fixed 1008; grad-ckpt dead; non-offload levers only; measurement via `reset_peak_memory_stats` + `max_memory_allocated`; current `gpu_t4` classification with `VRAM_CEIL_GB=10.0`).
2. **Run A(a) SDPA-backend result** — which backends RAN/RAISED on sm_61 for the representative shapes; the default unpinned backend; the resolution of the issue-#8-vs-manual-pass-doc contradiction (is mem-efficient usable, or is MATH forced?).
3. **Peak-VRAM table** — rows: static floor + forward-only (Run A(b)), baseline (Run B, or "OOM at ~7 GB ceiling"), stacked (Run C), Run-D ablation. Columns: configuration, peak GB, status (completed/OOM), loss finiteness.
4. **Run-D ablation attribution** — activation-bound vs optimizer/grad-bound, with the Run C→D delta.
5. **fp16-finiteness note** — whether loss stayed finite, explicitly **distinct from OOM**.
6. **§4.5 hypothesis** — stated as a hypothesis the measurements *test*, then adjudicated (confirmed/refuted) against Run A(a) + Run D.
7. **Verdict + minimum-achieved peak VRAM** — FIT or NO-FIT, the minimum achieved peak in GB, and (if NO-FIT) the limiting cause. State that this **informs** (does not change here) the `gpu_t4` classification.
8. **(Conditional) NO-FIT appendix** — only if the verdict is NO-FIT: include the reduced `min_gpu_qlora` config inline here as a **non-shipped appendix** (it is NOT committed under `configs/` in the NO-FIT case). See Task C-CONFIG.

If Run A early-exited (Task A-EXIT), the report records the NO-FIT-on-floor verdict, includes only the Run A rows in the table, and notes Runs B/C/D as not-run; the config goes in the appendix.

- [ ] **Step 2: Markdown-lint the report**

Run: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/research/2026-05-24-issue-137-qlora-8gb-feasibility.md"`
Expected: no findings (fix any).

- [ ] **Step 3: Commit the report**

```bash
git add docs/research/2026-05-24-issue-137-qlora-8gb-feasibility.md
git commit -m "docs(research): QLoRA 8GB fit investigation findings (#137)"
```

**Success criterion (recorded observation):** the report exists, lints clean, contains all §7.1 elements, and is committed.

---

## Task C-CONFIG: Conditional Pascal config (committed deliverable 2)

> Branches on the Run C decision gate (Task C Step 3). FIT → commit the config. NO-FIT (or Run A early-exit) → config lives only as the report appendix (already added in Task R Step 1.8); commit nothing under `configs/`.

**Files:**

- Create (FIT only): `configs/examples/min_gpu_qlora.yaml`

- [ ] **Step 1 (FIT branch): Write `configs/examples/min_gpu_qlora.yaml`**

Only if the verdict is **FIT**. Mirror `configs/examples/gpu_smoke_qlora.yaml` with the Run C levers baked in: `model.dtype: float16`, `peft.qlora.use_double_quant: true`, `peft.qlora.compute_dtype: float16`, the decoder-only `peft.target_modules` list, smaller `peft.r`, `train.batch_size: 1`, `train.grad_accum_steps: 1`. Use the optimizer field the schema supports (`adamw8bit`) and add a YAML comment that the paged variant was the measured lever (since `paged_adamw_8bit` is not a schema literal — do not invent a config value the loader rejects). Name it `min_gpu_qlora` (the `run.name`).

- [ ] **Step 2 (FIT branch): Validate the config loads**

Run: `uv run python -c "from custom_sam_peft.config.loader import load_config; load_config('configs/examples/min_gpu_qlora.yaml'); print('OK')"`
Expected: `OK` (no schema error). Note: this is a CPU-side load validation, not a GPU run — `uv run` without the gpu-pascal extra is fine here.

- [ ] **Step 3 (FIT branch): Markdown/format gate + commit**

```bash
git add configs/examples/min_gpu_qlora.yaml
git commit -m "feat(config): Pascal-tuned min_gpu_qlora QLoRA config (FIT, #137)"
```

- [ ] **Step 1 (NO-FIT branch): No config commit**

If the verdict is **NO-FIT** (including Run A early-exit), do **not** create `configs/examples/min_gpu_qlora.yaml`. Confirm the reduced `min_gpu_qlora` config is present as the non-shipped appendix in the report (Task R Step 1.8). Nothing to commit in this task.

**Success criterion (recorded observation):** on FIT, the config exists, loads without schema error, and is committed; on NO-FIT, no `configs/` file exists and the appendix is confirmed in the report.

---

## Task P: Wire the manual-GPU-pass Phase-3 placeholder (committed deliverable 3)

> No GPU. Populates the waiting placeholder with a link + short summary.

**Files:**

- Modify: `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md:296` (the `<!-- filled by C-4 -->` block under `### Phase-3 calibration numbers`)

- [ ] **Step 1: Replace the placeholder**

Replace the `<!-- filled by C-4 -->` line (and the parenthetical italic stub on lines 298-299) with a link to the findings report (`docs/research/2026-05-24-issue-137-qlora-8gb-feasibility.md`) and a 2–4 sentence summary: the FIT/NO-FIT verdict, the minimum achieved peak VRAM in GB, and the Run-D attribution (activation-bound vs optimizer/grad-bound). Keep it short — the report is the source of detail.

- [ ] **Step 2: Markdown-lint the edited doc**

Run: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md"`
Expected: no findings (fix any).

- [ ] **Step 3: Commit**

```bash
git add docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md
git commit -m "docs(testing): wire Phase-3 placeholder to QLoRA 8GB findings (#137)"
```

**Success criterion (recorded observation):** the placeholder is replaced with a working relative link + summary, lints clean, and is committed.

---

## Task X: Teardown — restore the dev env

> Final task. Restores the default dev env after all GPU runs are complete.

**Files:**

- No source files. Env teardown only.

- [ ] **Step 1: Restore the dev env**

Run: `uv sync --extra dev`
Expected: resolves the default (dev) toolchain without error.

- [ ] **Step 2: Confirm scratch artifacts are not staged**

Run: `git status --porcelain` and confirm no `<SCRATCH>/` scripts/logs/run-dirs are staged or tracked. Leave the ephemeral scripts in place on disk (they are gitignored) or delete them — either is fine; just do not commit them.

**Success criterion (recorded observation):** dev env restored; `git status` shows only the three committed deliverables (report, conditional config on FIT, placeholder edit) and no scratch artifacts.

---

## Self-review — spec coverage map

- **Run A (§4.1)** → Task A (A(a) SDPA probe + A(b) static floor & forward-only). **Early-exit (§4.1)** → Task A Step 4 + Task A-EXIT.
- **Run B (§4.2)** → Task B (baseline; OOM-is-a-datum, not an exit).
- **Run C (§4.3)** → Task C (stacked levers, 2–3-step max peak). **Decision gate (§4.3)** → Task C Step 3.
- **Run D (§4.4)** → Task D (Run-A(a)-driven lever choice, attribution). **Skip-on-early-exit (§4.4)** → Task D gating + Task A-EXIT Step 2.
- **§4.5 hypothesis** → Task D Step 4 + report §6 (Task R).
- **Levers / fp16 caveat (§5)** → Task C config + report fp16-finiteness note.
- **Deliverable 1 report (§7.1)** → Task R. **Deliverable 2 conditional config (§7.2)** → Task C-CONFIG. **Deliverable 3 placeholder (§7.3)** → Task P.
- **Env / isolation / measurement (§3)** → Task 0 + Task X + the hard-constraints block applied to every GPU task.
- **Out of scope (§2.2 / §9)** → no tasks (no `gpu_local` test, no `gpu-test-policy.md` edit, no offload run, no grad-ckpt, no convergence/throughput).
