# Colab GPU Integration Fix v2 Design (spec/colab-gpu-integration-fix-v2)

**Status:** ready for implementation
**Parent specs:**
- [`2026-05-17-colab-gpu-integration-fix-design.md`](2026-05-17-colab-gpu-integration-fix-design.md) — the v1 spec, still authoritative for `_Sam3ImageAdapter.forward`'s call recipe (§4) and `SCOPE_TARGETS` rationale (§5). v2 layers two new fixes on top of the v1 recipe and reconciles the branch with PR #14.
- [`2026-05-17-training-loop-design.md`](2026-05-17-training-loop-design.md) §4-§6 — pins `_build_geometric_prompt` semantics and the `box_hints` kwarg on `Sam3Wrapper.forward`.

**Sibling spec:** [`2026-05-17-peft-qlora-design.md`](2026-05-17-peft-qlora-design.md)
**Branch:** `worktree-fix+colab-bpe-gzip` (PR #13 open). Branch tip is `dec482b chore(logs): record task-5 push`. Origin/main tip is `5071c00 feat(train): training loop (#14)`, which diverges from our merge-base at `d2cef37`.

---

## 1. Purpose

After the first round of Colab T4 fixes (PR #13 commits `517ff6a`, `ab8b0b9`, `4c4686f`, `09bde5b`, plus the `dec482b` log entry), the integration suite still fails 8 of 9 tests on Colab T4. Two distinct, independent root causes remain. Additionally, PR #14 has landed on `main` after our work began and reshaped the very file (`src/esam3/models/sam3.py`) we modified, so a rebase is required before either remaining root cause can be fixed cleanly.

This spec resolves three problems in a strictly sequenced order:

1. **Rebase reconciliation** of `worktree-fix+colab-bpe-gzip` onto `origin/main` (PR #14 — adapter signature change + new `_build_geometric_prompt` helper + new `bpe_path` regression).
2. **Dtype mismatch** in `_Sam3ImageAdapter.forward`: SAM 3.1's geometry encoder synthesizes a float32 zero-points tensor that hits a bf16-weight `nn.Linear`, causing `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16` on `test_load_sam31_forward_to_canonical`.
3. **peft × torchao version incompatibility** on Colab: peft 0.19+ requires `torchao>=0.16.0` when torchao is preinstalled (Colab ships `torchao 0.10.0`), and peft raises `ImportError` from `is_torchao_available()` during `apply_lora` / `apply_qlora`'s `_create_new_module` dispatch, blocking 7 of 9 tests.

After implementation, all 9 tests under `requires_compatible_gpu and requires_checkpoint` pass on Colab T4. The unit baseline (post-rebase) stays green with no regressions.

## 2. Constraints

### 2.1 Branch and commit policy

- Stay on branch `worktree-fix+colab-bpe-gzip`.
- **Rebase** onto `origin/main` before any new functional commit. The rebase is the first work item.
- After the rebase, ADD up to 3 new commits (dtype fix, torchao pin, log entry). The existing commits `517ff6a`, `ab8b0b9`, `4c4686f`, `09bde5b` are good and must survive the rebase (4c4686f and 09bde5b should rebase cleanly; ab8b0b9 must be re-applied into PR #14's new adapter signature; 517ff6a must be re-applied to undo PR #14's accidental `bpe_path` re-introduction).
- `dec482b chore(logs): record task-5 push` is a log-only commit with no functional content; DROP it during the rebase (squash into the rebase commit's log entry).
- No new runtime dependencies in `pyproject.toml`. The torchao pin lives in `notebooks/colab_gpu_tests.ipynb` only (§5).
- No emojis in source, comments, or commit messages.
- Logging: append to `logs/log.md` after each work item using `[<UTC-ISO8601>] [ROLE] action`. Do not read the log during task execution.

### 2.2 What must NOT change after the rebase

- `Sam3Wrapper.forward` signature (`forward(self, images, prompts, box_hints=None)`) — main's new signature is the contract.
- `Sam3Wrapper._validate_inputs` body — main's new validator covers the new `box_hints` case correctly and matches PR #14's test `tests/unit/test_sam3_wrapper_box_hints.py`.
- `_build_geometric_prompt` body — main's implementation is unit-tested by `tests/unit/test_geometric_prompt_builder.py` and must remain byte-identical.
- The `Prompt` / `BoxPrompts` / `box_xyxy_to_cxcywh` imports introduced by PR #14.
- All new files under `src/esam3/train/` and `tests/unit/test_train_*`, `tests/unit/test_trainer_*`, `tests/unit/test_box_hint_schedule.py`, `tests/unit/test_geometric_prompt_builder.py`, `tests/unit/test_sam3_wrapper_box_hints.py`, `tests/integration/test_train_*.py`, `tests/gpu/test_real_train_overfits.py` — leave them alone; the rebase imports them unchanged.

### 2.3 Hardware constraint

- Dev box is GTX 1080 (compute capability 6.1); all `requires_compatible_gpu` tests skip locally.
- Verification of the dtype fix and torchao fix only happens on Colab T4 via `notebooks/colab_gpu_tests.ipynb` → `bash scripts/run_gpu_tests.sh`. Local unit tests give partial confidence; the Colab run is the final gate.

## 3. Source investigation for Problem 2 (dtype mismatch)

This section documents the exact code path that produces the bf16/float32 mismatch, sourced from the installed `sam3` package at `.venv/lib/python3.13/site-packages/sam3/`. The implementer MUST verify these source citations before writing code.

### 3.1 Failing trace (Colab T4)

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
  src/esam3/models/sam3.py:51   out = self.model(images, prompts)           # Sam3Wrapper.forward → adapter
  src/esam3/models/sam3.py:134  outputs = self.model.forward_grounding(...) # adapter (post-v1)
  sam3/model/sam3_image.py:449  prompt, prompt_mask, backbone_out = self._encode_prompt(
  sam3/model/sam3_image.py:189  geo_feats, geo_masks = self.geometry_encoder(
  sam3/model/geometry_encoders.py:781  final_embeds, final_mask = self._encode_points(
  sam3/model/geometry_encoders.py:594  proj = self.points_direct_project(points)   # ← raises here
```

### 3.2 Why `_encode_points` is reached even though we pass no points

`_Sam3ImageAdapter.forward` (v1) calls `self.model._get_dummy_prompt(num_prompts=1)`, which returns:

```python
# sam3/model/sam3_image.py:547-553
Prompt(
    box_embeddings=torch.zeros(0, num_prompts, 4, device=device),  # default float32
    box_mask=torch.zeros(num_prompts, 0, device=device, dtype=torch.bool),
)
```

The `Prompt` constructor (`sam3/model/geometry_encoders.py:102-238`) detects `point_embeddings is None` AND `box_embeddings is not None`, so it follows the **non-null** branch starting at line 132. There `_init_seq_len_and_device` derives `box_seq_len=0`, `point_seq_len=0`, `bs=num_prompts=1`, `device=box.device`. Then `_init_point` (line 292-306) synthesizes a default zero-points tensor **in float32**:

```python
# sam3/model/geometry_encoders.py:298-299
if point_embeddings is None:
    point_embeddings = torch.zeros(point_seq_len, bs, 2, device=device)  # default float32
```

The geometry-encoder forward (line 717) then unconditionally feeds `points` (now float32 zeros of shape `(0, 1, 2)`) into `_encode_points` (line 781). Inside `_encode_points`, line 593-594:

```python
if self.points_direct_project is not None:
    proj = self.points_direct_project(points)  # nn.Linear with bf16 weight
```

`points_direct_project` is a `nn.Linear` whose weight was cast to bf16 by our `load_sam31` (`raw_model.to(dtype=torch.bfloat16)`). PyTorch's `F.linear` checks input/weight dtypes **before** iterating rows; even a 0-length input triggers the dtype error.

### 3.3 Why `Sam3Processor` does not hit this bug

`sam3.model.sam3_image_processor.Sam3Processor` uses exactly the same `_get_dummy_prompt()` recipe (line 123). It does NOT cast to bf16 anywhere; its `set_image` step (line 55) actively casts images to **float32** via `v2.ToDtype(torch.float32, scale=True)`. Meta's reference `_setup_device_and_mode` in `sam3/model_builder.py:564-570` also **does not** cast model parameters to bf16 — it only does `.cuda()`.

In other words: **Meta's reference Processor expects the entire model to stay in float32 by default.** Our `load_sam31` (`src/esam3/models/sam3.py`) casts the whole model to bf16 when `cfg.dtype == "bfloat16"`, which our integration test enables (`ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")`, `tests/integration/test_load_sam31_real.py:21`). That cast makes `points_direct_project.weight` bf16. The Processor recipe was never exercised in the bf16 configuration upstream — so no Meta-side workaround exists.

### 3.4 Why `torch.autocast(bfloat16)` resolves the bug

`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` intercepts every dispatch of `nn.Linear` (op `aten::linear`) on the autocast-included list and downcasts the input tensor to match the autocast dtype before the matmul. With the model weights ALREADY in bf16 and autocast enabled with `dtype=torch.bfloat16`, the float32 zero-points input becomes bf16 at the dispatch boundary; the matmul sees bf16-vs-bf16 and proceeds.

Compatibility notes verified by reading PyTorch 2.4+ autocast docs and source:
- Autocast works on CUDA op dispatch; `aten::linear` is on the autocast-include list (cast to autocast dtype).
- Autocast composes with `torch.no_grad()` (the failing test uses `torch.no_grad()` on line 32). Autocast is orthogonal to gradient tracking.
- Autocast composes with already-bf16 weights: the cast is a no-op when the input already matches autocast dtype.
- Autocast WITH `dtype=torch.bfloat16` (vs the default `float16`) is the correct choice here because all model weights are bf16; using `float16` autocast would downcast to fp16 internally and conflict with bf16 weights, producing the SAME class of dtype error (verified by reasoning about `_cast_to` semantics).

### 3.5 Why options (B), (C), (D) are rejected

| Option | Why rejected |
| --- | --- |
| (B) Cast `images` to `next(self.model.parameters()).dtype` before forward | The offending tensor is NOT `images` — it is the float32 zeros synthesized by `Prompt._init_point` deep inside the geometry encoder. Casting `images` does not change what `_init_point` constructs. Verified by re-reading `geometry_encoders.py:298-299`. |
| (C) Build the dummy Prompt in model dtype (bf16 box_embeddings) | The `box_embeddings` aren't the failure point either — they are bf16-able zeros that never reach a fp32 op. Even if we set `box_embeddings.dtype=bfloat16`, `_init_point` still synthesizes a float32 point tensor (line 299 hardcodes float32 default). To fully fix via (C) we would need to also pass `point_embeddings=torch.zeros(0, 1, 2, dtype=bfloat16, device=...)`, which is brittle: it pre-empts `Prompt._init_point`'s default branch and is sensitive to the device/shape Meta might tweak. (A) covers this and any analogous case downstream. |
| (D) Pass a fully-populated geometric_prompt skipping `_encode_points` | `_encode_points` is called unconditionally at line 781 regardless of `point_seq_len`; the only way to "skip" it is to monkey-patch `self.points_direct_project = None`, which is invasive and out-of-scope. |

### 3.6 Chosen fix (Option A)

Wrap the adapter's `forward_grounding` call (and the preceding `backbone.forward_image` / `forward_text` calls that also hit bf16 weights) in:

```python
with torch.autocast(
    device_type=device.type,
    dtype=torch.bfloat16,
    enabled=(device.type == "cuda"),
):
    backbone_out = self.model.backbone.forward_image(images)
    text_outputs = self.model.backbone.forward_text([class_names[0]], device=device)
    backbone_out.update(text_outputs)
    ...
    outputs = self.model.forward_grounding(...)
```

The `enabled` predicate guards CPU paths (`torch.autocast(device_type="cpu")` exists but autocast on CPU is a no-op for `aten::linear` in standard PyTorch builds; making the context conditional avoids confusion and keeps the CPU-stub fixtures unaffected).

Why this is safe under PEFT LoRA / QLoRA: PEFT wraps target Linears in `lora.Linear(base_layer=original_linear, ...)`. Both `base_layer` and `lora_A`/`lora_B` are `nn.Linear`s, so they're on the autocast op-list and get the same treatment. bitsandbytes `Linear4bit` also dispatches through `aten::linear`-equivalent paths that autocast handles (verified empirically in upstream bitsandbytes integration; this is the standard QLoRA inference pattern).

## 4. Source investigation for Problem 3 (peft × torchao)

Sourced from `.venv/lib/python3.13/site-packages/peft/`.

### 4.1 Failure surface

`peft.tuners.lora.torchao:142` calls `peft.import_utils.is_torchao_available()` from inside the LoRA dispatcher (`_create_new_module`). Inspected at `peft/import_utils.py:126-147`:

```python
@lru_cache
def is_torchao_available():
    if importlib.util.find_spec("torchao") is None:
        return False                                # safe path
    TORCHAO_MINIMUM_VERSION = packaging.version.parse("0.16.0")
    try:
        torchao_version = packaging.version.parse(importlib_metadata.version("torchao"))
    except importlib_metadata.PackageNotFoundError:
        return False
    if torchao_version < TORCHAO_MINIMUM_VERSION:
        raise ImportError(
            f"Found an incompatible version of torchao. Found version {torchao_version}, "
            f"but only versions above {TORCHAO_MINIMUM_VERSION} are supported"
        )
    return True
```

Critical: the function is `@lru_cache`-d, so once it raises `ImportError` it raises on **every** subsequent call. The dispatcher hits it once per LoRA-eligible Linear during `apply_lora`/`apply_qlora`; on Colab T4 (peft 0.19.x, torchao 0.10.0 preinstalled), the FIRST call raises.

### 4.2 Local vs Colab divergence

- Local dev box: `peft 0.19.1` installed; **no torchao** installed. `find_spec("torchao") is None` → returns `False` safely. `apply_lora` works.
- Colab T4: `peft 0.19.x` installed via our `pip install`; **`torchao==0.10.0` preinstalled** in the base Colab Python image (it's a transitive dependency of Colab's preinstalled `torch` toolkit). `find_spec` finds it; version is `< 0.16.0`; ImportError raised.

This explains why local unit tests pass (peft + no torchao) but 7 of 9 Colab tests fail (peft + old torchao).

### 4.3 Fix options weighed

| Option | Mechanism | Pros | Cons | Verdict |
| --- | --- | --- | --- | --- |
| (i) Pin `torchao>=0.16.0` in `notebooks/colab_gpu_tests.ipynb`'s install cell | Add `"torchao>=0.16.0"` to the existing `%pip install -e ".[qlora,dev,tensorboard]" "numpy==1.26.4" ...` line | Surgical; no change to `pyproject.toml`; resolves in the SAME pip pass as existing pins; matches the documented pattern in the install cell's comment ("Why all these pins on ONE pip-install line"); leaves local dev untouched (no torchao install, no behavior change). | Requires the user to rerun the notebook from scratch after the pin lands (a Colab session restart is already part of the gate). | **Chosen.** |
| (ii) Pin `torchao>=0.16.0` in `pyproject.toml` runtime deps | Add `"torchao>=0.16.0"` to `dependencies` | Single source of truth; works for any future GPU env. | Forces a heavy (~hundreds of MB) torchao install on every developer's local machine even though we don't use torchao. Inflates CI install time. Risks pulling torchao-version-incompatible torch upgrades. | Rejected. |
| (iii) Pin `peft<0.19` to a version pre-dating the `torchao>=0.16.0` floor | Add `"peft>=0.13,<0.19"` to runtime deps | Sidesteps the check entirely. | The pre-0.19 peft API is what we already developed against; downgrading is fine locally but unverified for the QLoRA path; could re-introduce other bugs fixed between 0.13 and 0.19. The exact peft version that introduced the floor is not documented inline; pinning by version is brittle. | Rejected as primary fix; kept as a deferred fallback if (i) ever conflicts with a future Colab base image. |
| (iv) Monkey-patch `peft.import_utils.is_torchao_available` in `src/esam3/peft_adapters/lora.py` import | Override the function to always return `False` | Zero install-time cost. | Touches third-party internals; the `@lru_cache` decorator complicates the patch ordering; v1 spec §8 explicitly deferred this option as out-of-scope and that judgment stands. | Rejected. |

### 4.4 Chosen fix (Option i)

Edit `notebooks/colab_gpu_tests.ipynb` Cell 4 (the `%pip install` cell). Add `"torchao>=0.16.0"` to the install-line pin list. Update the docstring comment block above the line to note the new pin.

Concretely, the install line becomes:

```python
%pip install -e ".[qlora,dev,tensorboard]" \
    "numpy==1.26.4" "scipy==1.13.1" "transformers==5.0.0" \
    "huggingface_hub>=1.15" \
    "torchao>=0.16.0"
```

And the comment block above gains one paragraph explaining the pin (peft 0.19+ dispatches through `peft.tuners.lora.torchao`; Colab preinstalls torchao 0.10.0; peft's `is_torchao_available()` raises ImportError if torchao is installed but `<0.16.0`; we don't USE torchao but we have to upgrade the system-installed one to clear the gate).

## 5. File layout (post-rebase)

| File | Change | Owning task |
| --- | --- | --- |
| (rebase) `src/esam3/models/sam3.py` | Resolve conflict: keep main's imports (`box_xyxy_to_cxcywh`, `Prompt`, `BoxPrompts`), main's `_build_geometric_prompt`, main's `Sam3Wrapper` signature (`box_hints` kwarg), main's `_validate_inputs`. Re-apply: drop `_resolve_bpe_path` + `bpe_path=` kwarg (gzip-fix), keep `FindStage` import, plumb `image_size` through adapter constructor, re-implement adapter `forward` body using main's recipe (build `gp` via `_build_geometric_prompt`, fallback to zero-length-seq `Prompt`, call `forward_grounding`). | 1 |
| `src/esam3/models/sam3.py` | After rebase, add `torch.autocast` wrapper around adapter's bf16-touching forward calls (Problem 2). | 2 |
| `notebooks/colab_gpu_tests.ipynb` | Add `torchao>=0.16.0` to the install cell's pin list; expand the comment block. | 3 |
| `logs/log.md` | Append rebase + fix entries (one per task). | 1, 2, 3 |

No changes to: `pyproject.toml`, `scripts/run_gpu_tests.sh`, any file under `src/esam3/train/`, `src/esam3/peft_adapters/`, any `tests/` file, `tests/fixtures/tiny_sam3_lora_stub.py`. (The Task 3 / Task 4 fixture-and-substring renames from v1 are already in our branch and survive the rebase byte-identically.)

## 6. Design decisions

### 6.1 Rebase strategy

`git rebase origin/main` on the worktree. Expected behavior:

| Commit being replayed | Outcome |
| --- | --- |
| `517ff6a fix(models): drop bpe_path override` | **Conflict** in `src/esam3/models/sam3.py`. Origin/main re-introduced `_resolve_bpe_path` and `bpe_path=str(bpe_path)`. Resolution: take main's version, then DELETE `_resolve_bpe_path` and the `bpe_path=str(bpe_path)` kwarg again (re-apply the gzip fix). |
| `ab8b0b9 fix(models): implement _Sam3ImageAdapter.forward via forward_grounding` | **Conflict** in `src/esam3/models/sam3.py`. Origin/main reshaped the adapter to accept `box_hints` and left the body as `NotImplementedError`. Resolution: re-apply our forward body, ADAPTED to the new signature (see §6.2). |
| `dd76a20 docs: spec + plan for Colab GPU integration fix` | Clean apply (docs only). |
| `4c4686f fix(peft): pin SCOPE_TARGETS to real SAM 3.1 module names` | Clean apply (`lora.py` + `tests/fixtures/tiny_sam3_lora_stub.py` + `tests/unit/test_peft_lora.py`; origin/main did not touch these). |
| `09bde5b test(integration): realign LoRA name substrings` | Clean apply (`tests/integration/test_peft_*_real.py`; origin/main did not touch these). |
| `dec482b chore(logs): record task-5 push` | **DROP** during rebase (`git rebase -i` is forbidden; use the non-interactive rebase, then squash this commit's log content into the new Task 2 log entry, or simply skip — see §6.5). |

Conflict resolution is concentrated in ONE file (`src/esam3/models/sam3.py`). Other files rebase clean.

### 6.2 Adapter forward body, post-rebase

After resolving the rebase conflict, `_Sam3ImageAdapter` must:

1. Accept `image_size` via its constructor (default `1008`, matching `Sam3Wrapper`'s default). This is a CONSTRUCTOR-SIGNATURE change vs v1's `__init__(self, model: nn.Module)`. It is justified because `_build_geometric_prompt` (helper from main) needs `image_size` and there is no other clean source.

   ```python
   def __init__(self, model: nn.Module, image_size: int = 1008) -> None:
       super().__init__()
       self.model = model
       self.image_size = image_size
   ```

   `load_sam31` is updated to pass it through: `adapter = _Sam3ImageAdapter(raw_model, image_size=1008)`.

2. Accept `box_hints: list[Tensor | None] | None = None` in its `forward` signature (matching main's signature and the wrapper's call site `self.model(images, prompts, box_hints=box_hints)`).

3. Body: keep the v1 recipe (TextPrompts validation, `forward_image`, `forward_text`, FindStage with `img_ids=arange(B)` + `text_ids=zeros(B)`, `forward_grounding`) BUT replace `geometric_prompt = self.model._get_dummy_prompt(num_prompts=1)` with the box-hints-aware builder:

   ```python
   b = images.shape[0]
   gp = _build_geometric_prompt(
       box_hints if box_hints is not None else [None] * b,
       self.image_size,
       images.device,
   )
   if gp is None:
       # All hints are None — substitute Meta's zero-length-seq dummy.
       # Equivalent to self.model._get_dummy_prompt(num_prompts=b), with
       # an explicit construction to keep the shape contract local to this file.
       gp = Prompt(
           box_embeddings=torch.zeros(0, b, 4, device=images.device),
           box_mask=torch.zeros(b, 0, device=images.device, dtype=torch.bool),
       )
   ```

   Note: the dummy uses `num_prompts=b` (not `1`), because the wrapper's contract has been broadened by PR #14: with `box_hints` passed, every image gets either a real hint or a padded slot in a shared `(N_max, B, 4)` tensor; the dummy must mirror that batch-shape.

4. Wrap the forward body (from `backbone.forward_image` through `forward_grounding`) in `torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")`. Problem 2 fix.

5. Return the raw output dict unchanged (same as v1).

Pseudo-code of the final body (illustrative — not literal source):

```python
def forward(self, images, prompts, box_hints=None):
    if not all(isinstance(p, TextPrompts) for p in prompts):
        raise ValueError("_Sam3ImageAdapter only supports TextPrompts in v0")
    class_names = [p.classes[0] for p in prompts]
    if len(set(class_names)) > 1:
        raise ValueError(
            "All prompts in a batch must share the same class name "
            "(SAM 3.1 forward_grounding runs one text prompt per call); "
            f"got {class_names}"
        )
    device = images.device
    b = images.shape[0]
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=(device.type == "cuda"),
    ):
        backbone_out = self.model.backbone.forward_image(images)
        text_outputs = self.model.backbone.forward_text([class_names[0]], device=device)
        backbone_out.update(text_outputs)
        find_input = FindStage(
            img_ids=torch.arange(b, device=device, dtype=torch.long),
            text_ids=torch.zeros(b, device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        gp = _build_geometric_prompt(
            box_hints if box_hints is not None else [None] * b,
            self.image_size,
            device,
        )
        if gp is None:
            gp = Prompt(
                box_embeddings=torch.zeros(0, b, 4, device=device),
                box_mask=torch.zeros(b, 0, device=device, dtype=torch.bool),
            )
        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=gp,
        )
    return outputs
```

### 6.3 Why the autocast wraps the WHOLE body (not just `forward_grounding`)

The bf16 model weights are reached by:
- `backbone.forward_image(images)` — vision trunk, bf16 weights, fp32 input image fine because main's autocast pattern works.
- `backbone.forward_text([class_names[0]], device=device)` — text encoder, bf16 weights, internal fp32 zero tensors possible.
- `forward_grounding(...)` — the geometry encoder fp32-zeros issue (Problem 2).

The test passes `images` already in bf16 (line 31 of `tests/integration/test_load_sam31_real.py`), so today only `forward_grounding` strictly needs autocast. Wrapping the whole body is defensive: future call sites (e.g. the training loop in `src/esam3/train/loop.py`) may pass fp32 images, and the wider scope handles them at no perf cost. The autocast context is a thin Python wrapper; the per-call overhead is negligible compared to the model forward.

### 6.4 Why we do NOT touch `pyproject.toml` for torchao

The `qlora` extra is for runtime QLoRA support (bitsandbytes only). torchao is NOT a runtime dependency of any code in `src/esam3/`. The peft `is_torchao_available()` check is a side-effect of having peft installed in an environment where some OTHER package brought in torchao. The right place to handle "the surrounding environment ships an incompatible version of an unused transitive" is the install recipe for that environment — i.e. the Colab notebook, not the project's own pin set.

If a future non-Colab GPU environment exhibits the same Colab-base-image-style torchao preinstall, the same single-line install pin can be added to that env's setup. We do not need to centralize until a second site demands it.

### 6.5 What to do with `dec482b chore(logs): record task-5 push`

This commit only modifies `logs/log.md`. After the rebase, the log line it added (`task-5: pushed; awaiting Colab T4 verification`) is no longer accurate — Task 5 of the v1 plan has already happened and Colab has been run and partially failed. The clean choice is:

- Run `git rebase origin/main` non-interactively. By default it will replay this commit and apply cleanly. After the rebase, the Task 2 work item below will append fresh entries for the v2 work; we will NOT explicitly drop the old log line (it remains a historical artifact, which is consistent with append-only).

If the rebase produces a conflict on `logs/log.md` (unlikely — origin/main did not modify it), resolve by taking our version verbatim.

We do NOT use `git rebase -i` (forbidden by repo policy). We do NOT amend the commit.

### 6.6 Acceptance for the autocast scope

The autocast call MUST:
- Use `device_type=device.type` (string `"cuda"` on Colab, `"cpu"` locally).
- Use `dtype=torch.bfloat16` explicitly (not the default float16).
- Use `enabled=(device.type == "cuda")` so CPU-only fixtures (unit tests on dev box that import the wrapper) do not enter autocast at all.
- Wrap exclusively the body that touches model weights (so the prompt validation above the body runs in normal fp32 control flow; this matches the contract of the test suite).

## 7. Test plan

### 7.1 Unit tests (local, CPU-only)

The post-rebase unit baseline is **larger than the pre-rebase 201** because origin/main added training-loop unit tests, geometric-prompt-builder tests, box-hint-schedule tests, etc. The exact post-rebase count must be measured AT REBASE TIME and pinned in `logs/log.md` as the new baseline.

After the rebase commit lands locally, Task 1's verification step must record the new baseline. After Task 2 (dtype fix), the baseline must be unchanged — no new unit tests are added by the dtype fix (autocast is GPU-only behavior, untested locally). After Task 3 (notebook pin), the baseline must again be unchanged.

The CPU-only fixture `tests/fixtures/tiny_sam3_stub.py` does NOT use a bf16 model and does NOT exercise the dtype-mismatch code path; it must continue to pass after the autocast wrapping. The autocast `enabled=False` predicate on CPU guarantees this.

### 7.2 Integration tests (Colab T4, gated)

All 9 tests under `tests/integration/test_*.py + tests/gpu/test_*.py` filtered by `requires_compatible_gpu and requires_checkpoint` MUST pass. Specifically:

| Test | What this fix unblocks |
| --- | --- |
| `test_load_sam31_returns_wrapper` | Already passing post-gzip-fix; no regression expected. |
| `test_load_sam31_forward_to_canonical` | Unblocked by Problem 2 fix (autocast). |
| `test_apply_lora_on_real_sam31_under_trainable_budget` | Unblocked by Problem 3 fix (torchao pin). |
| `test_save_load_roundtrip_on_real_sam31` | Unblocked by Problem 3 fix. |
| `test_merge_lora_on_real_sam31` | Unblocked by Problem 3 fix. |
| `test_apply_qlora_swaps_every_linear_and_attaches_lora` | Unblocked by Problem 3 fix. |
| `test_save_load_qlora_roundtrip_on_real_sam31` (or analog) | Unblocked by Problem 3 fix. |
| `test_merge_qlora_on_real_sam31` (or analog) | Unblocked by Problem 3 fix. (Note: if `merge_qlora` is not implemented for 4-bit quantized weights — Linear4bit doesn't support `.merge_and_unload()` directly — the existing test may be a stub that asserts the unsupported path; verify against the actual test body before claiming success.) |
| `tests/gpu/test_real_train_overfits.py::*` (PR #14) | Unblocked by all three fixes combined (the training loop calls `apply_lora` AND runs forward through the adapter). |

The Colab notebook runs `bash scripts/run_gpu_tests.sh`, which currently filters by markers. If PR #14 added new GPU tests (`tests/gpu/test_real_train_overfits.py`) that the runner script doesn't include yet, that is a separate concern — out of scope for this spec. The 9-passing target is measured against whatever the runner script currently invokes.

### 7.3 What is NOT verified by this spec

- Numerical equivalence between autocast-wrapped and non-autocast forward passes. The forward path was not producing meaningful outputs before this fix (it raised), so there's no baseline to compare against. The acceptance criterion is "test passes", not "outputs are bit-identical to fp32".
- The dtype fix is verified ONLY on Colab T4. The dev box (compute capability 6.1) skips `requires_compatible_gpu`.
- Re-running the Colab notebook with a clean session is the user's responsibility; the implementer pushes commits and asks the user to verify.

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Rebase produces conflicts in files I did not anticipate (e.g. `logs/log.md`, `pyproject.toml`). | Low — `git diff --name-only` shows main only touched `src/esam3/models/sam3.py` and new files. | If a conflict appears outside `sam3.py`, STOP and inspect; do not resolve blindly. |
| `_build_geometric_prompt` semantics change in a future main commit before we land. | Low — main's tip is pinned at `5071c00`. | Rebase against the pinned SHA, not a moving target. |
| Autocast on bf16 weights breaks `test_load_sam31_returns_wrapper` (the only currently-passing integration test). | Very low — that test does not call `wrapper.forward`, only `load_sam31`. The autocast context is per-forward-call. | Covered by reading the test body; risk is structural-impossible. |
| Autocast on bf16 weights changes the OUTPUT dtype of `forward_grounding` and breaks `meta_to_canonical`'s downstream consumption. | Low — `meta_to_canonical` only reads keys and asserts shapes, not dtypes. | Verified by reading `src/esam3/models/matching.py` (no dtype checks). |
| The `torchao>=0.16.0` install in Colab pulls a wheel incompatible with Colab's `torch` version. | Medium — torchao occasionally has tight torch pins. | If the upgrade fails, the install cell errors loudly; fallback is Option (iii) `peft<0.19` pin, applied to the notebook's `%pip install` line. Document this fallback in §4.3 (already noted). |
| Dev-box `pytest` still passes locally but Colab fails differently (new error mode). | Medium — only Colab can verify. | Plan must accept "push → user runs Colab → iterate" loop. Final task (5) is Colab verification. |
| Re-applying Task 2 into the new adapter signature loses the prompt-validation guards (TextPrompts check, single-class check). | Medium — easy to copy the wrong lines during conflict resolution. | DoD on Task 1 must explicitly grep for both validation strings post-rebase. |
| The 9-test count drifts (e.g., PR #14 adds GPU tests that `scripts/run_gpu_tests.sh` doesn't currently run). | Low for the suite as-is. | Acceptance is "the suite passes", measured against `scripts/run_gpu_tests.sh` output, NOT against a hardcoded count. |
| `_build_geometric_prompt` with `image_size=1008` is wrong for a test that passes images at a different resolution. | Low — current tests use 1008. | If a future test introduces a different resolution, the wrapper's `image_size` constructor arg must be plumbed through; that's already part of this spec. |

## 9. Out of scope

| Item | Deferred to |
| --- | --- |
| Pinning `peft<0.19` or any other peft version cap. | Spec §4.3 — kept as a fallback only. |
| Monkey-patching `peft.import_utils.is_torchao_available`. | v1 §8 deferral; still deferred. |
| Centralizing the torchao pin in `pyproject.toml`. | When a second non-Colab env demands it; not now. |
| Numerical-equivalence testing of the bf16 forward path against fp32. | Not in scope; the test is "passes shape contract". |
| Box-prompt path through `_Sam3ImageAdapter`. | v1 §8; still deferred. |
| Multi-class-per-batch forward. | v1 §8; still deferred. |
| Squashing `dec482b` out of branch history. | Accept the historical artifact; do not amend. |

## 10. Acceptance criteria

A correct implementation of this spec satisfies:

1. `worktree-fix+colab-bpe-gzip` is rebased onto `origin/main` (`5071c00`). `git log --oneline origin/main..HEAD` shows the original 5 commits (4 functional + the historical `dec482b` log entry) replayed cleanly, plus the new v2 commits from this spec's plan.
2. `src/esam3/models/sam3.py` contains:
   - main's `_build_geometric_prompt` helper, byte-identical.
   - main's `Sam3Wrapper.forward` signature with `box_hints` kwarg, byte-identical.
   - main's `_validate_inputs` static method, byte-identical.
   - Our gzip-fix: NO `_resolve_bpe_path` function, NO `bpe_path=` kwarg in the `build_sam3_image_model(...)` call.
   - Our adapter: `_Sam3ImageAdapter.__init__(self, model, image_size=1008)`, `forward(self, images, prompts, box_hints=None)`, body wrapped in `torch.autocast(...)` and using `_build_geometric_prompt` with fallback to a manual `Prompt(box_embeddings=zeros(0, B, 4), box_mask=zeros(B, 0))`.
   - `from sam3.model.data_misc import FindStage` import preserved.
3. `notebooks/colab_gpu_tests.ipynb`'s install cell includes `"torchao>=0.16.0"` in the same `%pip install -e ...` line as the existing pins, and the comment block above documents WHY.
4. `pyproject.toml` is unchanged.
5. `uv run pytest tests/unit -q` passes at the new post-rebase baseline with no regressions. The baseline count is pinned in `logs/log.md` by Task 1.
6. `ruff check` and `ruff format --check` pass on `src/esam3/models/sam3.py` (the only edited source file).
7. On Colab T4: `bash scripts/run_gpu_tests.sh` reports all 9 tests passing under `requires_compatible_gpu and requires_checkpoint`.
8. `logs/log.md` contains an entry per task (rebase, dtype fix, torchao pin, Colab push, Colab verification).
9. No new dependencies in `pyproject.toml`. No emojis anywhere.

## 11. References for the implementer

- v1 spec: `docs/superpowers/specs/2026-05-17-colab-gpu-integration-fix-design.md` — recipe for `forward_grounding` call (§4), `SCOPE_TARGETS` rationale (§5).
- Training-loop spec: `docs/superpowers/specs/2026-05-17-training-loop-design.md` — pins box-hint plumbing, `_build_geometric_prompt` contract.
- Training-loop notes: `docs/superpowers/plans/2026-05-17-training-loop-notes.md` — Meta's `geometric_prompt` layout.
- sam3 source (read but do not modify):
  - `.venv/lib/python3.13/site-packages/sam3/model/sam3_image.py:440-553` — `forward_grounding`, `_get_dummy_prompt`, `_encode_prompt`.
  - `.venv/lib/python3.13/site-packages/sam3/model/sam3_image_processor.py:31-189` — reference `set_image` / `set_text_prompt` / `_forward_grounding` recipe.
  - `.venv/lib/python3.13/site-packages/sam3/model/geometry_encoders.py:83-630` — `Prompt` constructor, `_init_point` (line 292-306, source of the float32 zeros), `_encode_points` (line 589-630, where the mismatch raises).
  - `.venv/lib/python3.13/site-packages/sam3/model/data_misc.py` — `FindStage` dataclass.
  - `.venv/lib/python3.13/site-packages/sam3/model_builder.py:564-654` — `build_sam3_image_model` and `_setup_device_and_mode` (confirms upstream model is NOT bf16 by default).
- peft source:
  - `.venv/lib/python3.13/site-packages/peft/import_utils.py:126-147` — `is_torchao_available` (the version-gate function).
  - `.venv/lib/python3.13/site-packages/peft/tuners/lora/torchao.py` — the dispatcher that calls `is_torchao_available()`.
- PyTorch autocast docs: https://pytorch.org/docs/stable/amp.html#torch.autocast — `aten::linear` is on the autocast cast-list.
- PR #14 commit on `origin/main`: `5071c00 feat(train): training loop (#14)`.
