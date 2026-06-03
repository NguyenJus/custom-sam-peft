# spec/onnx-export — `csp export --to onnx` Bundle Export & `csp predict --use-onnx` Design

**Status:** Draft (2026-06-03)
**Tracking:** [#77](https://github.com/NguyenJus/custom-sam-peft/issues/77) — *ONNX export path*
**Scope:** Extend `csp export` with an ONNX export path that merges the trained adapter, traces SAM 3.1 into a two-file SAM-family bundle (`image_encoder.onnx` + `decoder.onnx`) with load-bearing sidecars, optionally fp16-casts, optionally verifies torch-vs-ORT parity (`--check`), and reserves (but does not implement) int8 quantization. Adds a predict-side `csp predict --use-onnx <bundle>` flag that loads the bundle through ONNX Runtime instead of PyTorch, composing with #74 and doubling as the integration smoke test. The default `csp export` (PyTorch) path is preserved **bit-for-bit**.

**Builds on:**
[`2026-05-22-cli-predict-design.md`](2026-05-22-cli-predict-design.md) ([#74](https://github.com/NguyenJus/custom-sam-peft/issues/74) — `csp predict` is the predict-side host for `--use-onnx`; reuses its forward loop, `queries_to_coco_results`, transforms, and writers);
the preprocessing-parity guard ([#69](https://github.com/NguyenJus/custom-sam-peft/issues/69) — `tests/predict/test_preprocessing_parity.py`; `preprocessor.json` must capture exactly what the eval/train transform pipeline resolves);
[`2026-05-17-peft-lora-design.md`](2026-05-17-peft-lora-design.md) / [`2026-05-17-peft-qlora-design.md`](2026-05-17-peft-qlora-design.md) (adapter restore + merge semantics; QLoRA dequant on merge);
[`2026-05-23-multiplex-forward-design.md`](2026-05-23-multiplex-forward-design.md) (the B×K multiplex row ordering the decoder bundle must reproduce);
the v1.0 release gate tracked from [#70](https://github.com/NguyenJus/custom-sam-peft/issues/70).
**Supersedes the dropped guide:** the beginner integration guide ([#68](https://github.com/NguyenJus/custom-sam-peft/issues/68) prose, `docs/integration/onnx-quickstart.md`, `examples/onnx_quickstart/run.py`, the bundle `README.md`) and the deployment-recipe expansion ([#34](https://github.com/NguyenJus/custom-sam-peft/issues/34)) are **out of date / dropped (OBE)** — see §1.3. This spec covers the runtime export+predict path only.

---

## 1. Goals & v1 Scope

### 1.1 Motivation

`csp train` produces an adapter and `csp predict` (#74) runs PyTorch inference. Deployment targets without a PyTorch runtime (browser, mobile-adjacent runtimes, C++/ORT services) need a portable graph. ONNX is the lowest-common-denominator export for the SAM family, and the established convention is a **two-file split** (image encoder + prompt/mask decoder). Issue #77 adds that export path plus a predict-side loader to prove the bundle round-trips.

### 1.2 In scope

| Deliverable | Where |
| --- | --- |
| New `csp export` flags (`--to`, `--opset`, `--fp16`, `--include`, `--dynamic-axes`, `--check`, `--quantize`) | `src/custom_sam_peft/cli/export_cmd.py` (CHANGED) |
| Dispatch seam: `to == "pytorch"` → unchanged `run_export`; `to == "onnx"` → `run_export_onnx` | `src/custom_sam_peft/cli/export_cmd.py` (CHANGED, after line 42) |
| ONNX export orchestrator + tracers + sidecar writers + `--check` | `src/custom_sam_peft/export/onnx.py` (NEW) |
| `--use-onnx <bundle>` predict flag | `src/custom_sam_peft/cli/predict_cmd.py` (CHANGED) |
| ONNX session wrapper (torch-bridge) + torch-free ORT core | `src/custom_sam_peft/predict/onnx_session.py` (NEW) |
| Bundle sidecar loaders | `src/custom_sam_peft/predict/onnx_bundle.py` (NEW) |
| Shared multiplex index helper (torch-free) | `src/custom_sam_peft/models/_multiplex.py` (NEW) |
| `PredictOptions.use_onnx` field + Step-4/5 dispatch branch | `src/custom_sam_peft/predict/runner.py` (CHANGED) |
| Extracted `validate_forward_inputs` free fn | `src/custom_sam_peft/models/sam3.py` (CHANGED) |
| `onnxruntime` dependency under an `onnx` extra (lazy-imported) | `pyproject.toml` (CHANGED) |
| CPU unit + smoke tests (round-trip, `--check`, QLoRA dequant, subprocess ORT-only load) | `tests/export/`, `tests/predict/` (NEW) |

### 1.3 Out of scope / explicitly dropped (OBE)

| Item | Reason / follow-up |
| --- | --- |
| Beginner integration guide prose (#68) | **Dropped (OBE).** Do not author. |
| `docs/integration/onnx-quickstart.md` | **Dropped (OBE).** Do not create. |
| `examples/onnx_quickstart/run.py` | **Dropped (OBE).** Do not create. |
| Bundle `README.md` walkthrough | **Dropped (OBE).** The bundle directory exists but ships **no** `README.md`. The three JSON/txt sidecars (§5) are the bundle's self-description. |
| int8 quantization (`--quantize int8-dynamic`) | **Reserved placeholder only** (§9). Flag parses; any value other than `none` raises `NotImplementedError`. |
| TensorRT / CoreML / TFLite export | Out of scope. Distinct toolchains. |
| Browser-optimized / split encoder graph | Out of scope; v1 ships the plain two-file split. |
| Benchmarking / latency harness (#34 expansion) | Out of scope (OBE for this issue). |
| dynamo-based export (`torch.export`) | Decision locked: **tracing** (`torch.onnx.export`) for v1; dynamo revisit is a follow-up. |

**Kept despite the guide drop (load-bearing):** `preprocessor.json`, `model_card.json`, `prompts.txt`. These are NOT part of the dropped guide — `csp predict --use-onnx` cannot run without them.

### 1.4 Bit-for-bit PyTorch guarantee

`--to` defaults to `pytorch`, which routes to the **unchanged** `run_export()` (`runs/bundle.py:48-85`). `save_adapter`/`save_merged` (`train/checkpoint.py:107-148`) are untouched. The ONNX path is reached only via `--to onnx`. No existing test or output of the PyTorch path changes.

---

## 2. Architectural Approach

Locked decisions from brainstorming:

> - **Tracing, not dynamo.** `torch.onnx.export()` for v1. Opset floor **17**; bump cautiously and only with a recorded reason.
> - **Merge is mandatory and unconditional.** ONNX cannot represent LoRA/QLoRA deltas, so the ONNX path always merges (independent of the PyTorch-path `--merge` flag, which is irrelevant to it). QLoRA dequantizes to its `compute_dtype` during merge.
> - **Two-file SAM-family split.** `image_encoder.onnx` (vision, prompt-independent) + `decoder.onnx` (text + grounding, prompt-dependent). Spatial dims pinned to `SAM3_IMAGE_SIZE` (1008); batch dim dynamic when `--dynamic-axes` is on.
> - **Sidecars are always written** regardless of `--include`; they are required for predict-side load.
> - **`--check` is opt-in, CPU-EP, staged.** Parity runs the merged torch model vs the composed ORT bundle on synthetic input; export is staged to a temp dir and promoted only on success so a failed `--check` leaves no loadable bundle.
> - **Inference core stays torch-free.** The ORT core (`_OrtCore`) imports only numpy + onnxruntime; a separate-process subprocess test proves no accidental torch import. The surrounding predict process may still import torch for shared pre/post-processing.

---

## 3. Module Layout

```text
src/custom_sam_peft/
├── cli/
│   ├── export_cmd.py          # CHANGED — new flags + dispatch on --to (bit-for-bit pytorch path)
│   └── predict_cmd.py         # CHANGED — --use-onnx option + mutual-exclusion validation
├── export/
│   └── onnx.py                # NEW — run_export_onnx + _merge_and_cast + tracers + sidecars + --check
├── predict/
│   ├── runner.py              # CHANGED — PredictOptions.use_onnx; Step-4/5 dispatch branch
│   ├── onnx_session.py        # NEW — OnnxSam3Session (torch bridge) + _OrtCore (torch-free)
│   └── onnx_bundle.py         # NEW — load_preprocessor / load_model_card / load_prompts
└── models/
    ├── sam3.py                # CHANGED — extract validate_forward_inputs; route FindStage via _multiplex
    └── _multiplex.py          # NEW — multiplex_index_arrays (torch-free)

tests/
├── export/                    # NEW — round-trip, --check pass/fail, QLoRA dequant
└── predict/                   # CHANGED — --use-onnx smoke + subprocess ORT-only load
```

---

## 4. CLI surface

### 4.1 `csp export` new flags

`src/custom_sam_peft/cli/export_cmd.py` — add after line 22 (the `--merge` option). Verbatim declarations:

```python
to: str = typer.Option("pytorch", "--to", help="Export format: pytorch (default) or onnx.")
opset: int = typer.Option(17, "--opset", help="ONNX opset version (floor 17).")
fp16: bool = typer.Option(False, "--fp16", help="Export weights in fp16 (required for QLoRA).")
include: str = typer.Option("all", "--include", help="ONNX bundle parts: encoder|decoder|all.")
dynamic_axes: bool = typer.Option(
    True, "--dynamic-axes/--no-dynamic-axes",
    help="Dynamic batch dim (spatial stays pinned to the model image size).",
)
check: bool = typer.Option(False, "--check", help="Verify torch-vs-ORT parity after export; fail on drift.")
quantize: str = typer.Option(
    "none", "--quantize",
    help="Quantization: none|int8-dynamic (int8-dynamic is RESERVED, not implemented).",
)
```

Defaults: `--to pytorch`, `--opset 17`, `--fp16` off, `--include all`, `--dynamic-axes` on, `--check` off, `--quantize none`.

Validation in `export()` before dispatch:
- `to` ∈ {`pytorch`,`onnx`} else `typer.BadParameter`.
- `include` ∈ {`encoder`,`decoder`,`all`} else `typer.BadParameter`.
- `opset < 17` → `typer.BadParameter("--opset floor is 17.")`.
- `quantize != "none"` → `typer.BadParameter("--quantize int8-dynamic is reserved and not yet implemented.")` (do not silently ignore; §9).
- If `to == "pytorch"` and any ONNX-only flag was set non-default → INFO log "ignored for --to pytorch" (do not error; keeps the pytorch path inert).

### 4.2 Dispatch (preserves pytorch bit-for-bit)

Replace the single dispatch at `export_cmd.py:42` and the output messages (lines 44-47):

```python
if to == "pytorch":
    out = run_export(cfg, checkpoint, merge=merge, output=output)   # UNCHANGED path
else:  # to == "onnx"
    from custom_sam_peft.export.onnx import run_export_onnx
    out = run_export_onnx(
        cfg, checkpoint, output=output,
        opset=opset, fp16=fp16, include=include,
        dynamic_axes=dynamic_axes, check=check,
    )

if to == "pytorch":
    rprint(f"[green]{'merged' if merge else 'adapter'}[/green] {out}")
else:
    rprint(f"[green]onnx bundle[/green] {out}")
```

The `from ... import run_export_onnx` is lazy (inside the `else`) so the pytorch path never imports torch.onnx/onnxruntime machinery.

### 4.3 `csp predict --use-onnx`

`src/custom_sam_peft/cli/predict_cmd.py` — add after the `--checkpoint`/`--config` block (~line 206):

```python
use_onnx: Path | None = typer.Option(
    None, "--use-onnx",
    help="Run inference from an exported ONNX bundle dir (image_encoder.onnx + decoder.onnx "
         "+ sidecars) instead of the PyTorch model. Mutually exclusive with --checkpoint/--merge.",
)
```

Validation in `predict()` before building `PredictOptions`:
- `use_onnx` set + `checkpoint` set → `typer.BadParameter("--use-onnx and --checkpoint are mutually exclusive; the bundle already has the adapter merged in.")`
- `use_onnx` set + `merge` set → same error class.
- `use_onnx` dir missing, or missing any of `decoder.onnx`, `preprocessor.json`, `model_card.json`, `prompts.txt` (and `image_encoder.onnx` when `model_card.json["include"] != "decoder"`) → `typer.BadParameter` listing the missing file(s).

Pass `use_onnx` through to `PredictOptions` (§7).

---

## 5. Export module — `src/custom_sam_peft/export/onnx.py`

Entry point:

```python
def run_export_onnx(
    cfg: TrainConfig,
    checkpoint: Path,
    *,
    output: Path,
    opset: int,
    fp16: bool,
    include: str,         # "encoder" | "decoder" | "all"
    dynamic_axes: bool,
    check: bool,
) -> Path:
    """Merge adapter, trace SAM 3.1 into a two-file ONNX bundle + sidecars at `output`.
    Returns the bundle directory. Raises on QLoRA+fp16-off, missing CUDA for QLoRA,
    or parity drift (--check)."""
```

Module-level constants:

```python
ENCODER_FILE = "image_encoder.onnx"
DECODER_FILE = "decoder.onnx"
PREPROCESSOR_FILE = "preprocessor.json"
PROMPTS_FILE = "prompts.txt"
MODEL_CARD_FILE = "model_card.json"
PREPROCESSOR_SCHEMA_VERSION = 1
MODEL_CARD_SCHEMA_VERSION = 1
_PARITY_KEYS = ("pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec")
_PARITY_TOL = {"fp32": (1e-3, 1e-3), "fp16": (1e-2, 1e-2)}  # (atol, rtol)
```

Orchestrator flow:

1. **Staging.** Write the whole bundle to a sibling temp dir `staging = output.with_name(output.name + ".tmp-onnx")` (created fresh; removed on any failure). Promote `staging → output` via `os.replace`/`shutil.move` only at the very end (after `--check` passes, if requested). This guarantees a `--check`-failed export leaves no loadable bundle.
2. `wrapper, method, export_dtype = _merge_and_cast(cfg, checkpoint, fp16=fp16)` (§5.1).
3. `class_names = _resolve_class_names(cfg)` — build the dataset from `cfg.data.train` and read `dataset.class_names` (same property used at `bundle.py:257`, `data/base.py:72`). Raise a clear error if empty (mirror `bundle.py:259`).
4. Trace the requested `.onnx` file(s) (§5.2/§5.3), gated on `include`.
5. Write sidecars **always** (§5): `_write_preprocessor`, `_write_prompts(class_names)`, then `_write_model_card` LAST.
6. If `check`: run `_run_parity_check(...)` (§6) against the **staging** bundle BEFORE writing `model_card.json` (so a drift abort never leaves a card claiming `parity_checked: true`). On success set `parity_checked=True` in the card.
7. Promote staging → output; return `output`.

### 5.1 Merge + precision — `_merge_and_cast`

```python
def _merge_and_cast(
    cfg: TrainConfig, checkpoint: Path, *, fp16: bool,
) -> tuple[Sam3Wrapper, str, torch.dtype]:
    """Load adapter, merge deltas (mandatory), cast to export precision.
    Returns (wrapper, method, export_dtype). method ∈ {"lora","qlora"};
    export_dtype ∈ {torch.float16, torch.float32}. The wrapper is .eval(),
    requires_grad=False, single uniform dtype. Caller reads wrapper.model.model
    (merged raw Sam3Image) and wrapper.model.channel_adapter."""
```

Steps (reuse existing functions; do NOT reimplement):

```python
from custom_sam_peft.peft_adapters import discover_method_from_checkpoint
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.train.checkpoint import load_adapter
from custom_sam_peft.peft_adapters.lora import merge_lora

method = discover_method_from_checkpoint(checkpoint)  # pure JSON, no torch/bnb (peft_adapters/__init__.py:161)

# Preflight guards (fail fast, before load+merge cost):
if method == "qlora" and not fp16:
    raise ValueError(
        "QLoRA adapters dequantize to fp16/bf16; exporting them to fp32 ONNX "
        "(--fp16 off) upcasts the full merged model and can OOM on memory-tight "
        "machines. Fix: re-run with --fp16."
    )
if method == "qlora" and not torch.cuda.is_available():
    raise RuntimeError(
        "ONNX export of a QLoRA adapter requires a CUDA device (4-bit dequantize "
        "runs on GPU). No CUDA device is visible. Fix: export on a GPU machine, "
        "or re-train/save a LoRA (non-quantized) adapter."
    )

device = torch.device("cuda" if (method == "qlora" or torch.cuda.is_available()) else "cpu")
wrapper = load_sam31(cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics)
wrapper = wrapper.to(device)
wrapper = load_adapter(wrapper, checkpoint)   # dispatches load_lora/load_qlora + restores channel_adapter
merge_lora(wrapper)                            # folds deltas; QLoRA dequantizes 4-bit -> compute_dtype (lora.py:236)
merged = wrapper.model.model                   # merged raw Sam3Image; wrapper.peft_model is now None

# Normalize precision to a single uniform graph dtype (never emit bf16):
if fp16:
    merged.half(); export_dtype = torch.float16
else:
    merged.float(); export_dtype = torch.float32   # LoRA only reaches here
merged.eval()
for p in merged.parameters():
    p.requires_grad_(False)
# QLoRA: move merged module to CPU before tracing once fp16 cast frees the VRAM transient.
if method == "qlora":
    wrapper.model.model = merged.to("cpu")
    if wrapper.model.channel_adapter is not None:
        wrapper.model.channel_adapter = wrapper.model.channel_adapter.to("cpu").to(export_dtype)
return wrapper, method, export_dtype
```

Notes:
- Reuse `merge_lora` (not `save_merged`) — `save_merged` also writes `pytorch_model.bin`, which the ONNX path does not want. `merge_lora` is the same function `save_merged` calls (`checkpoint.py:144`).
- QLoRA `compute_dtype` may be `bfloat16` or `float16`; we always normalize to fp16 (or fp32 for LoRA). bf16 ONNX at opset 17 is poorly supported and not shipped. `coerce_dtype_for_capability` is NOT used here — export precision is user-driven by `--fp16`.
- The transient OOM site is the merge dequant (full weight set materialized in `compute_dtype`), not tracing — hence the QLoRA+fp16-off fail-fast guard.

### 5.2 `image_encoder.onnx` — `_EncoderExport`

The vision path is prompt-independent. Trace a thin `nn.Module` shim wrapping `wrapper.model.channel_adapter` (optional, N→3 Conv2d on `_Sam3ImageAdapter`) + `merged.backbone.forward_image`:

```python
class _EncoderExport(nn.Module):
    def __init__(self, merged, channel_adapter):  # channel_adapter may be None
        ...
    def forward(self, images: Tensor):           # images: (B, C, 1008, 1008), C = cfg.data.channels
        x = self.channel_adapter(images) if self.channel_adapter is not None else images
        backbone_out = self.merged.backbone.forward_image(x)   # sam3.py:332
        # flatten backbone_fpn list -> fixed L feature tensors + L pos tensors
        return (*feats, *pos)
```

- **Inputs:** `images` `(B, C, 1008, 1008)`, `C = cfg.data.channels`.
- **Outputs:** the flattened vision feature tensors + positional tensors of `backbone_out` (fixed count `L` determined by the model's FPN levels; capture at trace time).
- **Dynamic axes** (when `--dynamic-axes`): dim 0 (batch) of `images` and every output marked dynamic; spatial pinned to 1008. Dummy `images` traced with `B=2` so the batch axis is genuinely exercised. When `--no-dynamic-axes`: all axes static at `B=2`.
- `torch.onnx.export(_EncoderExport(...), (images_dummy,), staging/ENCODER_FILE, opset_version=opset, input_names=["images"], output_names=[...], dynamic_axes=...)`.

Channel adapter is folded INTO `image_encoder.onnx` (not a third artifact). For `channel_semantics == "rgb"` with no adapter it is a passthrough. **Non-rgb (rgba/grayscale/freeform) is in scope for v1** (resolved §12.2): the folded N→3 Conv2d must trace cleanly, and `preprocessor.json` must carry the correct freeform `mean`/`std`/`channels`/`channel_semantics` (the §6.1 warning fires if a non-rgb config left `normalize` unset). A non-rgb export+round-trip is a required test (§10).

### 5.3 `decoder.onnx` — `_DecoderExport`

The decoder fuses `forward_text` + `forward_grounding`. `forward_text` consumes a Python `list[str]` (non-traceable), so the text class list is **baked at export time for the training class list** into a constant text-embedding tensor; the decoder graph consumes that constant + vision features + the multiplex index tensors + zero box/point prompt embeddings. The bundle is therefore class-list-specific (K baked).

```python
class _DecoderExport(nn.Module):
    def __init__(self, merged, baked_text_embed, b, k):
        ...
    def forward(self, *vision_feats):
        backbone_out = self._rebuild_backbone_out(vision_feats)
        find_input = self._build_find_stage(self.b, self.k)   # img_ids/text_ids via _multiplex semantics
        prompt = self._zero_prompt(self.b, self.k)            # dummy zero box/point embeddings
        # export-only grounding core: drops training _compute_matching, strips record_function
        out = self._grounding_core(backbone_out, find_input, find_target=None, geometric_prompt=prompt)
        return out["pred_logits"], out["pred_boxes"], out["pred_masks"], out["presence_logit_dec"]
```

- **Inputs:** the vision feature tensors (same tensors `image_encoder.onnx` emits — this is the encoder↔decoder boundary). Text embedding is baked (constant), not an input.
- **Outputs** (rows `R = B*K`, image-major / class-minor ordering per `_multiplex`):

  | key | shape | dtype |
  |---|---|---|
  | `pred_logits` | `(R, N, 1)` | export_dtype |
  | `pred_boxes` | `(R, N, 4)` normalized cxcywh | export_dtype |
  | `pred_masks` | `(R, N, 288, 288)` logits (`mask_size`) | export_dtype |
  | `presence_logit_dec` | `(R, 1)` | export_dtype |

- **Dynamic axes** (when `--dynamic-axes`): dim 0 (rows) of every output dynamic; spatial pinned. `B=2`, baked `K` from the class list at trace time.
- The multiplex index construction (`img_ids = arange(B).repeat_interleave(K)`, `text_ids = arange(K).repeat(B)`, `sam3.py:341-356`) is routed through `models/_multiplex.multiplex_index_arrays(b, k)` so the traced ordering and the ORT decoder feed share one source of truth (§7.2).

---

## 6. Bundle layout & sidecar schemas

Flat directory at `output/`. Contents by `--include`:

| File | `all` | `encoder` | `decoder` |
|---|---|---|---|
| `image_encoder.onnx` | ✓ | ✓ | – |
| `decoder.onnx` | ✓ | – | ✓ |
| `preprocessor.json` | ✓ | ✓ | ✓ |
| `prompts.txt` | ✓ | ✓ | ✓ |
| `model_card.json` | ✓ | ✓ | ✓ |

There is **no** `README.md` (§1.3). Sidecars are always written. `model_card.json` is the manifest and is written LAST (presence signals a complete bundle).

### 6.1 `preprocessor.json` (cross-ref #69 — must equal what the transform pipeline resolves)

```json
{
  "schema_version": 1,
  "image_size": 1008,
  "mean": [0.5, 0.5, 0.5],
  "std": [0.5, 0.5, 0.5],
  "max_pixel_value": 255.0,
  "normalization_path": "table-fallback",
  "channels": 3,
  "channel_semantics": "rgb",
  "resize_interpolation": "INTER_LINEAR",
  "mask_interpolation": "INTER_NEAREST",
  "pad_position": "top_left",
  "border_mode": "BORDER_CONSTANT",
  "border_fill_value": 0
}
```

Field sourcing (all reuse, no new resolution logic):

| Field | Source |
|---|---|
| `schema_version` | constant `PREPROCESSOR_SCHEMA_VERSION` = 1 |
| `image_size` | `from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE` (= 1008; `sam3.py:111`) — NOT from config |
| `mean`, `std`, `normalization_path` | `mean, std, path = resolve_normalization_with_path(cfg.model.name, cfg.data.normalize or NormalizeConfig(), channel_semantics=cfg.data.channel_semantics)` (`data/transforms.py:113`). Coalesce `None → NormalizeConfig()` exactly as the pipeline does |
| `max_pixel_value` | `(cfg.data.normalize or NormalizeConfig()).max_pixel_value` (default 255.0; `schema.py:316`) — record exactly (1.0 for float multi-band) |
| `channels` | `cfg.data.channels` (`schema.py:464`) |
| `channel_semantics` | `cfg.data.channel_semantics` (`schema.py:478`) |
| `resize_interpolation` | constant string `"INTER_LINEAR"` (`transforms.py:217`) |
| `mask_interpolation` | constant string `"INTER_NEAREST"` (`transforms.py:218`) — reference only |
| `pad_position` | constant `"top_left"` (`transforms.py:226`) |
| `border_mode` | constant string `"BORDER_CONSTANT"` (`transforms.py:223`) |
| `border_fill_value` | constant `0` (`transforms.py:224`) |

Store cv2 constants as **string names**, not the raw `cv2.INTER_*` ints (build-dependent); predict-side maps name→cv2 int. Add a comment cross-referencing `transforms.py:215-226`.

**Warning:** if `cfg.data.channel_semantics != "rgb"` and `cfg.data.normalize is None`, log a WARNING — freeform/grayscale has no default mean/std, the bundle would ship the schema default `[0.5,...]` (the `config-fallback` path), which is almost certainly wrong.

### 6.2 `model_card.json`

```json
{
  "schema_version": 1,
  "name": "facebook/sam3.1",
  "base": "facebook/sam3.1",
  "training_config_hash": "a3f1...e9",
  "opset": 17,
  "fp16": false,
  "include": "all",
  "dynamic_axes": true,
  "parity_checked": false,
  "git_sha": "272d540",
  "version": "0.9.1.dev6+g272d54094",
  "exported_at": "20260603-141233"
}
```

| Field | Source |
|---|---|
| `schema_version` | constant `MODEL_CARD_SCHEMA_VERSION` = 1 |
| `name` | `cfg.model.name` (`schema.py:120`) |
| `base` | `cfg.model.name` (same value for v1; kept distinct for a future base-vs-finetune registry) |
| `training_config_hash` | `from custom_sam_peft.train.checkpoint import _hash_cfg; _hash_cfg(cfg)` (`checkpoint.py:102`) — reuse the EXACT function so the hash matches the checkpoint's `cfg_hash` |
| `opset` | `--opset` |
| `fp16` | `--fp16` |
| `include` | `--include` (lets predict know which `.onnx` files to expect) |
| `dynamic_axes` | `--dynamic-axes` |
| `parity_checked` | `True` only if `--check` ran and passed (export fails on drift, so a card with `true` means parity held) |
| `git_sha` | subprocess `git rev-parse --short HEAD`, `cwd=Path(custom_sam_peft.__file__).parent`, `check=False`; JSON `null` if `returncode != 0` (mirror `predict/writers.py:223-230`; factor into a shared `_git_sha() -> str | None`) |
| `version` | `custom_sam_peft.__version__` (`_version.py`) |
| `exported_at` | `datetime.now(UTC).strftime("%Y%m%d-%H%M%S")` (`runner.py:30` pattern) — EXPORT-execution time |

### 6.3 `prompts.txt`

Newline-delimited UTF-8, one training class per line in `dataset.class_names` order (index = predict category order), trailing newline. Do NOT re-sort. Raise if `class_names` is empty.

### 6.4 Serialization

All JSON via `json.dumps(record, indent=2) + "\n"` then `Path.write_text(...)` (matches `save_qlora` convention, `qlora.py:391`; human-inspected so indent=2). `mean`/`std` emitted as plain `list[float]` (no numpy/torch types) so the subprocess load test path never imports them transitively.

---

## 7. `--check` parity contract

Private `_run_parity_check(staging_dir, wrapper, cfg, *, fp16, include)` invoked after artifacts are written to staging and BEFORE `model_card.json` / promotion. Compares the **composed ORT bundle** (encoder→decoder, wired exactly as `--use-onnx` wires it) against the **merged torch `Sam3Wrapper.forward`**, on the four `_PARITY_KEYS`. ORT uses `CPUExecutionProvider` only (deterministic, reproducible in CI, single fixed tolerance band).

### 7.1 Synthetic input (deterministic, no disk)

```python
g = torch.Generator(device="cpu").manual_seed(0)
C = cfg.data.channels; B = 2; H = W = SAM3_IMAGE_SIZE
dtype = torch.float16 if fp16 else torch.float32
images = torch.rand(B, C, H, W, generator=g).to(dtype)   # RAW floats; parity tests the GRAPH, not preprocessing
classes = _resolve_class_names(cfg); K = min(len(classes), 2) or 1
prompts = [TextPrompts(classes=tuple(classes[:K] or ["object"])) for _ in range(B)]
```

`B=2` exercises the dynamic batch axis (B=1 would silently pass a batch-hardcoded graph). `K=2` exercises the B×K multiplex. Spatial pinned to 1008.

### 7.2 Comparison & tolerances

Both sides upcast to fp32 numpy before differencing (fp16 subtraction would inject ~1e-3 noise). Torch reference runs the SAME merged module at the SAME export dtype (compare fp16-torch vs fp16-ORT), so the band measures ONNX/ORT divergence, not torch-fp32-vs-onnx-fp16 precision loss.

Per-key, `numpy.allclose` semantics `tol = atol + rtol*|ref|`; on failure report max-abs-diff and the worst-element index.

| band | atol | rtol | justification |
|---|---|---|---|
| fp32 (`--fp16` off) | 1e-3 | 1e-3 | spec-mandated; fp32 ORT/torch typically agree to ~1e-5 — 1e-3 is a wide margin that still catches op-mismatch bugs |
| fp16 (`--fp16` on) | 1e-2 | 1e-2 | 10× looser; fp16 mantissa ~11 bits → per-op resolution ~5e-4, accumulated over the ViT+decoder depth with ORT/torch op-fusion reordering. `# tbd:` if real-checkpoint `pred_masks` exceeds 1e-2, widen masks-only to 2e-2 (not silently shipped) |

### 7.3 Failure & partial-include

```python
class ExportParityError(RuntimeError):
    """torch-vs-ORT parity failed; the ONNX bundle was NOT promoted."""
```

First drifting (or shape-mismatched) key raises `ExportParityError` naming the key, the stat (max-abs-diff + worst index + both values), the active band tolerances, and a one-line hint (fp16 → retry without `--fp16` to isolate quantization drift; else unsupported-op/tracing-mismatch → bump `--opset`, file an issue). The orchestrator removes staging and re-raises.

For `--include encoder` / `--include decoder` the composed forward cannot run end-to-end; parity is scoped to the present graph's output against the corresponding torch intermediate (encoder feats; or decoder fed torch-produced encoder feats). The intermediate tensor boundary is the same one §5.2/§5.3 define.

---

## 8. `csp predict --use-onnx` seam

### 8.1 `PredictOptions` change

`predict/runner.py` frozen dataclass (~line 49): add `use_onnx: Path | None` as a **non-default** field (insert before `batch_size`). This is a required-field blast radius — every `PredictOptions(...)` constructor in `tests/predict`, `tests/gpu`, and `predict_cmd.py` must pass `use_onnx=None`. Grep `PredictOptions(` repo-wide; run the full predict+cli+gpu suites, not just the new test.

### 8.2 Dispatch branch (runner.py Step 4/5, ~lines 313-352)

```python
if opts.use_onnx is not None:
    from custom_sam_peft.predict.onnx_session import OnnxSam3Session
    from custom_sam_peft.predict.onnx_bundle import load_preprocessor, load_model_card, load_prompts
    pp = load_preprocessor(opts.use_onnx)        # overrides rcfg pre-proc values (§8.3)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if rcfg.device == "cuda" else ["CPUExecutionProvider"])
    model = OnnxSam3Session(opts.use_onnx, providers=providers)  # CPU fallback + WARN if CUDA EP missing
    adapter_kind_str = "onnx-bundle"
    # build transforms from pp (mean/std/max_pixel_value/channel_semantics/image_size)
else:
    # existing load_sam31 + adapter load + merge + transforms (UNCHANGED)
```

The forward loop (`runner.py:486-547`), warmup (`388-394`, runs under `suppress(Exception)`), OomLadder, semantic branch, and writers are unchanged. `run.json` records `"model_source": "onnx"`, `"onnx_bundle": str(opts.use_onnx)`, plus the card's `git_sha`/`opset` for provenance.

**Semantic task is in scope for `--use-onnx`** (resolved §12.4). `OnnxSam3Session.__call__` returns the same four-key dict pre-marginalization, so the existing semantic branch (`marginalize_group`/`build_semantic_logits`/`semantic_argmax`) runs unchanged over ORT outputs. The decoder graph (§5.3) emits raw `pred_logits`/`pred_masks` *before* any marginalization, so semantic reduction stays a predict-side concern exactly as in the torch path. A semantic `--use-onnx` round-trip is a required test (§10); the semantic eval reduction knobs apply identically to both paths (no ONNX-specific divergence).

### 8.3 Sidecar-driven preprocessing (config-free)

When `use_onnx` is set, skip adapter-based model-name resolution in `_resolve_config` and read `preprocessor.json` via `onnx_bundle.load_preprocessor`. Build `NormalizeConfig(mean=..., std=..., max_pixel_value=...)` and call `build_eval_transforms(image_size, model_name=<from model_card.json>, normalize=normalize_cfg, channel_semantics=...)`. Trust the sidecar `image_size` (== 1008), not the constant, so a future resolution change can't silently desync. `max_pixel_value` MUST come from the sidecar (the torch path silently takes the 255.0 default at `runner.py:346`). `load_prompts` is loaded for a WARN-level cross-check: a `--prompts` class not in the bundle's training list warns but proceeds.

### 8.4 ONNX session wrapper — `predict/onnx_session.py`

```python
class OnnxSam3Session:
    """Drop-in for Sam3Wrapper in the predict forward loop.
    __call__(images: Tensor, prompts: list[TextPrompts], support=None)
        -> dict[str, torch.Tensor]  # pred_logits, pred_boxes, pred_masks, presence_logit_dec
    """
    def __init__(self, bundle_dir: Path, *, providers: list[str]): ...
    def __call__(self, images, prompts, support=None) -> dict[str, "torch.Tensor"]:
        # 1. validate_forward_inputs(images, prompts, channels)  (extracted free fn, §8.5)
        # 2. images -> numpy
        # 3. encoder once per image batch (LRU keyed on images.data_ptr()+shape)
        # 4. _OrtCore.run_decoder(vision_feats, classes) per prompt group
        # 5. torch.from_numpy each output -> dict keyed pred_logits/pred_boxes/pred_masks/presence_logit_dec


class _OrtCore:
    """TORCH-FREE. numpy + onnxruntime only. The subprocess load test imports THIS."""
    def __init__(self, bundle_dir, providers):
        import onnxruntime as ort   # lazy; never torch
        self.enc = ort.InferenceSession(str(bundle_dir / "image_encoder.onnx"), providers=providers)
        self.dec = ort.InferenceSession(str(bundle_dir / "decoder.onnx"), providers=providers)
    def run_encoder(self, np_img) -> dict[str, np.ndarray]: ...
    def run_decoder(self, vision_feats, classes) -> dict[str, np.ndarray]:
        # builds FindStage index arrays (multiplex_index_arrays) + zero prompt embeddings as numpy
```

`torch` is imported in `onnx_session.py` ONLY to bridge ORT numpy → torch-typed postprocess (`queries_to_coco_results` does `.sigmoid()`, `F.interpolate`, `.cpu().numpy()`). `_OrtCore`, `models/_multiplex`, and `onnx_bundle.*` import only numpy + onnxruntime + stdlib.

### 8.5 Shared refactors (extract, don't duplicate)

- `models/_multiplex.multiplex_index_arrays(b, k) -> tuple[np.ndarray, np.ndarray]` (torch-free): the single source of truth for `img_ids`/`text_ids` row ordering, consumed by both the torch `_Sam3ImageAdapter.forward` (re-routed to build its `torch.arange` FindStage from this) and the ONNX decoder feed (export-time §5.3 and predict-time `_OrtCore`).
- `models/sam3.py`: extract `Sam3Wrapper._validate_inputs` (sam3.py:168-206) into a module-level free fn `validate_forward_inputs(images, prompts, channels)` so `OnnxSam3Session` enforces the identical K∈[1,CAP] + shared-class-list contract.
- **`_git_sha()` helper** (resolved §12.7): extract the duplicated `git rev-parse --short HEAD` block in `predict/writers.py:223-230` into one shared free fn (e.g. `custom_sam_peft/_provenance.py::git_sha() -> str | None`), consumed by both `writers.py` and `export/onnx.py`'s `model_card.json` writer (§6.2). Extract over mirror — keeps the provenance source single.

Pre/post-processing (`_read_image`, `build_eval_transforms`, `torch.stack`; `_row_outputs`, `queries_to_coco_results`, score filter, top-k) is reused verbatim — no churn.

---

## 9. Reserved-but-unimplemented: `--quantize`

`--quantize {none,int8-dynamic}` parses but only `none` is implemented. Any other value raises at CLI validation (`typer.BadParameter`, §4.1) — fail loud, never silently ignore. The flag exists so the surface is stable for a future int8 follow-up; no int8 codepath, calibration, or ORT quantization tooling is added in v1. `model_card.json` carries no `quantize` field in v1 (add when implemented).

---

## 10. Test plan

All CPU, using `TinySam3Stub` (`tests/fixtures/tiny_sam3_stub.py`, 4 queries / small mask) and `tiny_coco_dir`. The stub emits the four-key SAM3-shaped dict, so the composed comparison runs end-to-end at tiny scale.

`tests/export/`:
1. **Round-trip** (`include` parametrized): export a stub-backed adapter, assert the bundle dir contains EXACTLY the expected file set per `--include` (e.g. `include=encoder` → `image_encoder.onnx` present, `decoder.onnx` absent, all three sidecars present, **no** `README.md`). Assert `preprocessor.json` equals a direct `resolve_normalization_with_path(...)` call on the same `cfg` and `image_size == SAM3_IMAGE_SIZE`; `model_card.json["training_config_hash"] == _hash_cfg(cfg)`; `prompts.txt` == `dataset.class_names` in order.
2. **`--check` passes**: export with `--check`, assert no raise and bundle promoted to `output`.
3. **`--check` fails on drift**: monkeypatch the ORT session shim to perturb one output key (> tol, e.g. +0.5 to `pred_masks`); assert `ExportParityError` names `pred_masks` and NO dir exists at `output` (staging removed). Parametrize `fp16 ∈ {False, True}` and assert the message reports the matching band.
4. **QLoRA dequant export**: monkeypatch bnb availability + write a `custom_sam_peft_qlora.json` marker; assert the merged/traced module weights are fp16 (dtype assertion). Assert QLoRA + `--fp16 off` raises `ValueError` whose message contains the `--fp16` hint substring. Assert the CUDA-required guard raises `RuntimeError` when `torch.cuda.is_available()` is forced `False`.
5. **Non-rgb export** (resolved §12.2): export a stub-backed adapter with `cfg.data.channels != 3` / non-rgb `channel_semantics` (explicit `normalize` set). Assert the folded N→3 Conv2d channel adapter traces into `image_encoder.onnx` (encoder input channel dim == `cfg.data.channels`), `preprocessor.json` carries the freeform `mean`/`std`/`channels`/`channel_semantics`, and the no-`normalize` case emits the §6.1 WARNING.

`tests/predict/`:
6. **Separate-process ORT-only load** (the #77 guard): build a tiny bundle (trace `TinySam3Stub` submodules), then `subprocess.run([sys.executable, "-c", "import sys, onnxruntime; onnxruntime.InferenceSession(...); assert 'torch' not in sys.modules; print('OK')"])`; assert returncode 0, `"OK"` in stdout, no torch import.
7. **`--use-onnx` round-trip (CPU)**: build a tiny bundle, run `run_predict(opts with use_onnx=bundle)`, assert `predictions.json`/`run.json` written and `run.json["model_source"] == "onnx"`. Composition smoke test.
8. **Parity vs torch (CPU)**: run torch and ORT `run_predict` on the same synthetic image+prompts; assert identical entry count and `score`/`bbox` within atol/rtol 1e-3 (end-to-end COCO-entry parity; `--check` covers tensor-level).
9. **Semantic `--use-onnx` (CPU)** (resolved §12.4): build a tiny bundle, run `run_predict` with `task == "semantic"` and `use_onnx=bundle`; assert the semantic branch (`marginalize_group`/`build_semantic_logits`/`semantic_argmax`) produces a semantic output mask and that it matches the torch semantic path within tolerance on the same input. Guards that ORT's pre-marginalization decoder outputs feed the reduction faithfully.
10. **Torch-free unit**: import `_OrtCore`, `_multiplex`, `onnx_bundle` in a child `python -c` and assert `"torch" not in sys.modules`.
11. **Required-field blast radius**: full `tests/predict` + `tests/cli` + `tests/gpu` suites pass after adding `PredictOptions.use_onnx` (every constructor updated).

Run the export tests via the CPU lane (`-o "addopts="` to bypass the global `--cov-fail-under`, per repo convention); the subprocess test must not pull the real GPU suite.

---

## 11. Out of scope

See §1.3. Summary: dropped guide/examples/bundle-README (OBE); int8 quantization (reserved, §9); TensorRT/CoreML/TFLite; browser-optimized encoder; benchmarking; dynamo export.

---

## 12. Resolved decisions (review, 2026-06-03)

All seven review questions are resolved; carry these into implementation.

1. **Encoder↔decoder graph boundary — CONFIRMED.** Bake the training class list into a constant text-embedding tensor in `decoder.onnx`; the bundle is class-list-specific (K baked) and prompts beyond the training list are not representable (documented v1 limitation). The implementer must still verify at build time that `forward_text`+`forward_grounding` trace cleanly under `torch.onnx.export` with the `_DecoderExport` shim (the spike `_grounding_core` dropping `_compute_matching` and stripping `record_function`) — this is the load-bearing trace risk. The export cut (§5.3) and the predict-side `onnx_session.py` caching boundary (§8.4) move in lockstep.
2. **Non-rgb channel adapter — IN SCOPE.** v1 validates non-rgb (rgba/grayscale/freeform) bundles, not rgb-only. The folded N→3 Conv2d (§5.2) must trace; `preprocessor.json` carries the freeform `mean`/`std`/`channels`/`channel_semantics` (§6.1), with the WARNING firing when a non-rgb config left `normalize` unset. Covered by export test §10.5.
3. **fp16 `--check` tolerance — ACCEPTED as default.** Ship `atol/rtol 1e-2`; if `pred_masks` exceeds it on a real merged SAM3.1 fp16 export, widen masks-only to `2e-2` tagged `# tbd:`. One real-GPU export run validates the default before shipping (does not block spec sign-off).
4. **Semantic task under `--use-onnx` — IN SCOPE.** v1 supports `task == "semantic"`, not instance-only. The decoder emits raw pre-marginalization outputs (§5.3) so the existing semantic reduction runs predict-side unchanged (§8.2). Covered by predict test §10.9.
5. **`model_card.json["base"]` == `name` — CONFIRMED.** Duplicate `cfg.model.name` into `base` for v1; no separate base-vs-finetuned field is read.
6. **`prompts.txt` order — CONFIRMED required.** Rely on `Dataset.class_names` order; the implementer confirms every in-scope format (coco / hf / mask_png / semantic_hf / subset) returns a stable, category-aligned order, and test §10.1 asserts `prompts.txt == dataset.class_names` in order.
7. **Shared `_git_sha()` — EXTRACT.** Lift `predict/writers.py:223-230` into one shared helper consumed by both `writers.py` and `export/onnx.py` (§8.5), rather than mirroring.
