# Colab GPU Integration Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all 9 GPU integration tests pass on Colab T4 by (a) completing the `_Sam3ImageAdapter.forward` stub with the `forward_grounding` recipe drawn from `sam3.model.sam3_image_processor.Sam3Processor`, and (b) pinning `SCOPE_TARGETS` to the real SAM 3.1 module names. Unit tests stay green; existing gzip-fix commit `517ff6a` is preserved.

**Architecture:** Two surgical fixes on branch `worktree-fix+colab-bpe-gzip` (PR #13). Each fix is a single-file production change plus the matching test updates. No schema changes, no new dependencies, no incidental refactors.

**Tech stack:** Python 3.13, PyTorch, HuggingFace `peft`, Meta `sam3`, `pytest`, `ruff`. No new deps.

**Reference spec:** `docs/superpowers/specs/2026-05-17-colab-gpu-integration-fix-design.md`

---

## Pre-flight checks

Run these once before starting Task 1:

```bash
# Confirm you are in the worktree, not the main checkout.
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip rev-parse --show-toplevel
# Expected: /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip

# Confirm branch and tip.
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip status
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip log --oneline -5
# Expected branch: worktree-fix+colab-bpe-gzip
# Expected tip:    517ff6a fix(models): drop bpe_path override so sam3 uses its bundled gzipped vocab

# Confirm unit tests pass at the current baseline (200 passing).
uv run pytest tests/unit -q

# Confirm sam3 is importable and the expected helpers exist.
uv run python -c "import sam3; from sam3.model.data_misc import FindStage; from sam3.model.sam3_image import Sam3Image; print(hasattr(Sam3Image, 'forward_grounding'))"
# Expected: True
```

If any pre-flight check fails, STOP and investigate. Do not start the plan on top of a broken tree.

---

## File map (what gets touched)

| File | Action | Owning task |
| --- | --- | --- |
| `logs/log.md` | Create (new) | 1 |
| `src/esam3/models/sam3.py` | Modify (`_Sam3ImageAdapter.forward` body + new import) | 2 |
| `src/esam3/peft_adapters/lora.py` | Modify (`SCOPE_TARGETS` dict) | 3 |
| `tests/fixtures/tiny_sam3_lora_stub.py` | Modify (rename subtrees to real-name shape) | 3 |
| `tests/unit/test_peft_lora.py` | Modify (rename substrings; add 1 new regex test) | 3 |
| `tests/integration/test_peft_lora_real.py` | Modify (rename asserted substrings) | 4 |
| `tests/integration/test_peft_qlora_real.py` | Modify (rename asserted substrings) | 4 |
| `tests/integration/test_load_sam31_real.py` | No change (asserts shape contract Task 2 satisfies) | n/a |

No production code outside `sam3.py` and `lora.py` changes. No schema changes. No `pyproject.toml` changes.

---

## Task 1: Bootstrap logs/log.md

**Difficulty:** L
**Subagent:** `implementer-simple` (single new file, no logic).
**Files:**
- Create: `logs/log.md`

### Scope

Create the append-only log file required by the repo `CLAUDE.md` convention. The plan and spec mandate `[TIMESTAMP] [ROLE] action` entries; this task seeds the file with the bootstrap entry.

### Steps

- [ ] **Step 1: Verify the file does not exist**

```bash
test -e logs/log.md && echo "ALREADY EXISTS, do not overwrite" || echo "OK to create"
```

If the file already exists, STOP and ask before proceeding.

- [ ] **Step 2: Create the file with the bootstrap entry**

Content (replace `<UTC-ISO8601>` with the actual timestamp at write time):

```
# Log

[<UTC-ISO8601>] [planner] bootstrap logs/log.md for spec 2026-05-17-colab-gpu-integration-fix-design.md
```

### Definition of Done

- [ ] `ls logs/log.md` succeeds.
- [ ] The file is exactly 3 lines (heading, blank, one bootstrap entry).
- [ ] No emojis in the file.

### Verification

```bash
wc -l logs/log.md
# Expected: 3
```

### Rollback

```bash
rm logs/log.md
```

---

## Task 2: Implement `_Sam3ImageAdapter.forward`

**Difficulty:** M
**Subagent:** `implementer` (Sonnet/high). This is a single-file change but the logic is load-bearing for 1 integration test and downstream LoRA flow.
**Files:**
- Modify: `src/esam3/models/sam3.py` (lines 8-24 for the import block; lines 108-124 for the function body)
**Expected diff size:** roughly +25 / -16 lines net (+1 import line, replace the 17-line stub with ~25 lines of real logic).

### Scope

Replace `_Sam3ImageAdapter.forward` (currently a `NotImplementedError` stub) with the `forward_grounding`-driven recipe described in spec §4. Add the required `FindStage` import. Do NOT modify `_Sam3ImageAdapter.__init__`, `Sam3Wrapper`, `load_sam31`, or `_resolve_checkpoint_path`.

### Reference

- Spec section: `docs/superpowers/specs/2026-05-17-colab-gpu-integration-fix-design.md` §4 (esp. §4.1, §4.3, §4.5).
- Source citation: `/home/justin/projects/Efficient-SAM3-Finetuning/.venv/lib/python3.13/site-packages/sam3/model/sam3_image_processor.py` lines 31-39 (`FindStage` recipe), 112-125 (`set_text_prompt`), 182-189 (`_forward_grounding` call).
- The Meta `Sam3Image` class: `/home/justin/projects/Efficient-SAM3-Finetuning/.venv/lib/python3.13/site-packages/sam3/model/sam3_image.py` lines 440-496 (`forward_grounding` signature + body), 547-553 (`_get_dummy_prompt`).

### Steps

- [ ] **Step 1: Add the `FindStage` import.**

Insert `from sam3.model.data_misc import FindStage` in the import block. Suggested location: directly under `import sam3` at line 17 of `src/esam3/models/sam3.py`. Use absolute import (matches existing `import sam3` style).

Do NOT add `from sam3.model.geometry_encoders import Prompt` — we use the model's `_get_dummy_prompt` helper instead of constructing `Prompt` directly.

- [ ] **Step 2: Replace the stub body.**

Replace lines 108-124 (the entire `forward` method body of `_Sam3ImageAdapter`) with:

```python
def forward(self, images: Tensor, prompts: list[Prompts]) -> dict[str, Tensor]:
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
    backbone_out = self.model.backbone.forward_image(images)
    text_outputs = self.model.backbone.forward_text([class_names[0]], device=device)
    backbone_out.update(text_outputs)
    b = images.shape[0]
    find_input = FindStage(
        img_ids=torch.arange(b, device=device, dtype=torch.long),
        text_ids=torch.zeros(b, device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    geometric_prompt = self.model._get_dummy_prompt(num_prompts=1)
    outputs: dict[str, Tensor] = self.model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_input,
        find_target=None,
        geometric_prompt=geometric_prompt,
    )
    return outputs
```

Remove the `IMPLEMENTOR:` comment block. Keep the docstring on `_Sam3ImageAdapter` itself (the class) untouched — it remains accurate.

- [ ] **Step 3: Type-hint review.**

The function returns `dict[str, Tensor]` per the existing signature. `forward_grounding` returns a dict that includes non-`Tensor` keys (`aux_outputs: list[dict]`, `prev_encoder_out: dict`) — the existing signature is therefore strictly imprecise, but it matches what `meta_to_canonical` consumes. Keep the existing signature; do not change the return type to `dict[str, Any]` (it would diverge from the spec scope of "minimal change"). If `mypy --strict` complains, add `# type: ignore[assignment]` only on the offending line.

- [ ] **Step 4: Run linter and formatter.**

```bash
ruff check src/esam3/models/sam3.py
ruff format --check src/esam3/models/sam3.py
```

Fix any reported issue in-place.

- [ ] **Step 5: Run unit tests.**

```bash
uv run pytest tests/unit -q
```

Existing 200 unit tests must remain green. None of them invoke `_Sam3ImageAdapter.forward` directly; the change should be unit-test-neutral.

- [ ] **Step 6: Append to `logs/log.md`.**

Add an entry: `[<UTC-ISO8601>] [implementer] task-2: implement _Sam3ImageAdapter.forward via forward_grounding`.

### Definition of Done

- [ ] `src/esam3/models/sam3.py` no longer contains `raise NotImplementedError`.
- [ ] `src/esam3/models/sam3.py` no longer contains `IMPLEMENTOR:` anywhere.
- [ ] `from sam3.model.data_misc import FindStage` is present in the import block.
- [ ] The function body is ≤ 30 lines (excluding the function signature line).
- [ ] `_Sam3ImageAdapter.__init__` is byte-identical to the pre-change version.
- [ ] `Sam3Wrapper`, `load_sam31`, and `_resolve_checkpoint_path` are byte-identical to the pre-change versions.
- [ ] `ruff check` and `ruff format --check` pass on `src/esam3/models/sam3.py`.
- [ ] `uv run pytest tests/unit -q` reports the same number of passing tests as before (200), with no regressions.
- [ ] `logs/log.md` has a Task 2 entry.

### Verification (commands)

```bash
grep -n "NotImplementedError" src/esam3/models/sam3.py        # expect: no matches
grep -n "IMPLEMENTOR" src/esam3/models/sam3.py                  # expect: no matches
grep -n "from sam3.model.data_misc import FindStage" src/esam3/models/sam3.py   # expect: 1 match
ruff check src/esam3/models/sam3.py
ruff format --check src/esam3/models/sam3.py
uv run pytest tests/unit -q
```

### Rollback

```bash
git checkout HEAD -- src/esam3/models/sam3.py
```

Then re-run pre-flight checks before retry.

### Commit

After the Definition of Done is met, commit:

```bash
git add src/esam3/models/sam3.py logs/log.md
git commit -m "$(cat <<'EOF'
fix(models): implement _Sam3ImageAdapter.forward via forward_grounding

Drive sam3.Sam3Image.forward_grounding with a single text prompt per call,
following the sam3.model.sam3_image_processor.Sam3Processor recipe. The
adapter constructs a FindStage with img_ids=[0..B-1], text_ids=[0]*B and a
dummy geometric prompt, then returns the raw output dict (pred_logits,
pred_boxes, pred_masks, presence_logit_dec) unchanged for meta_to_canonical
to consume.

Unblocks tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical.
EOF
)"
```

---

## Task 3: Pin `SCOPE_TARGETS` to real SAM 3.1 module names + align fixtures and unit tests

**Difficulty:** M
**Subagent:** `implementer` (Sonnet/high). Touches three files (prod, fixture, unit tests) with logically-linked renames + one new test.
**Files:**
- Modify: `src/esam3/peft_adapters/lora.py` (lines 32-43 — `SCOPE_TARGETS` dict; remove TODO)
- Modify: `tests/fixtures/tiny_sam3_lora_stub.py` (rename subtrees per spec §5.4)
- Modify: `tests/unit/test_peft_lora.py` (substring renames per spec §6.1 + 1 new test per spec §6.1)
**Expected diff size:** ~+90 / -40 lines combined (most of it the new test in `test_peft_lora.py` plus the fixture rename).

### Scope

Replace `SCOPE_TARGETS` with the patterns in spec §5.1. Rename the dummy LoRA fixture subtrees so the unit tests still exercise the PEFT pipeline end-to-end on a small graph. Add ONE new unit test that asserts the new regexes against an inline tree using REAL SAM 3.1 prefixes (`backbone.vision_backbone.trunk.blocks.*`, `transformer.decoder.layers.*`). Existing 19 LoRA unit tests stay logically the same — only asserted substrings change.

### Reference

- Spec section: `docs/superpowers/specs/2026-05-17-colab-gpu-integration-fix-design.md` §5 and §6.1, §6.2.
- Source citations supporting the patterns:
  - `sam3/model/vitdet.py` lines 433-434 (`self.qkv = nn.Linear`, `self.proj = nn.Linear`)
  - `sam3/model/necks.py` lines 15, 34 (`Sam3DualViTDetNeck.trunk = trunk`)
  - `sam3/model/vl_combiner.py` lines 19, 41 (`SAM3VLBackbone.vision_backbone = ...`)
  - `sam3/model/decoder.py` lines 33-67 (`TransformerDecoderLayer.self_attn`, `.cross_attn`, optional `.ca_text`)
  - `sam3/model/model_misc.py` line 521 (`MultiheadAttentionWrapper.out_proj = NonDynamicallyQuantizableLinear` — subclass of `nn.Linear`)

### Steps

- [ ] **Step 1: Rewrite `tests/fixtures/tiny_sam3_lora_stub.py`.**

Apply the following structural changes (preserve `_AttnBlock`, `_DecoderAttn`, `_DecoderLayer`, `_StubAdapter`, and `make_stub_wrapper` class definitions; only attribute paths change):

| Existing path | New path |
| --- | --- |
| `self.vision_encoder` (a bare `nn.Module`) | `self.vision_trunk` (a bare `nn.Module`) |
| `self.vision_encoder.block0`, `.block1` (attributes) | `self.vision_trunk.blocks = nn.ModuleList([_AttnBlock(dim), _AttnBlock(dim)])` |
| `self.mask_decoder` (a bare `nn.Module`) | `self.transformer_decoder` (a bare `nn.Module`) |
| `self.mask_decoder.layer0` (attribute) | `self.transformer_decoder.layers = nn.ModuleList([_DecoderLayer(dim)])` |
| `self.neg_control_a`, `self.neg_control_b` | unchanged |

Update the module docstring to reflect the new naming shape:

```python
"""Tiny stub mirroring SAM 3.1's attention module-naming shape for LoRA tests.

The subtree paths use indexed `blocks` and `layers` ModuleLists so the same
regex shape that targets the real SAM 3.1 (`...trunk.blocks.\\d+.attn.(qkv|proj)$`,
`...decoder.layers.\\d+.(self_attn|cross_attn).out_proj$`) is exercised here.

forward() raises NotImplementedError — these tests never execute forward.
"""
```

Add fixture-only patterns for unit tests by introducing a module-level helper at the bottom of the file:

```python
# Regex patterns matching the renamed fixture subtrees. The production
# SCOPE_TARGETS in src/esam3/peft_adapters/lora.py target the REAL SAM 3.1
# names (backbone.vision_backbone.trunk.blocks.*); the fixture below uses
# truncated prefixes (`vision_trunk`, `transformer_decoder`) because the full
# nested chain would balloon the fixture without adding coverage.
FIXTURE_SCOPE_PATTERNS: dict[str, list[str]] = {
    "vision":         [r"vision_trunk\.blocks\.\d+\.attn\.(qkv|proj)$"],
    "vision_decoder": [r"vision_trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
                       r"transformer_decoder\.layers\.\d+\.(self_attn|cross_attn)\.out_proj$"],
    "all":            [r".*"],
}
```

This helper is used by the unit tests when they need to assert against the fixture (the production SCOPE_TARGETS would never match anything in the fixture). The unit tests pass `target_modules=FIXTURE_SCOPE_PATTERNS[scope]` to `PEFTConfig` to bypass the scope-key lookup, so the production code path is still exercised.

- [ ] **Step 2: Rewrite `src/esam3/peft_adapters/lora.py:SCOPE_TARGETS`.**

Replace lines 32-43 (the dict definition + the multi-line TODO comment immediately above it) with:

```python
# Real SAM 3.1 attention naming, verified against
# sam3/model/{vitdet.py,necks.py,vl_combiner.py,decoder.py,model_misc.py}.
# `meta_to_canonical` and SCOPE_TARGETS are the two single-points-of-contact
# for SAM 3.1's surface naming; if Meta renames modules, only these change.
SCOPE_TARGETS: dict[str, list[str]] = {
    # ViT vision trunk: fused qkv + output projection per block.
    "vision": [
        r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
    ],
    # Vision trunk + transformer decoder attention output projections.
    # MultiheadAttentionWrapper exposes only `out_proj` as nn.Linear; its
    # in_proj_weight/q,k,v_proj_weight are bare Parameters and not LoRA-targetable.
    "vision_decoder": [
        r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
        r"transformer\.decoder\.layers\.\d+\.(self_attn|cross_attn|ca_text)\.out_proj$",
    ],
    # Every nn.Linear in the tree. Existing intentional over-match; narrowing
    # is deferred (see TODO history in PRs #4 / #7).
    "all": [r".*"],
}
```

Note: the file already imports `re` and `nn`; no new imports needed.

- [ ] **Step 3: Update `tests/unit/test_peft_lora.py`.**

Apply these renames (search-and-replace, but be specific — do NOT global-replace `vision_encoder` to `vision_trunk` blindly; some inline test fixtures use the real-name `backbone.vision_backbone.trunk.blocks.0.attn.qkv` and must NOT be renamed to `vision_trunk`).

Concrete edits, grouped by test function:

1. `test_apply_lora_vision_scope_matches_only_vision`:
   - Override the scope-key path: replace `apply_lora(w, PEFTConfig(method="lora", scope="vision"))` with `apply_lora(w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]))`.
   - Replace `"vision_encoder" in n` -> `"vision_trunk" in n`.
   - Replace `"mask_decoder" in n` -> `"transformer_decoder" in n`.
   - Add the import: `from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper, FIXTURE_SCOPE_PATTERNS`.
2. `test_apply_lora_vision_decoder_scope`:
   - Same override pattern using `FIXTURE_SCOPE_PATTERNS["vision_decoder"]`.
   - Same two substring renames.
3. `test_apply_lora_all_scope_includes_negative_controls`: no change (production `"all"` -> `r".*"` matches everything in the fixture; the test still passes without override).
4. `test_target_modules_overrides_scope`:
   - Replace `target_modules=["vision_encoder.block0.attn.qkv"]` -> `target_modules=["vision_trunk.blocks.0.attn.qkv"]`.
   - Update `qkv_lora` filter substring accordingly.
   - Update `other` filter substring accordingly.
5. `test_apply_lora_no_match_raises`:
   - Replace `assert "vision_encoder" in msg or "neg_control" in msg` -> `assert "vision_trunk" in msg or "neg_control" in msg`.
6. `test_merge_lora_unwraps_and_clears_handle`:
   - Replace `w.model.model.vision_encoder.block0.attn.qkv.weight` (2 occurrences: `pre` and `post`) -> `w.model.model.vision_trunk.blocks[0].attn.qkv.weight`.
   - Replace `"vision_encoder.block0.attn.qkv" in n` -> `"vision_trunk.blocks.0.attn.qkv" in n`.
7. `test_resolve_targets_supports_custom_linear_types`:
   - In the inline `Base` class, replace the entire subtree with the real-name chain so the production `"vision"` regex applies:
     ```python
     class Base(nn.Module):
         def __init__(self) -> None:
             super().__init__()
             self.backbone = nn.Module()
             self.backbone.vision_backbone = nn.Module()  # type: ignore[assignment]
             self.backbone.vision_backbone.trunk = nn.Module()  # type: ignore[assignment]
             self.backbone.vision_backbone.trunk.blocks = nn.ModuleList(  # type: ignore[assignment]
                 [nn.Module()]
             )
             self.backbone.vision_backbone.trunk.blocks[0].attn = nn.Module()  # type: ignore[assignment]
             self.backbone.vision_backbone.trunk.blocks[0].attn.qkv = FakeLinear4bit(8, 24)  # type: ignore[assignment]
             self.backbone.vision_backbone.trunk.blocks[0].attn.proj = FakeLinear4bit(8, 8)  # type: ignore[assignment]
     ```
   - Update the asserted match list to the new full paths:
     ```python
     assert sorted(matched) == [
         "backbone.vision_backbone.trunk.blocks.0.attn.proj",
         "backbone.vision_backbone.trunk.blocks.0.attn.qkv",
     ]
     ```
8. `test_resolve_targets_default_still_filters_to_nn_linear`: same restructure as #7 but with `nn.Linear` instead of `FakeLinear4bit` and the matching asserted paths.
9. `test_scope_targets_keys_match_lora_scope_literal`: no change.
10. `test_save_load_lora_roundtrip`, `test_load_lora_idempotent_guard`, `test_save_lora_without_apply_raises`, `test_merge_lora_without_apply_raises`, `test_apply_lora_registered_under_peft_lora`, `test_apply_lora_default_scope_freezes_base`, `test_apply_lora_idempotent_guard`, `test_apply_lora_trainable_ratio_under_default_scope`, `test_apply_lora_preserves_forward_signature`, `test_apply_lora_sets_peft_model_handle`: no body change beyond the `make_stub_wrapper()` factory continuing to work (it does — fixture rename doesn't touch its signature).

Note on the default-scope tests (`test_apply_lora_default_scope_freezes_base`, `test_apply_lora_idempotent_guard`, `test_apply_lora_trainable_ratio_under_default_scope`, `test_apply_lora_preserves_forward_signature`, `test_apply_lora_sets_peft_model_handle`): these use `PEFTConfig(method="lora")` whose default `scope="vision_decoder"` resolves to the PRODUCTION SCOPE_TARGETS. After the rename, those production patterns will NOT match anything in the renamed fixture (the fixture uses truncated prefixes). To keep these tests working, each must be modified to pass `target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"]` explicitly. Concretely:

- Replace every `apply_lora(w, PEFTConfig(method="lora"))` (no scope override, no target_modules override) with `apply_lora(w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"]))`.

This is the necessary cost of the fixture rename. The new test in Step 4 below covers the production SCOPE_TARGETS path directly.

- [ ] **Step 4: Add the new regression test in `tests/unit/test_peft_lora.py`.**

Append:

```python
def test_scope_targets_match_real_sam3_module_naming() -> None:
    """Regression guard: the production SCOPE_TARGETS regexes match the real
    SAM 3.1 module-naming shape (sourced from sam3/model/{vitdet,necks,vl_combiner,decoder}.py).
    """
    from esam3.peft_adapters.lora import SCOPE_TARGETS, _resolve_targets

    class _RealNamingStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.vision_backbone = nn.Module()  # type: ignore[assignment]
            self.backbone.vision_backbone.trunk = nn.Module()  # type: ignore[assignment]
            blocks: list[nn.Module] = []
            for _ in range(2):
                block = nn.Module()
                block.attn = nn.Module()  # type: ignore[assignment]
                block.attn.qkv = nn.Linear(8, 24)  # type: ignore[assignment]
                block.attn.proj = nn.Linear(8, 8)  # type: ignore[assignment]
                block.mlp = nn.Module()  # type: ignore[assignment]
                block.mlp.fc1 = nn.Linear(8, 16)  # type: ignore[assignment]
                block.mlp.fc2 = nn.Linear(16, 8)  # type: ignore[assignment]
                blocks.append(block)
            self.backbone.vision_backbone.trunk.blocks = nn.ModuleList(blocks)  # type: ignore[assignment]
            self.transformer = nn.Module()
            self.transformer.decoder = nn.Module()  # type: ignore[assignment]
            decoder_layers: list[nn.Module] = []
            for _ in range(2):
                layer = nn.Module()
                for kind in ("self_attn", "cross_attn", "ca_text"):
                    sub = nn.Module()
                    sub.out_proj = nn.Linear(8, 8)  # type: ignore[assignment]
                    setattr(layer, kind, sub)
                layer.linear1 = nn.Linear(8, 16)  # type: ignore[assignment]  # FFN negative control
                decoder_layers.append(layer)
            self.transformer.decoder.layers = nn.ModuleList(decoder_layers)  # type: ignore[assignment]

    stub = _RealNamingStub()

    vision = _resolve_targets(stub, PEFTConfig(method="lora", scope="vision"))
    assert sorted(vision) == [
        "backbone.vision_backbone.trunk.blocks.0.attn.proj",
        "backbone.vision_backbone.trunk.blocks.0.attn.qkv",
        "backbone.vision_backbone.trunk.blocks.1.attn.proj",
        "backbone.vision_backbone.trunk.blocks.1.attn.qkv",
    ]

    vision_decoder = _resolve_targets(
        stub, PEFTConfig(method="lora", scope="vision_decoder")
    )
    assert "transformer.decoder.layers.0.self_attn.out_proj" in vision_decoder
    assert "transformer.decoder.layers.0.cross_attn.out_proj" in vision_decoder
    assert "transformer.decoder.layers.0.ca_text.out_proj" in vision_decoder
    assert "transformer.decoder.layers.1.self_attn.out_proj" in vision_decoder
    # vision scope subset is included.
    assert set(vision).issubset(set(vision_decoder))
    # FFN linears in the decoder are intentionally NOT adapted.
    assert all("linear1" not in n for n in vision_decoder)
    # Vision-trunk MLP is intentionally NOT adapted under vision_decoder.
    assert all(".mlp." not in n for n in vision_decoder)

    # SCOPE_TARGETS still exposes only the three documented scopes.
    assert set(SCOPE_TARGETS) == {"vision", "vision_decoder", "all"}
```

- [ ] **Step 5: Run linter and formatter.**

```bash
ruff check src/esam3/peft_adapters/lora.py tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_lora.py
ruff format --check src/esam3/peft_adapters/lora.py tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_lora.py
```

Fix any reported issue in-place.

- [ ] **Step 6: Run unit tests.**

```bash
uv run pytest tests/unit/test_peft_lora.py -q
uv run pytest tests/unit -q
```

Expected:
- `tests/unit/test_peft_lora.py` reports the existing 19 tests + 1 new test = 20 tests passing.
- `tests/unit` overall reports 201 passing (was 200).

- [ ] **Step 7: Append to `logs/log.md`.**

Entry: `[<UTC-ISO8601>] [implementer] task-3: pin SCOPE_TARGETS to real SAM 3.1 names + align fixture and unit tests`.

### Definition of Done

- [ ] `src/esam3/peft_adapters/lora.py` has the new `SCOPE_TARGETS` with three regex patterns matching real SAM 3.1 naming (`backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$` etc.).
- [ ] The `TODO(task-7)` comment block above the old `"all"` entry is removed (the `"all": [r".*"]` itself stays, with a one-line comment per Step 2).
- [ ] `tests/fixtures/tiny_sam3_lora_stub.py` exposes `FIXTURE_SCOPE_PATTERNS: dict[str, list[str]]` at module scope.
- [ ] `tests/fixtures/tiny_sam3_lora_stub.py` no longer contains the strings `vision_encoder`, `mask_decoder`, `block0`, `block1`, or `layer0` (replaced by `vision_trunk`, `transformer_decoder`, `blocks`, `layers`).
- [ ] `tests/unit/test_peft_lora.py` no longer contains the strings `vision_encoder` or `mask_decoder`.
- [ ] `tests/unit/test_peft_lora.py` defines `test_scope_targets_match_real_sam3_module_naming`.
- [ ] `ruff check` and `ruff format --check` pass on all three files.
- [ ] `uv run pytest tests/unit -q` reports 201 passing tests (200 baseline + 1 new test).

### Verification (commands)

```bash
grep -n "vision_encoder\|mask_decoder" src/esam3/peft_adapters/lora.py tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_lora.py
# Expected: no matches anywhere.

grep -n "FIXTURE_SCOPE_PATTERNS" tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_lora.py
# Expected: 1 definition + at least 2 usages.

grep -n "test_scope_targets_match_real_sam3_module_naming" tests/unit/test_peft_lora.py
# Expected: 1 match.

ruff check src/esam3/peft_adapters/lora.py tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_lora.py
ruff format --check src/esam3/peft_adapters/lora.py tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_lora.py
uv run pytest tests/unit -q
```

### Rollback

```bash
git checkout HEAD -- src/esam3/peft_adapters/lora.py tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_lora.py
```

Then re-run pre-flight checks before retry.

### Commit

After the Definition of Done is met, commit:

```bash
git add src/esam3/peft_adapters/lora.py tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_lora.py logs/log.md
git commit -m "$(cat <<'EOF'
fix(peft): pin SCOPE_TARGETS to real SAM 3.1 module names

Real SAM 3.1's loaded module tree has no `vision_encoder` or `mask_decoder`
substrings. The previous SCOPE_TARGETS regexes matched zero nn.Linear modules
on the released checkpoint, causing apply_lora and apply_qlora to fail with
`ValueError: no nn.Linear modules matched`.

Pin the regexes against the real module structure verified in sam3 source:
  backbone.vision_backbone.trunk.blocks.{i}.attn.{qkv,proj}
  transformer.decoder.layers.{i}.{self_attn,cross_attn,ca_text}.out_proj
The MultiheadAttentionWrapper used by transformer.decoder exposes only
`out_proj` as nn.Linear; its in_proj_weight/q,k,v_proj_weight are bare
Parameters and are unreachable by PEFT LoRA, so the decoder targets are
limited to `out_proj`.

Rename the unit-test fixture subtrees from `vision_encoder`/`mask_decoder`
to `vision_trunk`/`transformer_decoder` so the unit-test regex shape mirrors
the new production patterns; add a new regression test that asserts the
production regexes against an inline tree using the real-name prefixes.
EOF
)"
```

---

## Task 4: Update integration test assertions

**Difficulty:** L
**Subagent:** `implementer-simple` (two files, <20 lines total, no logic — just substring renames in assertions).
**Files:**
- Modify: `tests/integration/test_peft_lora_real.py` (rename two substrings on lines 35-36)
- Modify: `tests/integration/test_peft_qlora_real.py` (rename two substrings on lines 70-71)
**Expected diff size:** +2 / -2 in each file, 4 lines total.

### Scope

Update the two assertions in each integration test that check for `"vision_encoder"` / `"mask_decoder"` substrings in LoRA parameter names. The new SCOPE_TARGETS produces LoRA params under `backbone.vision_backbone.trunk.*.lora_*` and `transformer.decoder.layers.*.lora_*`. The substring assertions must match these real names.

### Steps

- [ ] **Step 1: Edit `tests/integration/test_peft_lora_real.py`.**

In `test_apply_lora_on_real_sam31_under_trainable_budget`:

```python
# OLD:
assert any("vision_encoder" in n for n in lora_names), "no vision-encoder LoRA targets"
assert any("mask_decoder" in n for n in lora_names), "no mask-decoder LoRA targets"

# NEW:
assert any("vision_backbone" in n for n in lora_names), "no vision-trunk LoRA targets"
assert any("transformer.decoder" in n for n in lora_names), "no transformer-decoder LoRA targets"
```

- [ ] **Step 2: Edit `tests/integration/test_peft_qlora_real.py`.**

In `test_apply_qlora_swaps_every_linear_and_attaches_lora`, apply the identical two substring renames as in Step 1.

- [ ] **Step 3: Run linter and formatter.**

```bash
ruff check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
ruff format --check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
```

Fix any reported issue in-place.

- [ ] **Step 4: Run unit tests (sanity).**

```bash
uv run pytest tests/unit -q
```

Expected: still 201 passing. The integration tests are skipped locally (`requires_compatible_gpu`).

- [ ] **Step 5: Append to `logs/log.md`.**

Entry: `[<UTC-ISO8601>] [implementer] task-4: realign integration-test LoRA name substrings`.

### Definition of Done

- [ ] `tests/integration/test_peft_lora_real.py` no longer contains the strings `vision_encoder` or `mask_decoder`.
- [ ] `tests/integration/test_peft_qlora_real.py` no longer contains the strings `vision_encoder` or `mask_decoder`.
- [ ] Both files contain exactly one `vision_backbone` assertion and one `transformer.decoder` assertion in the indicated tests.
- [ ] `ruff check` and `ruff format --check` pass.
- [ ] `uv run pytest tests/unit -q` reports 201 passing.

### Verification (commands)

```bash
grep -n "vision_encoder\|mask_decoder" tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
# Expected: no matches.

grep -n "vision_backbone\|transformer.decoder" tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
# Expected: 4 matches total (2 per file).

ruff check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
ruff format --check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run pytest tests/unit -q
```

### Rollback

```bash
git checkout HEAD -- tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
```

### Commit

After the Definition of Done is met, commit:

```bash
git add tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py logs/log.md
git commit -m "$(cat <<'EOF'
test(integration): realign LoRA name substrings to real SAM 3.1 naming

Match the new SCOPE_TARGETS regexes (Task 3): LoRA params now live under
backbone.vision_backbone.trunk.*.lora_* and transformer.decoder.layers.*.lora_*,
so the substring assertions check for "vision_backbone" and "transformer.decoder"
instead of the obsolete "vision_encoder" / "mask_decoder" placeholders.
EOF
)"
```

---

## Task 5: Push branch and trigger Colab verification

**Difficulty:** L
**Subagent:** Main thread (no subagent needed — the implementer pushes; verification is user-driven on Colab).
**Files:** None modified.

### Scope

Push the branch to the remote and request the user run the Colab notebook end-to-end. This is the only path to verify the integration tier; the dev box's GTX 1080 (compute capability 6.1) skips all `requires_compatible_gpu` tests.

### Steps

- [ ] **Step 1: Verify branch state.**

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip log --oneline -5
# Expected (top-down):
#   <new SHA> test(integration): realign LoRA name substrings to real SAM 3.1 naming
#   <new SHA> fix(peft): pin SCOPE_TARGETS to real SAM 3.1 module names
#   <new SHA> fix(models): implement _Sam3ImageAdapter.forward via forward_grounding
#   517ff6a   fix(models): drop bpe_path override so sam3 uses its bundled gzipped vocab
#   ...

git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip status
# Expected: clean working tree (or only logs/log.md unstaged — commit it now if so).
```

- [ ] **Step 2: Push.**

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip push
```

(No `--force`; this branch is shared with the user's open PR #13.)

- [ ] **Step 3: Notify the user.**

Reply to the user with:
- The exact list of new commit SHAs.
- A one-line instruction: "Open `notebooks/colab_gpu_tests.ipynb` on Colab T4 and re-run end-to-end; the suite should show 9 passing tests."

- [ ] **Step 4: Append to `logs/log.md`.**

Entry: `[<UTC-ISO8601>] [implementer] task-5: pushed; awaiting Colab T4 verification`.

### Definition of Done

- [ ] `git push` exits 0.
- [ ] PR #13 on GitHub shows the three new commits in addition to `517ff6a`.
- [ ] User is notified and given the Colab instruction.

### Rollback

If push is rejected (non-fast-forward or hook failure), STOP. Do not force-push. Investigate the upstream state with `git fetch && git log origin/worktree-fix+colab-bpe-gzip --oneline -5` and ask the user.

---

## Task 6: Address Colab integration-test failures (contingent)

**Difficulty:** Conditional (L if `meta_to_canonical` needs a fallback; M if `forward_grounding` rejects our inputs)
**Subagent:** `implementer` (Sonnet/high) only if invoked; otherwise skipped.

### Scope

ONLY execute if the Colab T4 run reports any failure among the 9 integration tests. Otherwise this plan is done after Task 5.

### Decision tree

- **Failure mode 1: `test_load_sam31_forward_to_canonical` fails on `presence_logit_dec` KeyError.**
  - Root cause: spec §7 risk. Some `Sam3Image` builds set `presence_logit_dec` only conditionally (`sam3_image.py:339-342`).
  - Fix: in `src/esam3/models/matching.py:meta_to_canonical`, default `img_presence` to a zero tensor shaped `(B,)` when `outputs.get("presence_logit_dec") is None`. Out of scope but cheap. Decide with the user before applying.
- **Failure mode 2: `test_load_sam31_forward_to_canonical` fails on a shape assertion (e.g., `pred_masks.shape[-1] != 288`).**
  - Root cause: model build size, not adapter code.
  - Fix: STOP and ask the user; this indicates a build-config drift, not a fix from this plan.
- **Failure mode 3: `test_apply_lora_on_real_sam31_under_trainable_budget` fails because trainable ratio ≥ 5%.**
  - Root cause: spec §7 risk. `vision_decoder` scope is wider than the budget allows on this build.
  - Fix: narrow `SCOPE_TARGETS["vision_decoder"]` second pattern to `r"transformer\.decoder\.layers\.\d+\.(self_attn|cross_attn)\.out_proj$"` (drop `ca_text`). Re-push; re-run Colab.
- **Failure mode 4: `apply_lora` raises `ValueError: no nn.Linear modules matched`.**
  - Root cause: the regex assumptions in §2.3.3 are off for this exact checkpoint.
  - Fix: capture the first-50-Linear-names error message printed by `_resolve_targets`. STOP and reopen the spec with the captured names; do NOT guess.
- **Failure mode 5: any QLoRA-specific failure (`Linear4bit` swap / `bnb` import).**
  - Out of scope for this plan; reopen with a fresh ticket.

### Verification

After any fix in this task, repeat Tasks 4-5's verification steps and re-trigger the Colab notebook.

### Rollback

`git revert <last commit>` only if the fix introduced a regression in the unit suite. Otherwise iterate forward.

---

## Final acceptance

A correct implementation of this plan satisfies:

1. PR #13 contains commits `517ff6a` (untouched) plus 3 new commits (Task 2, 3, 4).
2. `uv run pytest tests/unit -q` reports 201 passing tests, no skips beyond the existing baseline.
3. `ruff check` and `ruff format --check` pass on every touched file.
4. On Colab T4: `bash scripts/run_gpu_tests.sh` reports 9 passing tests under `requires_compatible_gpu and requires_checkpoint`.
5. `logs/log.md` contains at least one append-only entry per task (5 entries total: 1 bootstrap + Tasks 2/3/4/5).
6. No emojis anywhere in the diff.
7. No new dependencies in `pyproject.toml`.
8. No changes to `src/esam3/config/schema.py`, `src/esam3/models/matching.py`, `pyproject.toml`, or any file outside the file map in this plan (with the explicit exception of Task 6 if invoked).
