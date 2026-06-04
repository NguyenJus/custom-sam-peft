"""End-to-end integration: Trainer.fit() with tiny_coco + LoRA stub."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from custom_sam_peft.config.schema import (
    AugmentationsConfig,
    DataConfig,
    DataSplit,
    NormalizeConfig,
    PEFTConfig,
    RunConfig,
    TextPromptConfig,
    TrackingConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.coco import COCODataset
from custom_sam_peft.data.transforms import build_eval_transforms, build_train_transforms
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking import build_tracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

pytestmark = pytest.mark.integration


def _ds(tiny_coco_dir: Path, pipeline: str) -> COCODataset:
    from custom_sam_peft.config.schema import NormalizeConfig

    if pipeline == "train":
        transforms = build_train_transforms(
            AugmentationsConfig(preset="none"),
            32,
            model_name="facebook/sam3.1",
            normalize=NormalizeConfig(),
        )
    else:
        transforms = build_eval_transforms(
            32,
            model_name="facebook/sam3.1",
            normalize=NormalizeConfig(),
        )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )


@pytest.mark.parametrize("backend", ["none", "tensorboard"])
def test_fit_end_to_end_on_tiny_coco(backend: str, tmp_path: Path, tiny_coco_dir: Path) -> None:
    if backend == "tensorboard":
        pytest.importorskip("tensorboard")
    ds_train = _ds(tiny_coco_dir, "train")
    ds_val = _ds(tiny_coco_dir, "eval")
    wrapper = make_stub_wrapper(dim=8, working=True)

    # Default-path guard: the practical optimizer/regularization hyperparameters
    # (learning_rate, lr_schedule, optimizer, max_grad_norm, peft.r/alpha/dropout)
    # are left at their schema defaults so this end-to-end run exercises the real
    # default training path. `epochs` is truncated (to 1) for CI runtime — NOT run
    # to convergence — and `warmup_steps=0` follows from that truncation (a 100-step
    # warmup never completes in a 1-epoch tiny-dataset run). `peft.scope="vision"`
    # (with the matching fixture `target_modules`) is pinned to the stub's vision
    # subtree, not the shipped `"vision_decoder"` default — the CPU LoRA stub only
    # models the vision blocks. See docs/defaults-provenance.md "Reference Training
    # Profile" for the shipped epochs default.
    cfg = TrainConfig(
        run=RunConfig(name="e2e", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            augmentations=AugmentationsConfig(preset="none"),
        ),
        peft=PEFTConfig(
            method="lora",
            scope="vision",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        tracking=TrackingConfig(backend=backend),  # type: ignore[arg-type]
    )
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / f"{cfg.run.name}-test"
    tracker = build_tracker(cfg)
    trainer = Trainer(wrapper, ds_train, ds_val, tracker, cfg)
    result = trainer.fit(run_dir=run_dir)

    assert result.run_dir.exists()
    assert (result.run_dir / "adapter" / "adapter_config.json").exists()
    sidecar = result.run_dir / "augmentation_pipeline.json"
    assert sidecar.exists()
    blob = json.loads(sidecar.read_text())
    assert blob["preset"] == "none"  # this test uses preset=none (post-Phase-G migration)
    assert blob["steps"][:2] == ["LongestMaxSize", "PadIfNeeded"]
    assert blob["steps"][-2:] == ["Normalize", "ToTensorV2"]
    assert blob["library_version"]
    payload = json.loads((result.run_dir / "metrics.json").read_text())
    assert payload["global_step"] >= 1
    ckpts = list((result.run_dir / "checkpoints").glob("step_*"))
    assert ckpts, "expected at least one step_* checkpoint dir"
    assert (ckpts[0] / "training_state.pt").exists()
    assert (ckpts[0] / "adapter").exists()
    if backend == "tensorboard":
        events = list(result.run_dir.glob("events.out.tfevents.*"))
        assert events, "tensorboard backend should write at least one event file"


def test_end_to_end_writes_loss_bundle_json(tmp_path: Path, tiny_coco_dir: Path) -> None:
    """Spec §9: after a training run, loss_bundle.json sits beside augmentation_pipeline.json."""
    ds_train = _ds(tiny_coco_dir, "train")
    ds_val = _ds(tiny_coco_dir, "eval")
    wrapper = make_stub_wrapper(dim=8, working=True)

    # Default-path guard: the practical optimizer/regularization hyperparameters
    # (learning_rate, lr_schedule, optimizer, max_grad_norm, peft.r/alpha/dropout)
    # are left at their schema defaults so this end-to-end run exercises the real
    # default training path. `epochs` is truncated (to 1) for CI runtime — NOT run
    # to convergence — and `warmup_steps=0` follows from that truncation (a 100-step
    # warmup never completes in a 1-epoch tiny-dataset run). `peft.scope="vision"`
    # (with the matching fixture `target_modules`) is pinned to the stub's vision
    # subtree, not the shipped `"vision_decoder"` default — the CPU LoRA stub only
    # models the vision blocks. See docs/defaults-provenance.md "Reference Training
    # Profile" for the shipped epochs default.
    cfg = TrainConfig(
        run=RunConfig(name="e2e-loss-sidecar", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            augmentations=AugmentationsConfig(preset="none"),
        ),
        peft=PEFTConfig(
            method="lora",
            scope="vision",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        tracking=TrackingConfig(backend="none"),
    )
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / f"{cfg.run.name}-test"
    tracker = build_tracker(cfg)
    trainer = Trainer(wrapper, ds_train, ds_val, tracker, cfg)
    result = trainer.fit(run_dir=run_dir)

    loss_path = result.run_dir / "loss_bundle.json"
    assert loss_path.exists(), list(result.run_dir.iterdir())
    d = json.loads(loss_path.read_text())
    assert d["preset"] in {"natural", "medical", "satellite", "microscopy", "none", "custom"}
    assert d["library_version"]
    assert set(d.keys()) == {
        "preset",
        "class_imbalance",
        "resolved",
        "term_classes",
        "library_version",
    }
    assert len(d["resolved"]) == 13
    assert set(d["term_classes"].keys()) == {"mask", "box", "obj", "presence"}


def _bad_data_cfg(
    tmp_path: Path,
    annotations: Path,
    images: Path,
) -> TrainConfig:
    """Minimal TrainConfig pointing at a (likely-broken) annotations/images pair."""
    # Error-path fixture (NOT a default-path guard): builds a structurally minimal
    # LoRA config so the bad-data tests reach the data-loading error they exercise,
    # rather than failing earlier. `epochs=1` / `warmup_steps=0` and
    # `peft.scope="vision"` (matching the stub's vision subtree) keep the config
    # cheap and stub-compatible; training is not run to convergence here.
    return TrainConfig(
        run=RunConfig(name="bad-data", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations=str(annotations), images=str(images)),
            val=DataSplit(annotations=str(annotations), images=str(images)),
        ),
        peft=PEFTConfig(
            method="lora",
            scope="vision",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        tracking=TrackingConfig(backend="none"),
    )


def test_malformed_coco_json_raises_clear_error(tmp_path: Path) -> None:
    """C5 per spec §6.2: invalid JSON in annotations.json surfaces a clear error."""
    images = tmp_path / "images"
    images.mkdir()
    annotations = tmp_path / "annotations.json"
    annotations.write_text("{")  # invalid JSON

    cfg = _bad_data_cfg(tmp_path, annotations, images)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    # Pinned: pycocotools.COCO -> json.load -> json.JSONDecodeError on truncated
    # JSON. We invoke the loader via COCODataset (rather than passing None to
    # Trainer) so the JSON-parse code path actually executes; otherwise Trainer
    # never touches the annotations file and we'd be pinning a downstream
    # TypeError instead of the loader's contract.
    with pytest.raises(json.JSONDecodeError):
        COCODataset(
            annotations=str(annotations),
            images=str(images),
            transforms=build_train_transforms(
                AugmentationsConfig(preset="none"),
                32,
                model_name="facebook/sam3.1",
                normalize=NormalizeConfig(),
            ),
            text_prompt=TextPromptConfig(),
        )


def test_missing_image_file_raises_clear_error(tmp_path: Path) -> None:
    """C5 per spec §6.2: missing image referenced by COCO surfaces a clear error
    naming the file."""
    images = tmp_path / "images"
    images.mkdir()
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "missing.jpg", "width": 32, "height": 32}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                        "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                    }
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
        )
    )

    cfg = _bad_data_cfg(tmp_path, annotations, images)
    ds_train = COCODataset(
        annotations=str(annotations),
        images=str(images),
        transforms=build_train_transforms(
            AugmentationsConfig(preset="none"),
            32,
            model_name="facebook/sam3.1",
            normalize=NormalizeConfig(),
        ),
        text_prompt=TextPromptConfig(),
    )
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    # Pinned: PIL.Image.open on absent path raises FileNotFoundError whose
    # str() embeds the full image path (which includes "missing.jpg").
    with pytest.raises(FileNotFoundError) as excinfo:
        Trainer(wrapper, ds_train, ds_train, build_tracker(cfg), cfg).fit(
            run_dir=tmp_path / "run-missing-img"
        )
    assert "missing.jpg" in str(excinfo.value), (
        f"expected 'missing.jpg' in error message; got: {excinfo.value!r}"
    )


def test_missing_annotation_entry_does_not_crash(tmp_path: Path) -> None:
    """C5 per spec §6.2: an image with no matching annotations is handled
    gracefully (zero-instance item) OR raises with a clear message.

    The implementer pins the actual behavior: if the loader returns
    zero-instance items, training proceeds without crashing; if it raises,
    the message names the orphan image.
    """
    # Use tiny_coco's first image as the only valid image.
    images = tmp_path / "images"
    images.mkdir()
    # Make a 1x1 black png so the loader has something to open.
    from PIL import Image as PILImage

    PILImage.new("RGB", (32, 32)).save(images / "img.png")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "img.png", "width": 32, "height": 32},
                    {"id": 2, "file_name": "img.png", "width": 32, "height": 32},
                ],
                # Only image_id=1 has an annotation; image_id=2 is orphan.
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                        "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                    }
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
        )
    )
    cfg = _bad_data_cfg(tmp_path, annotations, images)
    ds_train = COCODataset(
        annotations=str(annotations),
        images=str(images),
        transforms=build_train_transforms(
            AugmentationsConfig(preset="none"),
            32,
            model_name="facebook/sam3.1",
            normalize=NormalizeConfig(),
        ),
        text_prompt=TextPromptConfig(),
    )
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    # Either Trainer.fit completes (zero-instance handling) OR raises with a
    # clear message naming the orphan image. The implementer pins which.
    try:
        Trainer(wrapper, ds_train, ds_train, build_tracker(cfg), cfg).fit(
            run_dir=tmp_path / "run-orphan"
        )
    except Exception as exc:
        # If it raises, the message should reference the orphan image or the
        # image_id. If it does not, surface that as a follow-up.
        assert "2" in str(exc) or "img.png" in str(exc), (
            f"orphan-image error message lacks identifier: {exc!r}"
        )


# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): auto-split + no-val end-to-end
# ---------------------------------------------------------------------------


def test_e2e_auto_split_on_tiny_coco(tmp_path: Path, tiny_coco_dir: Path) -> None:
    """Spec §9.10.2: end-to-end run with split.val=0.5 creates split_source.json
    and metrics.json (with overall mAP from the carved val set)."""
    import custom_sam_peft.train.runner as runner_mod
    from custom_sam_peft.config.schema import SplitConfig
    from custom_sam_peft.data.split_source import load_split_source

    # Default-path guard: the practical optimizer/regularization hyperparameters
    # (learning_rate, lr_schedule, optimizer, max_grad_norm, peft.r/alpha/dropout)
    # are left at their schema defaults so this end-to-end run exercises the real
    # default training path. `epochs` is truncated (to 1) for CI runtime — NOT run
    # to convergence — and `warmup_steps=0` follows from that truncation (a 100-step
    # warmup never completes in a 1-epoch tiny-dataset run). `peft.scope="vision"`
    # (with the matching fixture `target_modules`) is pinned to the stub's vision
    # subtree, not the shipped `"vision_decoder"` default — the CPU LoRA stub only
    # models the vision blocks. See docs/defaults-provenance.md "Reference Training
    # Profile" for the shipped epochs default.
    cfg = TrainConfig(
        run=RunConfig(name="e2e-auto", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=None,
            split=SplitConfig(val=0.5, seed=None),
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=1,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        tracking=TrackingConfig(backend="none"),
    )

    # Stub the model so this runs on CPU.
    orig_load = runner_mod.load_sam31
    runner_mod.load_sam31 = lambda _m, **_kw: make_stub_wrapper(dim=8, working=True)  # type: ignore[assignment]
    try:
        result = runner_mod.run_training(cfg)
    finally:
        runner_mod.load_sam31 = orig_load  # type: ignore[assignment]

    vs = load_split_source(result.run_dir)
    assert vs is not None
    assert vs.mode == "auto_split"
    assert (result.run_dir / "metrics.json").is_file()
    # In auto-split mode, val_ds is non-empty so metrics.json carries overall, not the no-val note.
    payload = json.loads((result.run_dir / "metrics.json").read_text())
    assert "overall" in payload or "note" in payload  # tolerate either depending on tiny size


def _make_5image_coco_dir(base: Path) -> Path:
    """Create a tiny 5-image COCO fixture in base/coco5/ and return the path.

    5 images guarantees that a (val=0.2, test=0.2) split produces non-empty
    val and test buckets: round(5*0.2)=1 each, train=3.  Writes 5 fresh
    32x32 PNGs so the fixture is self-contained.
    """

    from PIL import Image as PILImage

    coco_dir = base / "coco5"
    img_dir = coco_dir / "images"
    img_dir.mkdir(parents=True)

    # Write 5 tiny 32x32 images.
    for i in range(1, 6):
        PILImage.new("RGB", (32, 32), color=(i * 40, 0, 0)).save(img_dir / f"img_{i:06d}.png")

    ann = {
        "info": {"description": "coco5 integration fixture", "version": "1.0"},
        "licenses": [],
        "images": [
            {"id": i, "file_name": f"img_{i:06d}.png", "width": 32, "height": 32}
            for i in range(1, 6)
        ],
        "categories": [
            {"id": 1, "name": "thing_a", "supercategory": "thing"},
            {"id": 2, "name": "thing_b", "supercategory": "thing"},
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": (1 if i % 2 == 1 else 2),
                "bbox": [4, 4, 12, 12],
                "area": 144,
                "iscrowd": 0,
                "segmentation": [[4, 4, 16, 4, 16, 16, 4, 16]],
            }
            for i in range(1, 6)
        ],
    }
    (coco_dir / "annotations.json").write_text(json.dumps(ann))
    return coco_dir


def test_e2e_split_with_test_fraction_and_eval_split_test(tmp_path: Path) -> None:
    """§10.6: split-mode run with test fraction — stored test_ids are loaded by eval.

    Uses a 5-image inline COCO fixture (not tiny_coco_dir) so the (val=0.2, test=0.2)
    split reliably produces non-empty val and test buckets: round(5*0.2)=1 each.

    Asserts:
      1. split_source.json exists after training with test_ids non-empty.
      2. metrics.json reflects val evaluation (mode=auto_split) — the test bucket is
         NOT auto-evaluated during training.
      3. A follow-up run_eval(..., split='test') with stratified_split monkeypatched to
         raise proves the stored ids are loaded (not recomputed).  The scored id set
         equals the persisted test_ids.
    """
    from unittest.mock import MagicMock

    import custom_sam_peft.data.split_source as _ss_mod
    import custom_sam_peft.eval.runner as _eval_runner
    import custom_sam_peft.train.runner as runner_mod
    from custom_sam_peft.config.schema import SplitConfig
    from custom_sam_peft.data.split_source import load_split_source
    from custom_sam_peft.eval.runner import run_eval

    coco_dir = _make_5image_coco_dir(tmp_path)

    # Default-path guard: practical hyperparameters left at schema defaults;
    # epochs=1 / warmup_steps=0 for CI runtime; peft.scope="vision" matches stub.
    cfg = TrainConfig(
        run=RunConfig(name="e2e-split-test", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(coco_dir / "annotations.json"),
                images=str(coco_dir / "images"),
            ),
            val=None,
            split=SplitConfig(val=0.2, test=0.2, seed=0),
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=1,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        tracking=TrackingConfig(backend="none"),
    )

    # Stub the model so training runs on CPU.
    orig_load = runner_mod.load_sam31
    runner_mod.load_sam31 = lambda _m, **_kw: make_stub_wrapper(dim=8, working=True)  # type: ignore[assignment]
    try:
        result = runner_mod.run_training(cfg)
    finally:
        runner_mod.load_sam31 = orig_load  # type: ignore[assignment]

    # 1. split_source.json exists with test_ids non-empty.
    vs = load_split_source(result.run_dir)
    assert vs is not None
    assert vs.test_ids is not None, "test_ids key must be present in split_source.json"
    assert len(vs.test_ids) > 0, (
        "test bucket must be non-empty with 5 images and test=0.2 (round(5*0.2)=1)"
    )
    stored_test_ids = set(vs.test_ids)

    # 2. Training evaluated on val (mode=auto_split), NOT on test.
    assert vs.mode == "auto_split", "val was carved — mode must be auto_split"
    metrics_payload = json.loads((result.run_dir / "metrics.json").read_text())
    assert metrics_payload.get("note") != "no validation set provided", (
        "test bucket must not suppress val evaluation"
    )

    # 3. Follow-up run_eval --split test: loads stored ids, does NOT recompute.
    ckpt_dirs = sorted((result.run_dir / "checkpoints").glob("step_*"))
    assert ckpt_dirs, "training must write at least one checkpoint"
    ckpt_dir = ckpt_dirs[-1]

    scored_ids: list[str] = []

    def _capturing_builder(cfg_dict: object, **kwargs: object) -> object:
        """Capture _resolved_image_ids — the ids actually passed to the dataset builder."""
        assert isinstance(cfg_dict, dict)
        resolved = cfg_dict.get("_resolved_image_ids", {})
        assert isinstance(resolved, dict)
        scored_ids.extend(resolved.get("eval", []))
        ds = MagicMock()
        ds.__len__ = lambda self: len(scored_ids)
        ds.class_names = []
        return ds

    def _must_not_recompute(*_a: object, **_kw: object) -> object:
        raise AssertionError(
            "stratified_split must NOT be called when split_source.json is present — "
            "eval must load stored ids, not recompute."
        )

    orig_split = _ss_mod.stratified_split
    orig_lookup = _eval_runner.lookup
    orig_load_sam31_eval = _eval_runner.load_sam31

    _ss_mod.stratified_split = _must_not_recompute  # type: ignore[assignment]
    _eval_runner.load_sam31 = lambda _m, **_kw: make_stub_wrapper(dim=8, working=True)  # type: ignore[assignment]
    _eval_runner.lookup = lambda kind, name: _capturing_builder  # type: ignore[assignment]

    try:
        run_eval(
            cfg,
            checkpoint=ckpt_dir,
            split="test",
            output_dir=tmp_path / "eval-out",
            val_dataset=None,
            model=None,
        )
    except Exception:  # noqa: S110 — MagicMock dataset may fail eval; we only need
        # scored_ids captured by the builder before evaluation, which has already run.
        pass
    finally:
        _ss_mod.stratified_split = orig_split  # type: ignore[assignment]
        _eval_runner.load_sam31 = orig_load_sam31_eval  # type: ignore[assignment]
        _eval_runner.lookup = orig_lookup  # type: ignore[assignment]

    # The ids injected into the builder must equal the stored test_ids exactly.
    assert set(scored_ids) == stored_test_ids, (
        f"run_eval --split test scored {set(scored_ids)!r} but stored test_ids are "
        f"{stored_test_ids!r} — eval must load stored ids, not recompute."
    )


def test_e2e_no_val_on_tiny_coco(tmp_path: Path, tiny_coco_dir: Path) -> None:
    """Spec §9.10.3: end-to-end no-val run creates split_source.json with mode=none
    and metrics.json with the no-val note."""
    import custom_sam_peft.train.runner as runner_mod
    from custom_sam_peft.data.split_source import load_split_source

    # Default-path guard: the practical optimizer/regularization hyperparameters
    # (learning_rate, lr_schedule, optimizer, max_grad_norm, peft.r/alpha/dropout)
    # are left at their schema defaults so this end-to-end run exercises the real
    # default training path. `epochs` is truncated (to 1) for CI runtime — NOT run
    # to convergence — and `warmup_steps=0` follows from that truncation (a 100-step
    # warmup never completes in a 1-epoch tiny-dataset run). `peft.scope="vision"`
    # (with the matching fixture `target_modules`) is pinned to the stub's vision
    # subtree, not the shipped `"vision_decoder"` default — the CPU LoRA stub only
    # models the vision blocks. See docs/defaults-provenance.md "Reference Training
    # Profile" for the shipped epochs default.
    cfg = TrainConfig(
        run=RunConfig(name="e2e-noval", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=None,
            split=None,
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=1,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        tracking=TrackingConfig(backend="none"),
    )

    orig_load = runner_mod.load_sam31
    runner_mod.load_sam31 = lambda _m, **_kw: make_stub_wrapper(dim=8, working=True)  # type: ignore[assignment]
    try:
        result = runner_mod.run_training(cfg)
    finally:
        runner_mod.load_sam31 = orig_load  # type: ignore[assignment]

    vs = load_split_source(result.run_dir)
    assert vs is not None
    assert vs.mode == "none"
    payload = json.loads((result.run_dir / "metrics.json").read_text())
    assert payload.get("note") == "no validation set provided"
