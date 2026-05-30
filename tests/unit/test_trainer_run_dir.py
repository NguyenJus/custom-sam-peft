"""Trainer.fit must use the caller-provided run_dir, not compute one internally."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

import custom_sam_peft.train.trainer as trainer_mod
from custom_sam_peft.config.schema import (
    AugmentationsConfig,
    DataConfig,
    DataSplit,
    LossConfig,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.base import Example, Instance, TextPrompts
from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper


def test_fit_uses_caller_provided_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a minimal Trainer that won't actually train; we just want to see
    # which dir gets the config.yaml write.
    cfg = MagicMock()
    cfg.run.output_dir = str(tmp_path / "ignored")  # Trainer must NOT use this.
    cfg.run.name = "irrelevant"
    cfg.run.seed = 0

    cfg.data.augmentations = AugmentationsConfig(preset="none")
    cfg.train.loss = LossConfig()
    cfg.train.num_workers = 0
    cfg.train.batch_size = 1
    cfg.train.multiplex.classes_per_forward = 1  # OomState reads this (int, not MagicMock).
    cfg.train.epochs = 0  # Skip the train loop entirely.
    cfg.train.warmup_steps = 0
    cfg.train.lr_schedule = "constant"
    cfg.train.learning_rate = 1e-4
    cfg.train.optimizer = "adamw"
    cfg.train.save_every = 1000
    cfg.train.eval_every = 500
    cfg.peft.method = "lora"
    cfg.export.merge = False
    cfg.model_dump.return_value = {"run": {"name": "irrelevant"}}
    # model_copy returns self so that the chained .model_dump() call still uses
    # the configured return_value above.
    cfg.model_copy.return_value = cfg
    cfg.train.model_copy.return_value = cfg.train

    # Stub model with at least one trainable parameter on CPU.
    # Use side_effect so each parameters() call gets a fresh iterator.
    _param = torch.nn.Parameter(torch.zeros(1))
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([_param])

    # Length=1 satisfies DataLoader's RandomSampler; epochs=0 means it's never iterated.
    train_ds = MagicMock(__len__=lambda self: 1, class_names=[])
    val_ds = MagicMock(__len__=lambda self: 0, class_names=[])
    tracker = MagicMock()

    # Patch Evaluator and save_adapter where trainer.py imports them (module-level names).
    monkeypatch.setattr(
        trainer_mod,
        "Evaluator",
        lambda _cfg: MagicMock(
            evaluate=MagicMock(
                return_value=MagicMock(overall={}, per_class={}, n_images=0, n_predictions=0)
            )
        ),
    )
    monkeypatch.setattr(
        trainer_mod, "save_adapter", lambda model, path: path.mkdir(parents=True, exist_ok=True)
    )

    chosen = tmp_path / "explicit-run"
    trainer = Trainer(model, train_ds, val_ds, tracker, cfg)
    result = trainer.fit(run_dir=chosen)

    assert result.run_dir == chosen
    assert (chosen / "config.yaml").exists()
    assert not (tmp_path / "ignored").exists()


class _TinyTextDataset:
    """Two-example dataset with text prompts, suitable for the stub wrapper."""

    def __init__(self) -> None:
        self._examples = [
            Example(
                image=torch.zeros(3, 8, 8),
                image_id=f"img{i}",
                prompts=TextPrompts(classes=["A"]),
                instances=[
                    Instance(
                        mask=torch.zeros(8, 8, dtype=torch.bool),
                        class_id=0,
                        box=torch.tensor([1.0, 1.0, 5.0, 5.0]),
                    )
                ],
            )
            for i in range(2)
        ]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> Example:
        return self._examples[i]

    @property
    def class_names(self) -> list[str]:
        return ["A"]


def test_fit_creates_expected_layout(tmp_path: Path) -> None:
    ds = _TinyTextDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = TrainConfig(
        run=RunConfig(name="layout-test", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
        ),
        peft=PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]),
        train=TrainHyperparams(
            epochs=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
    )
    apply_lora(wrapper, cfg.peft)
    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    run_dir = tmp_path / "layout-test-run"
    result = trainer.fit(run_dir=run_dir)
    rd = result.run_dir
    assert rd == run_dir
    assert rd.exists()
    assert (rd / "config.yaml").exists()
    assert (rd / "adapter" / "adapter_config.json").exists()
    assert (rd / "metrics.json").exists()
    assert (rd / "checkpoints").exists()
    assert isinstance(result, EvalArtifacts)
    assert result.peft_method == "lora"
    payload = json.loads((rd / "metrics.json").read_text())
    assert "global_step" in payload
    assert "overall" in payload


def test_fit_calls_start_run_once_before_first_log(tmp_path: Path) -> None:
    """Regression: Trainer.fit() must call tracker.start_run before any log call."""
    from unittest.mock import MagicMock

    from custom_sam_peft.config.schema import (
        AugmentationsConfig,
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        TextPromptConfig,
        TrainConfig,
        TrainHyperparams,
    )
    from custom_sam_peft.data.coco import COCODataset
    from custom_sam_peft.data.transforms import build_eval_transforms, build_train_transforms
    from custom_sam_peft.peft_adapters.lora import apply_lora
    from custom_sam_peft.train.trainer import Trainer
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

    # Reuse the integration test's tiny_coco directory via the conftest fixture path.
    tiny_coco_dir = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_coco"
    from custom_sam_peft.config.schema import NormalizeConfig

    transforms_t = build_train_transforms(
        AugmentationsConfig(preset="none"),
        32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    transforms_v = build_eval_transforms(
        32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    ds_train = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        transforms=transforms_t,
        text_prompt=TextPromptConfig(),
    )
    ds_val = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        transforms=transforms_v,
        text_prompt=TextPromptConfig(),
    )
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = TrainConfig(
        run=RunConfig(name="sr", output_dir=str(tmp_path), seed=0),
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
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=10_000,
            log_every=10_000,
            warmup_steps=0,
            num_workers=0,
        ),
    )
    apply_lora(wrapper, cfg.peft)

    tracker = MagicMock()
    # Record call order: start_run must precede any log_* call.
    order: list[str] = []
    tracker.start_run.side_effect = lambda *a, **k: order.append("start_run")
    tracker.log_scalars.side_effect = lambda *a, **k: order.append("log_scalars")
    tracker.log_images.side_effect = lambda *a, **k: order.append("log_images")
    tracker.close.side_effect = lambda: order.append("close")

    Trainer(wrapper, ds_train, ds_val, tracker, cfg).fit()
    assert order, "tracker received no calls"
    assert order[0] == "start_run", f"first call was {order[0]!r}, expected start_run"
    assert order[-1] == "close"

    tracker.start_run.assert_called_once()
    args = tracker.start_run.call_args
    # First positional: run_dir (a Path under tmp_path); second: config dict
    assert isinstance(args.args[0], Path)
    assert args.args[0].is_dir()
    assert isinstance(args.args[1], dict)
    # resume_from must be passed through (None for fresh runs)
    assert args.kwargs.get("resume_from", args.args[2] if len(args.args) > 2 else "MISSING") is None


# ---------------------------------------------------------------------------
# Device-placement contract for Trainer._log_image_panel: dataset images must
# reach the model on the model's device. Parallel to the Evaluator regression
# pinned in tests/unit/test_evaluator.py::test_evaluate_moves_image_to_model_device.
# ---------------------------------------------------------------------------


class _PanelDeviceRecordingStub(torch.nn.Module):
    """Records image.device per forward; param lives on a sentinel device."""

    def __init__(self, param_device: str = "meta") -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1, device=param_device))
        self.received_image_devices: list[torch.device] = []

    def forward(self, image: torch.Tensor, prompts: object, support: object = None) -> dict:
        del prompts, support
        self.received_image_devices.append(image.device)
        b = image.shape[0]
        return {
            "pred_logits": torch.zeros(b, 4, 1),
            "pred_boxes": torch.zeros(b, 4, 4),
            "pred_masks": torch.zeros(b, 4, 16, 16),
            "presence_logit_dec": torch.zeros(b, 1),
        }


def test_log_image_panel_moves_image_to_model_device() -> None:
    """Trainer._log_image_panel must move dataset images onto the model's device."""
    stub = _PanelDeviceRecordingStub(param_device="meta")
    fake_self = MagicMock()
    fake_self.model = stub
    fake_self.tracker = NoopTracker()

    example = Example(
        image=torch.zeros(3, 16, 16),
        image_id="img-0",
        prompts=TextPrompts(classes=["cat"]),
        instances=[
            Instance(
                mask=torch.zeros(16, 16, dtype=torch.bool),
                class_id=0,
                box=torch.tensor([0.0, 0.0, 1.0, 1.0]),
            )
        ],
    )

    Trainer._log_image_panel(fake_self, [example], ["cat"], global_step=0)

    assert stub.received_image_devices, "model.forward was never called"
    assert all(d.type == "meta" for d in stub.received_image_devices), (
        f"expected every forward to receive a meta-device image; got "
        f"{[str(d) for d in stub.received_image_devices]}"
    )


def test_run_dir_writes_augmentation_pipeline_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After fit constructs run_dir, augmentation_pipeline.json is present and shaped."""
    from custom_sam_peft.config.schema import (
        AugmentationsConfig,
        DataConfig,
        DataSplit,
        NormalizeConfig,
        PEFTConfig,
        RunConfig,
        TextPromptConfig,
        TrainConfig,
        TrainHyperparams,
    )
    from custom_sam_peft.data.coco import COCODataset
    from custom_sam_peft.data.transforms import build_eval_transforms, build_train_transforms
    from custom_sam_peft.peft_adapters.lora import apply_lora

    tiny_coco_dir = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_coco"
    transforms_t = build_train_transforms(
        AugmentationsConfig(preset="medical", intensity="medium"),
        32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    transforms_v = build_eval_transforms(
        32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    ds_train = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        transforms=transforms_t,
        text_prompt=TextPromptConfig(),
    )
    ds_val = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        transforms=transforms_v,
        text_prompt=TextPromptConfig(),
    )
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = TrainConfig(
        run=RunConfig(name="sidecar", output_dir=str(tmp_path), seed=0),
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
            augmentations=AugmentationsConfig(preset="medical", intensity="medium"),
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
            save_every=10_000,
            log_every=10_000,
            warmup_steps=0,
            num_workers=0,
        ),
    )
    apply_lora(wrapper, cfg.peft)

    run_dir = tmp_path / "sidecar-run"
    trainer = Trainer(wrapper, ds_train, ds_val, NoopTracker(), cfg)
    trainer.fit(run_dir=run_dir)

    sidecar = run_dir / "augmentation_pipeline.json"
    assert sidecar.exists()
    blob = json.loads(sidecar.read_text())
    assert set(blob.keys()) == {"preset", "intensity", "resolved", "steps", "library_version"}
    assert blob["preset"] == "medical"
    assert blob["intensity"] == "medium"
    assert set(blob["resolved"].keys()) == {
        "hflip",
        "vflip",
        "rotate90",
        "rotate_arbitrary",
        "color_jitter",
        "stain_jitter",
        "blur",
        "gauss_noise",
    }
    assert isinstance(blob["library_version"], str) and blob["library_version"]


def test_run_dir_writes_loss_bundle_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §9: trainer writes loss_bundle.json alongside augmentation_pipeline.json."""
    from custom_sam_peft.config.schema import (
        AugmentationsConfig,
        DataConfig,
        DataSplit,
        NormalizeConfig,
        PEFTConfig,
        RunConfig,
        TextPromptConfig,
        TrainConfig,
        TrainHyperparams,
    )
    from custom_sam_peft.data.coco import COCODataset
    from custom_sam_peft.data.transforms import build_eval_transforms, build_train_transforms
    from custom_sam_peft.peft_adapters.lora import apply_lora

    tiny_coco_dir = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_coco"
    transforms_t = build_train_transforms(
        AugmentationsConfig(preset="medical", intensity="medium"),
        32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    transforms_v = build_eval_transforms(
        32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    ds_train = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        transforms=transforms_t,
        text_prompt=TextPromptConfig(),
    )
    ds_val = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        transforms=transforms_v,
        text_prompt=TextPromptConfig(),
    )
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = TrainConfig(
        run=RunConfig(name="loss-sidecar", output_dir=str(tmp_path), seed=0),
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
            augmentations=AugmentationsConfig(preset="medical", intensity="medium"),
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
            save_every=10_000,
            log_every=10_000,
            warmup_steps=0,
            num_workers=0,
        ),
    )
    apply_lora(wrapper, cfg.peft)

    run_dir = tmp_path / "loss-sidecar-run"
    trainer = Trainer(wrapper, ds_train, ds_val, NoopTracker(), cfg)
    trainer.fit(run_dir=run_dir)

    loss_path = run_dir / "loss_bundle.json"
    assert loss_path.exists(), list(tmp_path.rglob("*"))
    d = json.loads(loss_path.read_text())
    assert set(d.keys()) == {
        "preset",
        "class_imbalance",
        "resolved",
        "term_classes",
        "library_version",
    }
    assert len(d["resolved"]) == 13
    assert set(d["term_classes"].keys()) == {"mask", "box", "obj", "presence"}
