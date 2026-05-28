"""run_eval builds dataset via registry, loads adapter, calls Evaluator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from custom_sam_peft.eval.runner import run_eval


def _make_cfg(
    format_: str = "coco", peft_method: str = "lora", has_test: bool = False
) -> MagicMock:
    cfg = MagicMock()
    cfg.data.format = format_
    cfg.data.model_dump.return_value = {
        "format": format_,
        "train": {"annotations": "t.json", "images": "t/"},
        "val": {"annotations": "v.json", "images": "v/"},
        "test": ({"annotations": "te.json", "images": "te/"} if has_test else None),
    }
    cfg.data.val = MagicMock()
    cfg.data.val_split = None
    cfg.data.test = MagicMock() if has_test else None
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = peft_method
    cfg.eval.model_copy = lambda update=None: cfg.eval
    return cfg


def test_run_eval_dispatches_qlora_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval with peft_method='qlora' and model=None must dispatch via load_from_disk
    (calling load_qlora) and call _load_channel_adapter, without raising."""
    cfg = _make_cfg(peft_method="qlora")

    qlora_loader_calls: list[tuple[object, object]] = []
    channel_adapter_calls: list[tuple[object, object]] = []

    def fake_load_qlora(wrapper: object, dirpath: object) -> object:
        qlora_loader_calls.append((wrapper, dirpath))
        return wrapper

    def fake_load_channel_adapter(wrapper: object, dirpath: object) -> None:
        channel_adapter_calls.append((wrapper, dirpath))

    monkeypatch.setattr("custom_sam_peft.peft_adapters.qlora.load_qlora", fake_load_qlora)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner._load_channel_adapter", fake_load_channel_adapter
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    fake_report = MagicMock()
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    result = run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    assert result is fake_report
    assert len(qlora_loader_calls) == 1, "load_qlora must be called exactly once"
    assert len(channel_adapter_calls) == 1, "_load_channel_adapter must be called exactly once"
    _, dirpath = qlora_loader_calls[0]
    assert dirpath == tmp_path


def test_run_eval_rejects_test_split_when_data_test_none(tmp_path: Path) -> None:
    cfg = _make_cfg(has_test=False)
    with pytest.raises(ValueError, match=r"data\.test"):
        run_eval(cfg, checkpoint=tmp_path, split="test")


def test_run_eval_lora_calls_load_channel_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval with peft_method='lora' and model=None must call _load_channel_adapter."""
    cfg = _make_cfg(peft_method="lora")
    channel_adapter_calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        "custom_sam_peft.peft_adapters.lora.load_lora",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner._load_channel_adapter",
        lambda wrapper, dirpath: channel_adapter_calls.append((wrapper, dirpath)),
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    fake_report = MagicMock()
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    assert len(channel_adapter_calls) == 1, "_load_channel_adapter must be called exactly once"
    _, dirpath = channel_adapter_calls[0]
    assert dirpath == tmp_path


def test_run_eval_dispatches_dataset_via_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Format 'hf' must reach the @register('dataset', 'hf') factory."""
    cfg = _make_cfg(format_="hf")

    calls: list[tuple[str, str]] = []

    builder_mock = MagicMock(return_value=MagicMock(__len__=lambda self: 0, class_names=[]))

    def fake_lookup(kind: str, name: str) -> object:
        calls.append((kind, name))
        return builder_mock

    monkeypatch.setattr("custom_sam_peft.eval.runner.lookup", fake_lookup)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.peft_adapters.lora.load_lora", lambda *_a, **_kw: None)

    fake_report = MagicMock()
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    result = run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert ("dataset", "hf") in calls
    assert result is fake_report
    # Verify builder was called with the expected shape.
    builder_mock.assert_called_once()
    call_args = builder_mock.call_args
    assert call_args.kwargs.get("pipeline") == "eval"
    assert call_args.kwargs.get("model_name") == "facebook/sam3.1"
    assert isinstance(call_args.args[0], dict)


def test_run_eval_accepts_prebuilt_val_dataset_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If val_dataset/model are provided, runner MUST NOT call lookup('dataset', …)
    or load_sam31."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    forbidden: list[str] = []

    def _forbidden_lookup(kind: str, name: str) -> object:
        forbidden.append(f"{kind}:{name}")
        return lambda *a, **kw: None

    def _forbidden_load(_m: object) -> object:
        forbidden.append("load_sam31")
        return None

    monkeypatch.setattr("custom_sam_peft.eval.runner.lookup", _forbidden_lookup)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", _forbidden_load)
    monkeypatch.setattr("custom_sam_peft.peft_adapters.lora.load_lora", lambda *_a, **_kw: None)

    fake_report = MagicMock(overall={"mAP": 0.5}, per_class={}, n_images=3, n_predictions=3)
    captured: dict[str, object] = {}

    def _fake_evaluator_init(_cfg: object) -> object:
        ev = MagicMock()

        def _evaluate(
            model: object,
            dataset: object,
            *,
            return_per_example_iou: bool = False,
        ) -> object:
            captured["model"] = model
            captured["dataset"] = dataset
            captured["return_per_example_iou"] = return_per_example_iou
            if return_per_example_iou:
                return fake_report, [0.1, 0.5, 0.9]
            return fake_report

        ev.evaluate = _evaluate
        ev.evaluate_and_save = MagicMock(return_value=fake_report)
        return ev

    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", _fake_evaluator_init)

    fake_ds = MagicMock(__len__=lambda self: 3, class_names=["a"])
    fake_model = MagicMock()
    report, per_ex = run_eval(
        cfg,
        checkpoint=tmp_path,
        split="val",
        output_dir=tmp_path,
        val_dataset=fake_ds,
        model=fake_model,
        return_per_example_iou=True,
    )
    assert report is fake_report
    assert per_ex == [0.1, 0.5, 0.9]
    assert captured["dataset"] is fake_ds
    assert captured["model"] is fake_model
    assert captured["return_per_example_iou"] is True
    assert forbidden == []  # neither lookup nor load_sam31 should have been called


def test_run_eval_return_per_example_iou_default_false_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default kwarg path returns MetricsReport (not tuple) — existing CLI contract."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    _empty_ds = MagicMock(__len__=lambda self: 0, class_names=[])
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: _empty_ds,
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.peft_adapters.lora.load_lora", lambda *_a, **_kw: None)

    fake_report = MagicMock(overall={"mAP": 0.0})
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    out = run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert out is fake_report
    assert not isinstance(out, tuple)


# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): --split val guard + auto-split in eval
# ---------------------------------------------------------------------------


def test_run_eval_rejects_val_split_when_data_val_and_val_split_none(
    tmp_path: Path,
) -> None:
    """Spec §7.4 A: --split val requires data.val or data.val_split."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.data.val = None  # type: ignore[attr-defined]
    cfg.data.val_split = None  # type: ignore[attr-defined]
    cfg.data.test = None
    with pytest.raises(ValueError, match=r"--split val requires data\.val"):
        run_eval(cfg, checkpoint=tmp_path, split="val")


def test_run_eval_auto_split_threads_resolved_image_ids_to_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §7.4 B: when val_dataset is None and val_split is set, builder receives
    _resolved_image_ids."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.data.val = None  # type: ignore[attr-defined]
    # Build a real ValSplitConfig so the guard passes.
    from custom_sam_peft.config.schema import ValSplitConfig

    cfg.data.val_split = ValSplitConfig(fraction=0.5, seed=1)  # type: ignore[attr-defined]

    # Mock resolve_val_source to return a known partition.
    from custom_sam_peft.data.val_source import ValSource

    fake_vs = ValSource(
        mode="auto_split",
        train_ids=("1", "2"),
        val_ids=("3", "4"),
        realized_fraction=0.5,
        per_class_counts={0: (2, 2)},
        missing_in_val=(),
        fraction_requested=0.5,
        seed_used=1,
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.resolve_val_source", lambda *_a, **_kw: fake_vs
    )

    captured: dict[str, object] = {}
    fake_ds = MagicMock(__len__=lambda self: 2, class_names=["c"])

    def fake_builder(cfg_dict: object, **kwargs: object) -> object:
        captured["cfg_dict"] = cfg_dict
        return fake_ds

    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: fake_builder,
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.peft_adapters.lora.load_lora", lambda *_a, **_kw: None)
    fake_report = MagicMock(overall={"mAP": 0.5}, per_class={}, n_images=2, n_predictions=2)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert "cfg_dict" in captured
    cfg_dict = captured["cfg_dict"]
    assert isinstance(cfg_dict, dict)
    assert "_resolved_image_ids" in cfg_dict
    assert cfg_dict["_resolved_image_ids"] == {"eval": ["3", "4"]}


# ---------------------------------------------------------------------------
# EvalConfig.batch_size auto-resolution in run_eval (T10)
# ---------------------------------------------------------------------------


def test_run_eval_resolves_auto_via_decide_eval_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval calls presets.decide_eval_batch_size when cfg.eval.batch_size == 'auto'."""
    called: dict[str, object] = {}

    def _fake_decide(classes_per_forward: int = 16) -> tuple[int, int, str]:
        called["k"] = classes_per_forward
        return (3, 1, "analytic")

    # Patch at the import site inside runner.py
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.decide_eval_batch_size",
        _fake_decide,
        raising=False,
    )
    # Also patch the presets module in case runner imports it lazily
    monkeypatch.setattr("custom_sam_peft.presets.decide_eval_batch_size", _fake_decide)

    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.train.batch_size = 8  # higher than decide's return (3) so the cap never fires

    from custom_sam_peft.config.schema import EvalConfig

    cfg.eval = EvalConfig(batch_size="auto")

    evaluator_init_cfg: list[object] = []

    def _fake_evaluator(eval_cfg: object) -> object:
        evaluator_init_cfg.append(eval_cfg)
        ev = MagicMock()
        ev.evaluate_and_save = MagicMock(return_value=MagicMock())
        return ev

    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.peft_adapters.lora.load_lora", lambda *_a, **_kw: None)
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", _fake_evaluator)

    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    assert called.get("k") == 16, f"decide not called with k=16; got {called}"
    # The resolved batch_size must be 3 (what _fake_decide returned).
    assert len(evaluator_init_cfg) == 1
    resolved = evaluator_init_cfg[0]
    assert resolved.batch_size == 3, f"Evaluator got batch_size={resolved.batch_size!r}, want 3"


def test_run_eval_cpu_fallback_logs_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """On CPU, decide_eval_batch_size returns 1; presets logs 'eval.batch_size=auto on CPU'."""
    import logging

    caplog.set_level(logging.INFO)

    # Force CUDA unavailable
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.train.batch_size = 8  # higher than CPU decide's return (1) so the cap never fires

    from custom_sam_peft.config.schema import EvalConfig

    cfg.eval = EvalConfig(batch_size="auto")

    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.peft_adapters.lora.load_lora", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=MagicMock())),
    )

    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    log_messages = " ".join(r.message for r in caplog.records)
    assert "eval.batch_size=auto on CPU" in log_messages, (
        f"Expected 'eval.batch_size=auto on CPU' in logs; got: {log_messages!r}"
    )
