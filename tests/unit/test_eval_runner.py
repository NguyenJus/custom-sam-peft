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
    cfg.data.split = None
    cfg.data.test = MagicMock() if has_test else None
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = peft_method
    cfg.eval.model_copy = lambda update=None: cfg.eval
    cfg.eval.visualize = False
    return cfg


def test_run_eval_dispatches_qlora_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval with a qlora sentinel checkpoint must dispatch load_qlora and
    call _load_channel_adapter (both now happen inside load_adapter)."""
    cfg = _make_cfg(peft_method="qlora")

    # Write the qlora sentinel so discover_method_from_checkpoint returns "qlora".
    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}")
    (tmp_path / "adapter_config.json").write_text("{}")

    qlora_loader_calls: list[tuple[object, object]] = []
    channel_adapter_calls: list[tuple[object, object]] = []

    def fake_load_qlora(wrapper: object, dirpath: object) -> object:
        qlora_loader_calls.append((wrapper, dirpath))
        return wrapper

    def fake_load_channel_adapter(wrapper: object, dirpath: object) -> None:
        channel_adapter_calls.append((wrapper, dirpath))

    # load_adapter in train/checkpoint.py uses module-level imports; patch there.
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_qlora", fake_load_qlora)
    monkeypatch.setattr(
        "custom_sam_peft.train.checkpoint._load_channel_adapter", fake_load_channel_adapter
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


def test_run_eval_rejects_test_split_when_data_test_and_split_test_both_none(
    tmp_path: Path,
) -> None:
    """§10.4 case 1: --split test requires data.test or data.split.test."""
    cfg = _make_cfg(has_test=False)
    cfg.data.split = None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match=r"--split test requires data\.test or data\.split\.test"):
        run_eval(cfg, checkpoint=tmp_path, split="test")


def test_run_eval_lora_calls_load_channel_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval with peft_method='lora' and model=None must call _load_channel_adapter
    (via load_adapter in train/checkpoint.py)."""
    cfg = _make_cfg(peft_method="lora")
    # No qlora sentinel → discover_method_from_checkpoint returns "lora"
    (tmp_path / "adapter_config.json").write_text("{}")
    channel_adapter_calls: list[tuple[object, object]] = []

    # load_adapter in train/checkpoint.py uses module-level imports; patch there.
    monkeypatch.setattr(
        "custom_sam_peft.train.checkpoint.load_lora",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "custom_sam_peft.train.checkpoint._load_channel_adapter",
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
    # load_adapter dispatches load_lora via train.checkpoint's module-level import.
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)

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
    # load_adapter dispatches load_lora via train.checkpoint's module-level import.
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)

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
                return fake_report, [0.1, 0.5, 0.9], None
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
    # load_adapter dispatches load_lora via train.checkpoint's module-level import.
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)

    fake_report = MagicMock(overall={"mAP": 0.0})
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    out = run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert out is fake_report
    assert not isinstance(out, tuple)


# ---------------------------------------------------------------------------
# §10.4: eval runner split-source cases
# ---------------------------------------------------------------------------


def test_run_eval_rejects_val_split_when_data_val_and_split_none(
    tmp_path: Path,
) -> None:
    """§10.4: --split val requires data.val, data.split, or data.hf.split_val."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.data.val = None  # type: ignore[attr-defined]
    cfg.data.split = None  # type: ignore[attr-defined]
    cfg.data.test = None
    with pytest.raises(ValueError, match=r"--split val requires data\.val"):
        run_eval(cfg, checkpoint=tmp_path, split="val")


def test_run_eval_rejects_val_split_when_split_is_test_only(
    tmp_path: Path,
) -> None:
    """§7.1 guard: --split val with a test-only config (data.split={test:0.2}, no val)
    must raise ValueError — not silently evaluate an empty val set.

    Spec §3.4: test-only resolves to mode='none' (no val bucket carved).
    resolve_split_source returns val_ids=() (empty tuple). The guard must check
    that data.split actually carved a val bucket (split.val is not None), not
    just that data.split exists.
    """
    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.data.val = None  # type: ignore[attr-defined]
    cfg.data.test = None  # no explicit test; split.test provides it
    split_mock = MagicMock()
    split_mock.val = None  # test-only: no val carved
    split_mock.test = 0.2
    cfg.data.split = split_mock  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match=r"--split val requires data\.val"):
        run_eval(cfg, checkpoint=tmp_path, split="val")


def test_run_eval_split_test_with_data_split_test_threads_image_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§10.4 case 2: --split test with data.split.test set → builder gets test_ids."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.data.val = None  # type: ignore[attr-defined]
    cfg.data.test = None  # no explicit data.test; use split path
    # data.split.test must be set for the guard and the branch
    split_mock = MagicMock()
    split_mock.test = 0.2
    cfg.data.split = split_mock  # type: ignore[attr-defined]

    from custom_sam_peft.data.split_source import SplitSource

    fake_vs = SplitSource(
        mode="none",
        train_ids=("1", "2"),
        val_ids=(),
        test_ids=("3", "4"),
        realized_fraction=(0.0, 0.2),
        per_class_counts={0: (2, 0, 2)},
        missing_in_val=(),
        missing_in_test=(),
        val_fraction_requested=None,
        test_fraction_requested=0.2,
        seed_used=0,
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.resolve_split_source", lambda *_a, **_kw: fake_vs
    )

    captured: dict[str, object] = {}
    fake_ds = MagicMock(__len__=lambda self: 2, class_names=["c"])

    def fake_builder(cfg_dict: object, **kwargs: object) -> object:
        captured["cfg_dict"] = cfg_dict
        return fake_ds

    monkeypatch.setattr("custom_sam_peft.eval.runner.lookup", lambda kind, name: fake_builder)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)
    fake_report = MagicMock(overall={"mAP": 0.5}, per_class={}, n_images=2, n_predictions=2)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    run_eval(cfg, checkpoint=tmp_path, split="test", output_dir=tmp_path)
    assert "cfg_dict" in captured
    cfg_dict = captured["cfg_dict"]
    assert isinstance(cfg_dict, dict)
    assert "_resolved_image_ids" in cfg_dict
    assert cfg_dict["_resolved_image_ids"] == {"eval": ["3", "4"]}


def test_run_eval_split_test_with_explicit_data_test_uses_existing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§10.4 case 3: --split test with explicit data.test → cfg_dict['val'] = cfg_dict['test']."""
    cfg = _make_cfg(format_="coco", peft_method="lora", has_test=True)
    cfg.data.val = None  # type: ignore[attr-defined]
    cfg.data.split = None  # no auto-split; explicit data.test

    captured: dict[str, object] = {}
    fake_ds = MagicMock(__len__=lambda self: 2, class_names=["c"])

    def fake_builder(cfg_dict: object, **kwargs: object) -> object:
        captured["cfg_dict"] = cfg_dict
        return fake_ds

    monkeypatch.setattr("custom_sam_peft.eval.runner.lookup", lambda kind, name: fake_builder)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)
    fake_report = MagicMock(overall={"mAP": 0.5}, per_class={}, n_images=2, n_predictions=2)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    run_eval(cfg, checkpoint=tmp_path, split="test", output_dir=tmp_path)
    assert "cfg_dict" in captured
    cfg_dict = captured["cfg_dict"]
    assert isinstance(cfg_dict, dict)
    # explicit test path: val key should be set to the test dict
    assert cfg_dict.get("val") == cfg_dict.get("test")
    assert "_resolved_image_ids" not in cfg_dict


def test_run_eval_split_val_with_data_split_threads_val_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§10.4 case 4: --split val with data.split carving a val bucket → builder gets val_ids."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.data.val = None  # type: ignore[attr-defined]
    split_mock = MagicMock()
    split_mock.test = None
    cfg.data.split = split_mock  # type: ignore[attr-defined]

    from custom_sam_peft.data.split_source import SplitSource

    fake_vs = SplitSource(
        mode="auto_split",
        train_ids=("1", "2"),
        val_ids=("3", "4"),
        test_ids=None,
        realized_fraction=(0.5, 0.0),
        per_class_counts={0: (2, 2, 0)},
        missing_in_val=(),
        missing_in_test=(),
        val_fraction_requested=0.5,
        test_fraction_requested=None,
        seed_used=0,
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.resolve_split_source", lambda *_a, **_kw: fake_vs
    )

    captured: dict[str, object] = {}
    fake_ds = MagicMock(__len__=lambda self: 2, class_names=["c"])

    def fake_builder(cfg_dict: object, **kwargs: object) -> object:
        captured["cfg_dict"] = cfg_dict
        return fake_ds

    monkeypatch.setattr("custom_sam_peft.eval.runner.lookup", lambda kind, name: fake_builder)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)
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


def test_run_eval_load_not_recompute_when_split_source_json_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§10.4 case 5: stored split_source.json wins over recompute.

    Monkeypatches stratified_split to raise — asserts it's never called when
    split_source.json is already present.  Mirrors the run_dir derivation:
    checkpoint is at tmp_path/checkpoints/step_1/; run_dir = tmp_path.
    """
    # Set up a run dir with a hand-written split_source.json
    ckpt_dir = tmp_path / "checkpoints" / "step_1"
    ckpt_dir.mkdir(parents=True)
    stored_test_ids = ["stored_a", "stored_b"]
    (tmp_path / "split_source.json").write_text(
        __import__("json").dumps(
            {
                "mode": "none",
                "val_fraction_requested": None,
                "test_fraction_requested": 0.2,
                "seed_used": 0,
                "realized_fraction": [0.0, 0.2],
                "n_train": 8,
                "n_val": 0,
                "n_test": 2,
                "per_class_counts": None,
                "missing_in_val": None,
                "missing_in_test": [],
                "train_ids": ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"],
                "val_ids": [],
                "test_ids": stored_test_ids,
            }
        )
    )

    def _must_not_split(*_a: object, **_kw: object) -> object:
        raise AssertionError("stratified_split must not be called when split_source.json exists")

    monkeypatch.setattr("custom_sam_peft.data.split_source.stratified_split", _must_not_split)

    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.data.val = None  # type: ignore[attr-defined]
    cfg.data.test = None
    split_mock = MagicMock()
    split_mock.test = 0.2
    split_mock.val = None
    split_mock.seed = 0
    cfg.data.split = split_mock  # type: ignore[attr-defined]
    cfg.run = MagicMock()
    cfg.run.seed = 0

    captured: dict[str, object] = {}
    fake_ds = MagicMock(__len__=lambda self: 2, class_names=["c"])

    def fake_builder(cfg_dict: object, **kwargs: object) -> object:
        captured["cfg_dict"] = cfg_dict
        return fake_ds

    monkeypatch.setattr("custom_sam_peft.eval.runner.lookup", lambda kind, name: fake_builder)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)
    fake_report = MagicMock(overall={"mAP": 0.5}, per_class={}, n_images=2, n_predictions=2)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    # checkpoint is at <run_dir>/checkpoints/step_1/; run_dir derived as parent.parent
    run_eval(cfg, checkpoint=ckpt_dir, split="test", output_dir=tmp_path)
    assert "cfg_dict" in captured
    assert captured["cfg_dict"]["_resolved_image_ids"] == {"eval": stored_test_ids}  # type: ignore[index]


def test_run_eval_load_not_recompute_val_branch_when_split_source_json_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§10.4 case 5 (val branch): stored split_source.json wins over recompute for val."""
    ckpt_dir = tmp_path / "checkpoints" / "step_1"
    ckpt_dir.mkdir(parents=True)
    stored_val_ids = ["val_x", "val_y"]
    (tmp_path / "split_source.json").write_text(
        __import__("json").dumps(
            {
                "mode": "auto_split",
                "val_fraction_requested": 0.2,
                "test_fraction_requested": None,
                "seed_used": 0,
                "realized_fraction": [0.2, 0.0],
                "n_train": 8,
                "n_val": 2,
                "n_test": None,
                "per_class_counts": None,
                "missing_in_val": [],
                "missing_in_test": None,
                "train_ids": ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"],
                "val_ids": stored_val_ids,
                "test_ids": None,
            }
        )
    )

    def _must_not_split(*_a: object, **_kw: object) -> object:
        raise AssertionError("stratified_split must not be called when split_source.json exists")

    monkeypatch.setattr("custom_sam_peft.data.split_source.stratified_split", _must_not_split)

    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.data.val = None  # type: ignore[attr-defined]
    cfg.data.test = None
    split_mock = MagicMock()
    split_mock.test = None
    split_mock.val = 0.2
    split_mock.seed = 0
    cfg.data.split = split_mock  # type: ignore[attr-defined]
    cfg.run = MagicMock()
    cfg.run.seed = 0

    captured: dict[str, object] = {}
    fake_ds = MagicMock(__len__=lambda self: 2, class_names=["c"])

    def fake_builder(cfg_dict: object, **kwargs: object) -> object:
        captured["cfg_dict"] = cfg_dict
        return fake_ds

    monkeypatch.setattr("custom_sam_peft.eval.runner.lookup", lambda kind, name: fake_builder)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)
    fake_report = MagicMock(overall={"mAP": 0.5}, per_class={}, n_images=2, n_predictions=2)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    run_eval(cfg, checkpoint=ckpt_dir, split="val", output_dir=tmp_path)
    assert "cfg_dict" in captured
    assert captured["cfg_dict"]["_resolved_image_ids"] == {"eval": stored_val_ids}  # type: ignore[index]


def test_run_eval_recompute_fallback_when_no_split_source_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§10.4 case 6: baseline eval (checkpoint=None) with data.split.test → recomputes."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.data.val = None  # type: ignore[attr-defined]
    cfg.data.test = None
    split_mock = MagicMock()
    split_mock.test = 0.2
    split_mock.val = None
    split_mock.seed = 0
    cfg.data.split = split_mock  # type: ignore[attr-defined]
    cfg.run = MagicMock()
    cfg.run.seed = 0

    from custom_sam_peft.data.split_source import SplitSource

    recomputed = SplitSource(
        mode="none",
        train_ids=("r1", "r2"),
        val_ids=(),
        test_ids=("r3",),
        realized_fraction=(0.0, 0.33),
        per_class_counts=None,
        missing_in_val=None,
        missing_in_test=(),
        val_fraction_requested=None,
        test_fraction_requested=0.2,
        seed_used=0,
    )

    resolve_calls: list[int] = []

    def _fake_resolve(*_a: object, **_kw: object) -> SplitSource:
        resolve_calls.append(1)
        return recomputed

    monkeypatch.setattr("custom_sam_peft.eval.runner.resolve_split_source", _fake_resolve)

    captured: dict[str, object] = {}
    fake_ds = MagicMock(__len__=lambda self: 1, class_names=["c"])

    def fake_builder(cfg_dict: object, **kwargs: object) -> object:
        captured["cfg_dict"] = cfg_dict
        return fake_ds

    monkeypatch.setattr("custom_sam_peft.eval.runner.lookup", lambda kind, name: fake_builder)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    fake_report = MagicMock(overall={"mAP": 0.5}, per_class={}, n_images=1, n_predictions=1)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    # checkpoint=None → no run_dir derived → recompute
    run_eval(cfg, checkpoint=None, split="test", output_dir=tmp_path)
    assert resolve_calls, "resolve_split_source must have been called (recompute path)"
    assert "cfg_dict" in captured
    assert captured["cfg_dict"]["_resolved_image_ids"] == {"eval": ["r3"]}  # type: ignore[index]


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

    cfg.eval = EvalConfig(batch_size="auto", visualize=False)

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
    # load_adapter dispatches load_lora via train.checkpoint's module-level import.
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)
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

    cfg.eval = EvalConfig(batch_size="auto", visualize=False)

    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    # load_adapter dispatches load_lora via train.checkpoint's module-level import.
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=MagicMock())),
    )

    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    log_messages = " ".join(r.message for r in caplog.records)
    assert "eval.batch_size=auto on CPU" in log_messages, (
        f"Expected 'eval.batch_size=auto on CPU' in logs; got: {log_messages!r}"
    )


# ---------------------------------------------------------------------------
# Phase 4: baseline path, sentinel dispatch, advisory warning, output-dir fix
# ---------------------------------------------------------------------------


def test_peft_inferred_lora_overrides_cfg_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lora checkpoint (no qlora sentinel) dispatches load_lora even when cfg says qlora."""
    cfg = _make_cfg(peft_method="qlora")
    # No qlora sentinel → discover_method_from_checkpoint returns "lora"
    (tmp_path / "adapter_config.json").write_text("{}")
    calls: list[str] = []
    # load_adapter in train/checkpoint.py calls module-level load_lora/load_qlora.
    monkeypatch.setattr(
        "custom_sam_peft.train.checkpoint.load_lora", lambda *a, **k: calls.append("lora")
    )
    monkeypatch.setattr(
        "custom_sam_peft.train.checkpoint.load_qlora", lambda *a, **k: calls.append("qlora")
    )
    monkeypatch.setattr(
        "custom_sam_peft.train.checkpoint._load_channel_adapter", lambda *a, **k: None
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: lambda cfg_dict, **kw: MagicMock(),
    )
    ev = MagicMock()
    ev.evaluate_and_save.return_value = MagicMock(overall={})
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert calls == ["lora"]


def test_peft_inferred_qlora_overrides_cfg_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A qlora checkpoint dispatches load_qlora even when cfg says lora."""
    cfg = _make_cfg(peft_method="lora")
    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}")
    (tmp_path / "adapter_config.json").write_text("{}")
    calls: list[str] = []
    # load_adapter in train/checkpoint.py calls module-level load_lora/load_qlora.
    monkeypatch.setattr(
        "custom_sam_peft.train.checkpoint.load_lora", lambda *a, **k: calls.append("lora")
    )
    monkeypatch.setattr(
        "custom_sam_peft.train.checkpoint.load_qlora", lambda *a, **k: calls.append("qlora")
    )
    monkeypatch.setattr(
        "custom_sam_peft.train.checkpoint._load_channel_adapter", lambda *a, **k: None
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: lambda cfg_dict, **kw: MagicMock(),
    )
    ev = MagicMock()
    ev.evaluate_and_save.return_value = MagicMock(overall={})
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert calls == ["qlora"]


def test_peft_mismatch_logs_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """cfg.peft.method='qlora' but checkpoint has no qlora sentinel → WARNING logged."""
    cfg = _make_cfg(peft_method="qlora")  # config says qlora; dir has no sentinel → lora
    (tmp_path / "adapter_config.json").write_text("{}")
    # Monkeypatch load_adapter at the runner's binding so the real dispatch is bypassed.
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_adapter", lambda *a, **k: None)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: lambda cfg_dict, **kw: MagicMock(),
    )
    ev = MagicMock()
    ev.evaluate_and_save.return_value = MagicMock(overall={})
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    with caplog.at_level("WARNING"):
        run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert any("checkpoint" in r.message and "lora" in r.message for r in caplog.records)


def test_checkpoint_none_skips_adapter_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """checkpoint=None (baseline) must skip load_adapter entirely."""
    cfg = _make_cfg()
    load_calls: list[str] = []
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.load_adapter",
        lambda *a, **k: load_calls.append("adapter"),
    )
    sam_calls: list[int] = []
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.load_sam31",
        lambda _m, **_kw: (sam_calls.append(1), MagicMock())[1],
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: lambda cfg_dict, **kw: MagicMock(),
    )
    ev = MagicMock()
    ev.evaluate_and_save.return_value = MagicMock(overall={})
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    run_eval(cfg, checkpoint=None, split="val", output_dir=tmp_path)
    assert load_calls == []  # no adapter load on baseline
    assert sam_calls == [1]  # base model loaded once
    assert ev.evaluate_and_save.called


def test_baseline_output_dir_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """checkpoint=None + output_dir=None falls back to cfg.run.output_dir (no crash)."""
    cfg = _make_cfg()
    cfg.run.output_dir = str(tmp_path / "runs")
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: lambda cfg_dict, **kw: MagicMock(),
    )
    captured: dict[str, object] = {}
    ev = MagicMock()

    def _save(wrapper: object, dataset: object, out: object) -> object:
        captured["out"] = out
        return MagicMock(overall={})

    ev.evaluate_and_save.side_effect = _save
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    run_eval(cfg, checkpoint=None, split="val", output_dir=None)
    assert str(captured["out"]) == str(tmp_path / "runs")  # no NoneType.parent crash


# ---------------------------------------------------------------------------
# Task 8: visualize parameter wiring in run_eval
# ---------------------------------------------------------------------------


def test_run_eval_calls_write_eval_visualizations_when_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """visualize resolves True (from cfg) → write_eval_visualizations is called on
    the default (Branch-2) path, after metrics persist."""
    cfg = _make_cfg()
    cfg.eval.visualize = True
    cfg.eval.visualize_count = 7
    cfg.eval.mask_threshold = 0.0
    cfg.data.normalize = None
    cfg.data.channel_semantics = "rgb"
    cfg.eval.save_predictions = False

    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=["cat"]),
    )
    # Evaluator.evaluate(..., return_per_example_iou=True) -> (report, iou_list, gt_counts)
    ev = MagicMock()
    ev.evaluate.return_value = (
        MagicMock(overall={}, per_class={}, n_images=1, n_predictions=0),
        [0.5],
        [1],
    )
    ev._last_predictions = []
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)

    captured: dict[str, object] = {}

    def _fake_write(model, dataset, out, **kw):
        captured.update(kw)
        captured["out"] = out
        return []

    monkeypatch.setattr("custom_sam_peft.eval.visualize.write_eval_visualizations", _fake_write)

    run_eval(cfg, checkpoint=None, split="val", output_dir=tmp_path)
    assert captured["count"] == 7
    assert "per_example_iou" in captured
    assert captured["per_example_iou"] == [0.5]


def test_run_eval_no_visualize_skips_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """visualize=False overrides cfg → write_eval_visualizations is NOT called and the
    plain evaluate_and_save path is used."""
    cfg = _make_cfg()
    cfg.eval.visualize = True  # cfg says on; flag says off → off wins.
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=["cat"]),
    )
    ev = MagicMock()
    ev.evaluate_and_save.return_value = MagicMock(overall={})
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    called: list[int] = []
    monkeypatch.setattr(
        "custom_sam_peft.eval.visualize.write_eval_visualizations",
        lambda *a, **k: called.append(1),
    )
    run_eval(cfg, checkpoint=None, split="val", output_dir=tmp_path, visualize=False)
    assert called == []
    assert ev.evaluate_and_save.called  # plain path preserved


def test_run_eval_viz_failure_does_not_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A whole-pass viz failure is caught in run_eval (metrics already persisted)."""
    cfg = _make_cfg()
    cfg.eval.visualize = True
    cfg.eval.save_predictions = False
    cfg.data.normalize = None
    cfg.data.channel_semantics = "rgb"
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=["cat"]),
    )
    ev = MagicMock()
    ev.evaluate.return_value = (
        MagicMock(overall={}, per_class={}, n_images=1, n_predictions=0),
        [0.5],
        [1],
    )
    ev._last_predictions = []
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    monkeypatch.setattr(
        "custom_sam_peft.eval.visualize.write_eval_visualizations",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("viz boom")),
    )
    with caplog.at_level("WARNING"):
        run_eval(cfg, checkpoint=None, split="val", output_dir=tmp_path)
    assert (tmp_path / "metrics.json").exists()  # persisted before viz
    assert any("viz" in r.message.lower() or "visuali" in r.message.lower() for r in caplog.records)
