"""Semantic predict: label-map writer + colorized PNG + runner branch + CLI prompt-defaulting.

All tests are CPU-only; load_sam31 is monkeypatched where model loading would occur.
No GPU forward passes are performed.
"""

from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app
from custom_sam_peft.predict.runner import PredictOptions, PredictReport
from custom_sam_peft.predict.writers import write_semantic_label_map

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

_cli_runner = CliRunner()


def _make_image_dir(tmp_path: Path, n: int = 1) -> Path:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(n):
        Image.new("RGB", (32, 32), color=(i * 30, 100, 200)).save(img_dir / f"img_{i:03d}.png")
    return img_dir


def _make_semantic_opts(
    tmp_path: Path,
    *,
    config: Path | None = None,
    prompts: str = "road,tree",
    n_images: int = 1,
) -> PredictOptions:
    img_dir = _make_image_dir(tmp_path, n=n_images)
    return PredictOptions(
        images=img_dir,
        prompts=prompts,
        output=tmp_path / "out",
        checkpoint=None,
        merge_adapter=False,
        config=config,
        score_threshold=0.0,
        top_k=100,
        save_masks="none",
        visualize=False,
        device="cpu",
        dtype="float32",
        seed=0,
        dry_run=False,
        verbose=False,
        batch_size=1,
    )


# ---------------------------------------------------------------------------
# Semantic stub nn.Module
# (mirrors the instance stub in tests/predict/test_runner_smoke.py)
# The semantic path calls marginalize_group on pred_logits/pred_masks/presence_logit_dec,
# so we return the same keys the instance stub does.
# ---------------------------------------------------------------------------

H_LOW = 8  # low-res mask spatial size used by the stub
W_LOW = 8


class _SemanticStubModule(torch.nn.Module):
    """Stub whose forward returns instance-path keys; semantic path marginalizes them."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_call_count = 0

    def forward(
        self,
        images: torch.Tensor,
        prompts: list[Any],
        support: Any = None,
    ) -> dict[str, torch.Tensor]:
        self.forward_call_count += 1
        b = images.shape[0]
        from custom_sam_peft.data.base import TextPrompts as _TP

        k_g = len(prompts[0].classes) if prompts and isinstance(prompts[0], _TP) else 1
        total = b * k_g
        # High positive logits → sigmoid near 1 → strong foreground signal after marginalize
        return {
            "pred_logits": torch.full((total, 4, 1), 5.0),
            "pred_boxes": torch.full((total, 4, 4), 0.5),
            "pred_masks": torch.full((total, 4, H_LOW, W_LOW), 2.0),
            "presence_logit_dec": torch.full((total, 1), 5.0),
        }


def _patch_load_semantic(stub: torch.nn.Module) -> mock.MagicMock:
    def _factory(cfg: Any, **kwargs: Any) -> torch.nn.Module:
        return stub

    return mock.patch("custom_sam_peft.models.sam3.load_sam31", side_effect=_factory)


def _make_semantic_config_yaml(tmp_path: Path, class_map_path: Path) -> Path:
    """Write a minimal semantic YAML config and return its path."""
    config_path = tmp_path / "semantic_config.yaml"
    config_path.write_text(
        f"""\
task: semantic
data:
  format: mask_png
  train:
    root: /tmp/fake
  semantic:
    class_map: {class_map_path}
    ignore_index: 255
""",
        encoding="utf-8",
    )
    return config_path


def _make_class_map(tmp_path: Path, mapping: dict[str, str] | None = None) -> Path:
    if mapping is None:
        mapping = {"0": "background", "1": "road", "2": "tree"}
    class_map_path = tmp_path / "class_map.json"
    class_map_path.write_text(json.dumps(mapping), encoding="utf-8")
    return class_map_path


# ===========================================================================
# Part 1: write_semantic_label_map
# ===========================================================================


class TestWriteSemanticLabelMap:
    """Unit tests for predict/writers.write_semantic_label_map."""

    def test_emits_index_and_colorized(self, tmp_path: Path) -> None:
        """Provided by the task spec verbatim."""
        label_map = torch.tensor([[0, 1], [2, 0]], dtype=torch.int64)
        paths = write_semantic_label_map(
            label_map, image_id="a", out_dir=tmp_path, class_names=["road", "tree"]
        )
        idx = np.array(Image.open(paths["index_path"]))
        assert idx.dtype in (np.uint8, np.uint16)
        assert set(np.unique(idx).tolist()) <= {0, 1, 2}
        col = Image.open(paths["colorized_path"])
        assert col.mode in ("RGB", "RGBA")
        col_arr = np.array(col)
        assert tuple(col_arr[0, 0][:3]) == (0, 0, 0)  # label 0 == background == black

    def test_returns_dict_with_both_keys(self, tmp_path: Path) -> None:
        label_map = torch.zeros(4, 4, dtype=torch.int64)
        paths = write_semantic_label_map(
            label_map, image_id="x", out_dir=tmp_path, class_names=["cat"]
        )
        assert "index_path" in paths
        assert "colorized_path" in paths
        assert Path(paths["index_path"]).exists()
        assert Path(paths["colorized_path"]).exists()

    def test_index_png_values_match_label_map(self, tmp_path: Path) -> None:
        label_map = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int64)
        paths = write_semantic_label_map(
            label_map, image_id="b", out_dir=tmp_path, class_names=["road", "sky"]
        )
        idx = np.array(Image.open(paths["index_path"]))
        np.testing.assert_array_equal(idx, label_map.numpy())

    def test_uint16_used_when_k_exceeds_255(self, tmp_path: Path) -> None:
        """When K+1 > 255, the index PNG must use uint16."""
        K = 255  # labels 0..255 → K+1 = 256 > 255
        label_map = torch.zeros(4, 4, dtype=torch.int64)
        label_map[0, 0] = 255
        class_names = [f"class_{i}" for i in range(K)]
        paths = write_semantic_label_map(
            label_map, image_id="c", out_dir=tmp_path, class_names=class_names
        )
        idx = np.array(Image.open(paths["index_path"]))
        assert idx.dtype == np.uint16, f"Expected uint16 for K+1>255; got {idx.dtype}"

    def test_colorized_background_is_black(self, tmp_path: Path) -> None:
        """Label 0 must always map to (0, 0, 0) regardless of class_names."""
        label_map = torch.zeros(8, 8, dtype=torch.int64)
        paths = write_semantic_label_map(
            label_map, image_id="d", out_dir=tmp_path, class_names=["thing"]
        )
        col = np.array(Image.open(paths["colorized_path"]))
        # All pixels are label 0 → all black
        assert (col[:, :, :3] == 0).all(), "Background (label 0) must be black in colorized PNG"

    def test_colorized_concept_color_deterministic(self, tmp_path: Path) -> None:
        """Same concept index must produce the same color across two calls."""
        label_map = torch.ones(4, 4, dtype=torch.int64)  # all label 1
        paths_a = write_semantic_label_map(
            label_map, image_id="e1", out_dir=tmp_path / "a", class_names=["road"]
        )
        paths_b = write_semantic_label_map(
            label_map, image_id="e2", out_dir=tmp_path / "b", class_names=["road"]
        )
        col_a = np.array(Image.open(paths_a["colorized_path"]))
        col_b = np.array(Image.open(paths_b["colorized_path"]))
        np.testing.assert_array_equal(col_a[:, :, :3], col_b[:, :, :3])

    def test_output_filenames_include_image_id(self, tmp_path: Path) -> None:
        label_map = torch.zeros(2, 2, dtype=torch.int64)
        paths = write_semantic_label_map(
            label_map, image_id="myimg", out_dir=tmp_path, class_names=["x"]
        )
        assert "myimg" in str(paths["index_path"])
        assert "myimg" in str(paths["colorized_path"])


# ===========================================================================
# Part 2: runner predictions.json entries under task=semantic
# ===========================================================================


class TestSemanticRunnerPredictionsJson:
    """The semantic runner branch emits predictions.json entries with the correct schema."""

    def test_semantic_runner_emits_predictions_json(self, tmp_path: Path) -> None:
        """semantic task -> predictions.json has label_map_path + concepts keys."""
        stub = _SemanticStubModule()
        opts = _make_semantic_opts(tmp_path)

        # Build a minimal semantic YAML config so _ResolvedConfig gets task=semantic
        class_map_path = _make_class_map(tmp_path)
        config_path = _make_semantic_config_yaml(tmp_path, class_map_path)
        opts_with_cfg = PredictOptions(
            images=opts.images,
            prompts=opts.prompts,
            output=opts.output,
            checkpoint=None,
            merge_adapter=False,
            config=config_path,
            score_threshold=0.0,
            top_k=100,
            save_masks="none",
            visualize=False,
            device="cpu",
            dtype="float32",
            seed=0,
            dry_run=False,
            verbose=False,
            batch_size=1,
        )

        with _patch_load_semantic(stub):
            report = run_predict_semantic(opts_with_cfg)

        pred_path = opts_with_cfg.output / "predictions.json"
        assert pred_path.exists(), "predictions.json must be written"
        entries = json.loads(pred_path.read_text())
        assert isinstance(entries, list)
        assert len(entries) >= 1
        for entry in entries:
            assert "image_id" in entry, f"Missing image_id in entry {entry!r}"
            assert "label_map_path" in entry, f"Missing label_map_path in entry {entry!r}"
            assert "concepts" in entry, f"Missing concepts in entry {entry!r}"
            assert isinstance(entry["concepts"], list)
        assert report.n_images == 1

    def test_semantic_runner_label_map_file_exists(self, tmp_path: Path) -> None:
        """The label_map_path in predictions.json points to a real file."""
        stub = _SemanticStubModule()
        class_map_path = _make_class_map(tmp_path)
        config_path = _make_semantic_config_yaml(tmp_path, class_map_path)
        opts = PredictOptions(
            images=_make_image_dir(tmp_path),
            prompts="road,tree",
            output=tmp_path / "out",
            checkpoint=None,
            merge_adapter=False,
            config=config_path,
            score_threshold=0.0,
            top_k=100,
            save_masks="none",
            visualize=False,
            device="cpu",
            dtype="float32",
            seed=0,
            dry_run=False,
            verbose=False,
            batch_size=1,
        )

        with _patch_load_semantic(stub):
            run_predict_semantic(opts)

        entries = json.loads((opts.output / "predictions.json").read_text())
        assert len(entries) >= 1
        lm_path = Path(entries[0]["label_map_path"])
        assert lm_path.exists(), f"label_map_path file not found: {lm_path}"

    def test_semantic_concepts_match_prompts(self, tmp_path: Path) -> None:
        """concepts list in each entry must equal the resolved class_names."""
        stub = _SemanticStubModule()
        class_map_path = _make_class_map(tmp_path)
        config_path = _make_semantic_config_yaml(tmp_path, class_map_path)
        opts = PredictOptions(
            images=_make_image_dir(tmp_path),
            prompts="road,tree",
            output=tmp_path / "out",
            checkpoint=None,
            merge_adapter=False,
            config=config_path,
            score_threshold=0.0,
            top_k=100,
            save_masks="none",
            visualize=False,
            device="cpu",
            dtype="float32",
            seed=0,
            dry_run=False,
            verbose=False,
            batch_size=1,
        )

        with _patch_load_semantic(stub):
            run_predict_semantic(opts)

        entries = json.loads((opts.output / "predictions.json").read_text())
        for entry in entries:
            assert entry["concepts"] == ["road", "tree"]


# ---------------------------------------------------------------------------
# Helper: run_predict with task=semantic wired through a config YAML
# ---------------------------------------------------------------------------


def run_predict_semantic(opts: PredictOptions) -> PredictReport:
    """Thin wrapper so tests import run_predict centrally."""
    from custom_sam_peft.predict.runner import run_predict

    return run_predict(opts)


# ===========================================================================
# Part 3: instance path invariance (byte-identical when task != semantic)
# ===========================================================================


class TestInstancePathInvariance:
    """Instance path must be byte-identical when task != semantic."""

    def test_instance_run_produces_expected_keys(self, tmp_path: Path) -> None:
        """Instance predict produces predictions.json with COCO schema (image_id, category_id)."""
        from custom_sam_peft.predict.runner import run_predict

        stub = _SemanticStubModule()
        img_dir = _make_image_dir(tmp_path)
        opts = PredictOptions(
            images=img_dir,
            prompts="cat,dog",
            output=tmp_path / "out",
            checkpoint=None,
            merge_adapter=False,
            config=None,  # no config → default instance task
            score_threshold=0.0,
            top_k=100,
            save_masks="rle",
            visualize=False,
            device="cpu",
            dtype="float32",
            seed=0,
            dry_run=False,
            verbose=False,
            batch_size=1,
        )

        with _patch_load_semantic(stub):
            report = run_predict(opts)

        entries = json.loads((tmp_path / "out" / "predictions.json").read_text())
        assert len(entries) > 0
        for entry in entries:
            assert "image_id" in entry
            assert "category_id" in entry
            assert "segmentation" in entry
            assert "label_map_path" not in entry, (
                "Instance predictions must NOT have label_map_path"
            )
            assert "concepts" not in entry, "Instance predictions must NOT have concepts"
        assert report.n_images == 1


# ===========================================================================
# Part 4: CLI prompt-defaulting under task=semantic
# ===========================================================================


class TestCliPromptDefaulting:
    """Under --config with task:semantic, --prompts defaults to class_names from class_map."""

    def _make_full_semantic_config(self, tmp_path: Path) -> Path:
        """Full semantic config with class_map containing road + tree."""
        mapping = {"0": "background", "1": "road", "2": "tree"}
        class_map_path = _make_class_map(tmp_path, mapping)
        return _make_semantic_config_yaml(tmp_path, class_map_path)

    def test_semantic_prompts_omitted_defaults_from_class_map(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI: semantic config + no --prompts → run_predict called with road,tree."""
        img_dir = _make_image_dir(tmp_path)
        config_path = self._make_full_semantic_config(tmp_path)
        out_dir = tmp_path / "out"

        captured: list[PredictOptions] = []

        def fake_run_predict(opts: PredictOptions) -> PredictReport:
            captured.append(opts)
            return PredictReport(n_images=0, n_predictions=0, elapsed_sec=0.1)

        with mock.patch(
            "custom_sam_peft.cli.predict_cmd.run_predict", side_effect=fake_run_predict
        ):
            result = _cli_runner.invoke(
                app,
                [
                    "predict",
                    "--images",
                    str(img_dir),
                    "--output",
                    str(out_dir),
                    "--config",
                    str(config_path),
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, f"CLI exited {result.exit_code}: {result.output}"
        assert len(captured) == 1
        # prompts should be the class names from class_map (road,tree) or similar
        opts = captured[0]
        from custom_sam_peft.predict.inputs import parse_prompts

        resolved = parse_prompts(opts.prompts)
        assert "road" in resolved, f"Expected 'road' in resolved prompts; got {resolved}"
        assert "tree" in resolved, f"Expected 'tree' in resolved prompts; got {resolved}"

    def test_instance_prompts_still_required_without_config(self, tmp_path: Path) -> None:
        """Instance mode without --config: --prompts omitted → non-zero exit (still required)."""
        img_dir = _make_image_dir(tmp_path)
        out_dir = tmp_path / "out"

        result = _cli_runner.invoke(
            app,
            [
                "predict",
                "--images",
                str(img_dir),
                "--output",
                str(out_dir),
                # no --prompts, no --config
            ],
        )
        # Should fail since --prompts is required in the non-semantic path
        assert result.exit_code != 0, (
            f"Expected non-zero exit when --prompts omitted for instance mode; "
            f"got exit_code={result.exit_code}, output={result.output!r}"
        )

    def test_explicit_prompts_override_class_map(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit --prompts overrides class_map defaults even under semantic config."""
        img_dir = _make_image_dir(tmp_path)
        config_path = self._make_full_semantic_config(tmp_path)
        out_dir = tmp_path / "out"

        captured: list[PredictOptions] = []

        def fake_run_predict(opts: PredictOptions) -> PredictReport:
            captured.append(opts)
            return PredictReport(n_images=0, n_predictions=0, elapsed_sec=0.1)

        with mock.patch(
            "custom_sam_peft.cli.predict_cmd.run_predict", side_effect=fake_run_predict
        ):
            result = _cli_runner.invoke(
                app,
                [
                    "predict",
                    "--images",
                    str(img_dir),
                    "--output",
                    str(out_dir),
                    "--config",
                    str(config_path),
                    "--prompts",
                    "custom_class",
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, f"CLI exited {result.exit_code}: {result.output}"
        assert len(captured) == 1
        opts = captured[0]
        assert "custom_class" in opts.prompts, (
            f"Explicit prompts should override class_map; got {opts.prompts!r}"
        )

    def test_semantic_instance_only_flags_emit_info(self, tmp_path: Path) -> None:
        """Under semantic config, instance-only flags (--score-threshold/--top-k/--save-masks)
        trigger one INFO log and are ignored (not an error).

        Note: configure_logging(..., force=True) installs a new root handler that replaces
        pytest's caplog handler, so we assert on result.output (CliRunner captures the
        basicConfig StreamHandler's output) rather than caplog.records.
        """
        img_dir = _make_image_dir(tmp_path)
        config_path = self._make_full_semantic_config(tmp_path)
        out_dir = tmp_path / "out"

        captured: list[PredictOptions] = []

        def fake_run_predict(opts: PredictOptions) -> PredictReport:
            captured.append(opts)
            return PredictReport(n_images=0, n_predictions=0, elapsed_sec=0.1)

        with mock.patch(
            "custom_sam_peft.cli.predict_cmd.run_predict", side_effect=fake_run_predict
        ):
            result = _cli_runner.invoke(
                app,
                [
                    "predict",
                    "--images",
                    str(img_dir),
                    "--output",
                    str(out_dir),
                    "--config",
                    str(config_path),
                    "--score-threshold",
                    "0.5",
                    "--top-k",
                    "50",
                    "--save-masks",
                    "png",
                ],
                catch_exceptions=False,
            )

        # Must not error
        assert result.exit_code == 0, f"CLI exited {result.exit_code}: {result.output}"
        # run_predict must still be called (flags ignored, not errored)
        assert len(captured) == 1
        # The one-time INFO must appear in the captured output
        assert "instance-only" in result.output, (
            f"Expected INFO about instance-only flags in output; got:\n{result.output}"
        )

    def test_semantic_instance_only_flags_no_info_when_defaults(self, tmp_path: Path) -> None:
        """Under semantic config with default flag values, the instance-only INFO is NOT emitted."""
        img_dir = _make_image_dir(tmp_path)
        config_path = self._make_full_semantic_config(tmp_path)
        out_dir = tmp_path / "out"

        def fake_run_predict(opts: PredictOptions) -> PredictReport:
            return PredictReport(n_images=0, n_predictions=0, elapsed_sec=0.1)

        with mock.patch(
            "custom_sam_peft.cli.predict_cmd.run_predict", side_effect=fake_run_predict
        ):
            result = _cli_runner.invoke(
                app,
                [
                    "predict",
                    "--images",
                    str(img_dir),
                    "--output",
                    str(out_dir),
                    "--config",
                    str(config_path),
                    # No instance-only flags set — all defaults
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, f"CLI exited {result.exit_code}: {result.output}"
        assert "instance-only" not in result.output, (
            f"Unexpected instance-only INFO in output:\n{result.output}"
        )
