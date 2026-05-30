"""Tests for epoch-relative schedule resolution (issue #163).

RED phase: all tests in this file are expected to FAIL before the fix is implemented.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Helper: import the resolver (will fail until implemented)
# ---------------------------------------------------------------------------


def _get_resolver():
    from custom_sam_peft.train.trainer import resolve_schedule_steps

    return resolve_schedule_steps


# ---------------------------------------------------------------------------
# (a) None → epoch-relative fill for save_every and eval_every
# ---------------------------------------------------------------------------


class TestNoneResolvesToEpochRelative:
    def test_save_every_defaults_to_steps_per_epoch(self):
        resolve = _get_resolver()
        save_every, _eval = resolve(
            save_every=None,
            eval_every=None,
            epochs=3,
            steps_per_epoch=100,
        )
        assert save_every == 100, f"Expected 100, got {save_every}"

    def test_eval_every_defaults_to_steps_per_epoch(self):
        resolve = _get_resolver()
        _save, eval_every = resolve(
            save_every=None,
            eval_every=None,
            epochs=3,
            steps_per_epoch=100,
        )
        assert eval_every == 100, f"Expected 100, got {eval_every}"


# ---------------------------------------------------------------------------
# (b) Explicit values pass through unchanged
# ---------------------------------------------------------------------------


class TestExplicitValuesPassThrough:
    def test_explicit_save_every_unchanged(self):
        resolve = _get_resolver()
        save_every, _eval = resolve(
            save_every=250,
            eval_every=None,
            epochs=2,
            steps_per_epoch=50,
        )
        assert save_every == 250

    def test_explicit_eval_every_unchanged(self):
        resolve = _get_resolver()
        _save, eval_every = resolve(
            save_every=None,
            eval_every=75,
            epochs=2,
            steps_per_epoch=50,
        )
        assert eval_every == 75

    def test_all_two_explicit_pass_through(self):
        resolve = _get_resolver()
        save_every, eval_every = resolve(
            save_every=500,
            eval_every=200,
            epochs=10,
            steps_per_epoch=200,
        )
        assert save_every == 500
        assert eval_every == 200


# ---------------------------------------------------------------------------
# (c) Short-run flooring: save/eval ≥ 1
# ---------------------------------------------------------------------------


class TestShortRunFlooring:
    def test_save_every_minimum_one(self):
        resolve = _get_resolver()
        save_every, eval_every = resolve(
            save_every=None,
            eval_every=None,
            epochs=1,
            steps_per_epoch=1,
        )
        assert save_every >= 1
        assert eval_every >= 1


# ---------------------------------------------------------------------------
# (d) Schema accepts both None and explicit ints for save/eval fields
# ---------------------------------------------------------------------------


class TestSchemaAcceptsNoneAndInt:
    def test_train_hyperparams_save_every_none(self):
        from custom_sam_peft.config.schema import TrainHyperparams

        cfg = TrainHyperparams(epochs=2, save_every=None)
        assert cfg.save_every is None

    def test_train_hyperparams_eval_every_none(self):
        from custom_sam_peft.config.schema import TrainHyperparams

        cfg = TrainHyperparams(epochs=2, eval_every=None)
        assert cfg.eval_every is None

    def test_train_hyperparams_save_every_explicit_int(self):
        from custom_sam_peft.config.schema import TrainHyperparams

        cfg = TrainHyperparams(epochs=2, save_every=500)
        assert cfg.save_every == 500

    def test_train_hyperparams_eval_every_explicit_int(self):
        from custom_sam_peft.config.schema import TrainHyperparams

        cfg = TrainHyperparams(epochs=2, eval_every=200)
        assert cfg.eval_every == 200

    def test_default_train_hyperparams_has_none_for_schedule_fields(self):
        """With the fix, the default for save_every and eval_every should be None."""
        from custom_sam_peft.config.schema import TrainHyperparams

        cfg = TrainHyperparams(epochs=5)
        assert cfg.save_every is None, f"Expected None, got {cfg.save_every}"
        assert cfg.eval_every is None, f"Expected None, got {cfg.eval_every}"


# ---------------------------------------------------------------------------
# (e) Resolved values land in the persisted config (model_copy)
# ---------------------------------------------------------------------------


class TestResolvedValuesInConfig:
    def test_resolved_values_reflect_in_model_copy(self, tmp_path: Path):
        """After resolve + model_copy, the updated cfg has resolved integers (not None)."""
        from custom_sam_peft.config.schema import (
            DataConfig,
            DataSplit,
            PEFTConfig,
            RunConfig,
            TrainConfig,
            TrainHyperparams,
        )
        from custom_sam_peft.train.trainer import resolve_schedule_steps

        cfg = TrainConfig(
            run=RunConfig(name="test", output_dir=str(tmp_path), seed=0),
            data=DataConfig(
                format="coco",
                train=DataSplit(annotations="a.json", images="i"),
                val=DataSplit(annotations="a.json", images="i"),
            ),
            peft=PEFTConfig(method="lora"),
            train=TrainHyperparams(
                epochs=4,
                save_every=None,
                eval_every=None,
            ),
        )

        epochs = cfg.train.epochs
        steps_per_epoch = 100
        save_every, eval_every = resolve_schedule_steps(
            save_every=cfg.train.save_every,
            eval_every=cfg.train.eval_every,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )

        # Build an updated cfg
        updated_train = cfg.train.model_copy(
            update={
                "save_every": save_every,
                "eval_every": eval_every,
            }
        )
        updated_cfg = cfg.model_copy(update={"train": updated_train})

        assert updated_cfg.train.save_every == steps_per_epoch
        assert updated_cfg.train.eval_every == steps_per_epoch

    def test_original_cfg_unchanged_after_resolve(self, tmp_path: Path):
        """The original cfg must be immutable; model_copy must not mutate it."""
        from custom_sam_peft.config.schema import (
            DataConfig,
            DataSplit,
            PEFTConfig,
            RunConfig,
            TrainConfig,
            TrainHyperparams,
        )
        from custom_sam_peft.train.trainer import resolve_schedule_steps

        cfg = TrainConfig(
            run=RunConfig(name="test", output_dir=str(tmp_path), seed=0),
            data=DataConfig(
                format="coco",
                train=DataSplit(annotations="a.json", images="i"),
                val=DataSplit(annotations="a.json", images="i"),
            ),
            peft=PEFTConfig(method="lora"),
            train=TrainHyperparams(
                epochs=4,
                save_every=None,
                eval_every=None,
            ),
        )

        save_every, eval_every = resolve_schedule_steps(
            save_every=cfg.train.save_every,
            eval_every=cfg.train.eval_every,
            epochs=cfg.train.epochs,
            steps_per_epoch=50,
        )

        updated_train = cfg.train.model_copy(
            update={
                "save_every": save_every,
                "eval_every": eval_every,
            }
        )
        _ = cfg.model_copy(update={"train": updated_train})

        # Original must be unchanged
        assert cfg.train.save_every is None
        assert cfg.train.eval_every is None
