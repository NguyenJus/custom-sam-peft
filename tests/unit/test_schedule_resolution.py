"""Tests for epoch-relative schedule resolution (issue #163).

RED phase: all tests in this file are expected to FAIL before the fix is implemented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helper: import the resolver (will fail until implemented)
# ---------------------------------------------------------------------------


def _get_resolver():
    from custom_sam_peft.train.trainer import resolve_schedule_steps

    return resolve_schedule_steps


# ---------------------------------------------------------------------------
# (a) None → epoch-relative fill for all three
# ---------------------------------------------------------------------------


class TestNoneResolvesToEpochRelative:
    def test_save_every_defaults_to_steps_per_epoch(self):
        resolve = _get_resolver()
        save_every, _eval, _decay = resolve(
            save_every=None,
            eval_every=None,
            decay_steps=None,
            epochs=3,
            steps_per_epoch=100,
        )
        assert save_every == 100, f"Expected 100, got {save_every}"

    def test_eval_every_defaults_to_steps_per_epoch(self):
        resolve = _get_resolver()
        _save, eval_every, _decay = resolve(
            save_every=None,
            eval_every=None,
            decay_steps=None,
            epochs=3,
            steps_per_epoch=100,
        )
        assert eval_every == 100, f"Expected 100, got {eval_every}"

    def test_decay_steps_defaults_to_75_percent_of_run(self):
        resolve = _get_resolver()
        # epochs=4, steps_per_epoch=100 → total=400 → 0.75*400 = 300
        _save, _eval, decay_steps = resolve(
            save_every=None,
            eval_every=None,
            decay_steps=None,
            epochs=4,
            steps_per_epoch=100,
        )
        assert decay_steps == 300, f"Expected 300, got {decay_steps}"

    def test_decay_steps_formula_rounds(self):
        resolve = _get_resolver()
        # epochs=3, steps_per_epoch=10 → total=30 → 0.75*30 = 22.5 → round → 22 or 23
        _save, _eval, decay_steps = resolve(
            save_every=None,
            eval_every=None,
            decay_steps=None,
            epochs=3,
            steps_per_epoch=10,
        )
        expected = max(1, round(0.75 * 3 * 10))
        assert decay_steps == expected, f"Expected {expected}, got {decay_steps}"


# ---------------------------------------------------------------------------
# (b) Explicit values pass through unchanged
# ---------------------------------------------------------------------------


class TestExplicitValuesPassThrough:
    def test_explicit_save_every_unchanged(self):
        resolve = _get_resolver()
        save_every, _eval, _decay = resolve(
            save_every=250,
            eval_every=None,
            decay_steps=None,
            epochs=2,
            steps_per_epoch=50,
        )
        assert save_every == 250

    def test_explicit_eval_every_unchanged(self):
        resolve = _get_resolver()
        _save, eval_every, _decay = resolve(
            save_every=None,
            eval_every=75,
            decay_steps=None,
            epochs=2,
            steps_per_epoch=50,
        )
        assert eval_every == 75

    def test_explicit_decay_steps_unchanged(self):
        resolve = _get_resolver()
        _save, _eval, decay_steps = resolve(
            save_every=None,
            eval_every=None,
            decay_steps=9999,
            epochs=2,
            steps_per_epoch=50,
        )
        assert decay_steps == 9999

    def test_all_three_explicit_pass_through(self):
        resolve = _get_resolver()
        save_every, eval_every, decay_steps = resolve(
            save_every=500,
            eval_every=200,
            decay_steps=3000,
            epochs=10,
            steps_per_epoch=200,
        )
        assert save_every == 500
        assert eval_every == 200
        assert decay_steps == 3000


# ---------------------------------------------------------------------------
# (c) Short-run flooring: decay_steps ≥ 1
# ---------------------------------------------------------------------------


class TestShortRunFlooring:
    def test_single_epoch_single_step_decay_at_least_one(self):
        resolve = _get_resolver()
        _save, _eval, decay_steps = resolve(
            save_every=None,
            eval_every=None,
            decay_steps=None,
            epochs=1,
            steps_per_epoch=1,
        )
        # 0.75 * 1 * 1 = 0.75 → round → 1
        assert decay_steps >= 1, f"decay_steps must be ≥ 1, got {decay_steps}"

    def test_save_every_minimum_one(self):
        resolve = _get_resolver()
        save_every, eval_every, _decay = resolve(
            save_every=None,
            eval_every=None,
            decay_steps=None,
            epochs=1,
            steps_per_epoch=1,
        )
        assert save_every >= 1
        assert eval_every >= 1

    def test_tiny_run_decay_steps_floor(self):
        """epochs=1, steps_per_epoch=1 → 0.75*1*1=0.75 → rounds to 1 → max(1,1)=1."""
        resolve = _get_resolver()
        _s, _e, decay_steps = resolve(
            save_every=None,
            eval_every=None,
            decay_steps=None,
            epochs=1,
            steps_per_epoch=1,
        )
        assert decay_steps == 1


# ---------------------------------------------------------------------------
# (d) Schema accepts both None and explicit ints
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

    def test_box_hint_decay_steps_none(self):
        from custom_sam_peft.config.schema import BoxHintSchedule

        schedule = BoxHintSchedule(decay_steps=None)
        assert schedule.decay_steps is None

    def test_train_hyperparams_save_every_explicit_int(self):
        from custom_sam_peft.config.schema import TrainHyperparams

        cfg = TrainHyperparams(epochs=2, save_every=500)
        assert cfg.save_every == 500

    def test_train_hyperparams_eval_every_explicit_int(self):
        from custom_sam_peft.config.schema import TrainHyperparams

        cfg = TrainHyperparams(epochs=2, eval_every=200)
        assert cfg.eval_every == 200

    def test_box_hint_decay_steps_explicit_int(self):
        from custom_sam_peft.config.schema import BoxHintSchedule

        schedule = BoxHintSchedule(decay_steps=3000)
        assert schedule.decay_steps == 3000

    def test_box_hint_monotone_validator_still_works_with_none_decay(self):
        """The _check_monotone validator must still work when decay_steps is None."""
        from custom_sam_peft.config.schema import BoxHintSchedule

        # Valid: p_start >= p_end
        schedule = BoxHintSchedule(p_start=1.0, p_end=0.0, decay_steps=None)
        assert schedule.p_start == 1.0
        assert schedule.p_end == 0.0

    def test_box_hint_monotone_validator_still_rejects_invalid_with_none_decay(self):
        """The validator should reject p_end > p_start even when decay_steps=None."""
        from pydantic import ValidationError

        from custom_sam_peft.config.schema import BoxHintSchedule

        with pytest.raises(ValidationError, match="must decay"):
            BoxHintSchedule(p_start=0.0, p_end=1.0, decay_steps=None)

    def test_default_train_hyperparams_has_none_for_schedule_fields(self):
        """With the fix, the default for save_every and eval_every should be None."""
        from custom_sam_peft.config.schema import TrainHyperparams

        cfg = TrainHyperparams(epochs=5)
        assert cfg.save_every is None, f"Expected None, got {cfg.save_every}"
        assert cfg.eval_every is None, f"Expected None, got {cfg.eval_every}"

    def test_default_box_hint_has_none_for_decay_steps(self):
        """With the fix, the default for decay_steps should be None."""
        from custom_sam_peft.config.schema import BoxHintSchedule

        schedule = BoxHintSchedule()
        assert schedule.decay_steps is None, f"Expected None, got {schedule.decay_steps}"


# ---------------------------------------------------------------------------
# (e) Resolved values land in the persisted config (model_copy)
# ---------------------------------------------------------------------------


class TestResolvedValuesInConfig:
    def test_resolved_values_reflect_in_model_copy(self, tmp_path: Path):
        """After resolve + model_copy, the updated cfg has resolved integers (not None)."""
        from custom_sam_peft.config.schema import (
            BoxHintSchedule,
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
                box_hint=BoxHintSchedule(decay_steps=None),
            ),
        )

        epochs = cfg.train.epochs
        steps_per_epoch = 100
        save_every, eval_every, decay_steps = resolve_schedule_steps(
            save_every=cfg.train.save_every,
            eval_every=cfg.train.eval_every,
            decay_steps=cfg.train.box_hint.decay_steps,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )

        # Build an updated cfg
        updated_box_hint = cfg.train.box_hint.model_copy(update={"decay_steps": decay_steps})
        updated_train = cfg.train.model_copy(
            update={
                "save_every": save_every,
                "eval_every": eval_every,
                "box_hint": updated_box_hint,
            }
        )
        updated_cfg = cfg.model_copy(update={"train": updated_train})

        assert updated_cfg.train.save_every == steps_per_epoch
        assert updated_cfg.train.eval_every == steps_per_epoch
        assert updated_cfg.train.box_hint.decay_steps == max(
            1, round(0.75 * epochs * steps_per_epoch)
        )

    def test_original_cfg_unchanged_after_resolve(self, tmp_path: Path):
        """The original cfg must be immutable; model_copy must not mutate it."""
        from custom_sam_peft.config.schema import (
            BoxHintSchedule,
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
                box_hint=BoxHintSchedule(decay_steps=None),
            ),
        )

        save_every, eval_every, decay_steps = resolve_schedule_steps(
            save_every=cfg.train.save_every,
            eval_every=cfg.train.eval_every,
            decay_steps=cfg.train.box_hint.decay_steps,
            epochs=cfg.train.epochs,
            steps_per_epoch=50,
        )

        updated_box_hint = cfg.train.box_hint.model_copy(update={"decay_steps": decay_steps})
        updated_train = cfg.train.model_copy(
            update={
                "save_every": save_every,
                "eval_every": eval_every,
                "box_hint": updated_box_hint,
            }
        )
        _ = cfg.model_copy(update={"train": updated_train})

        # Original must be unchanged
        assert cfg.train.save_every is None
        assert cfg.train.eval_every is None
        assert cfg.train.box_hint.decay_steps is None
