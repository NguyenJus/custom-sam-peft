"""Unit tests for custom_sam_peft.profiling (issue #255).

CPU-only — no model, no GPU required.

Run with:
    uv run pytest tests/unit/test_profiling.py -o "addopts=" -p no:cacheprovider
"""

from __future__ import annotations

import json
import time

import pytest

import custom_sam_peft.profiling as prof

# ---------------------------------------------------------------------------
# Fixture: ensure each test starts with a clean, disabled profiler state.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_profiler() -> None:  # type: ignore[return]
    """Reset + disable before and after every test."""
    prof.reset()
    prof.disable()
    yield
    prof.reset()
    prof.disable()


# ---------------------------------------------------------------------------
# Disabled no-op behaviour
# ---------------------------------------------------------------------------


class TestDisabledNoop:
    def test_is_enabled_false_after_disable(self) -> None:
        assert not prof.is_enabled()

    def test_bucket_does_not_touch_timer_or_dicts(self) -> None:
        """bucket() when disabled must yield without touching any state."""
        with prof.bucket("eval.forward"):
            time.sleep(0)  # trivial body

        buckets, meta = prof.snapshot()
        assert buckets == {}
        assert meta == {}

    def test_note_noop(self) -> None:
        prof.note(x=42, y="hello")
        _, meta = prof.snapshot()
        assert meta == {}

    def test_incr_noop(self) -> None:
        prof.incr("counter")
        prof.incr("counter", by=5)
        _, meta = prof.snapshot()
        assert meta == {}

    def test_snapshot_returns_empty_dicts(self) -> None:
        buckets, meta = prof.snapshot()
        assert buckets == {}
        assert meta == {}

    def test_multiple_bucket_calls_stay_empty(self) -> None:
        for _ in range(3):
            with prof.bucket("train.forward"):
                pass
        buckets, _ = prof.snapshot()
        assert buckets == {}


# ---------------------------------------------------------------------------
# Disabled no-op PROOF: the timer / CUDA sync must not even be *invoked* when
# disabled (the actual overhead the no-op guarantee is about — empty dicts are
# necessary but not sufficient; a stray sync before the short-circuit is the
# real regression we guard against).
# ---------------------------------------------------------------------------


class TestDisabledNoopProof:
    def test_disabled_bucket_never_calls_perf_counter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(time, "perf_counter", lambda: calls.append(1) or 0.0)

        with prof.bucket("eval.forward"):
            pass

        assert calls == [], "disabled bucket() must not invoke time.perf_counter"

    def test_enabled_bucket_does_call_perf_counter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []
        monkeypatch.setattr(time, "perf_counter", lambda: calls.append(1) or 0.0)

        prof.enable()
        with prof.bucket("eval.forward"):
            pass

        assert calls, "enabled bucket() must time via time.perf_counter"

    def test_disabled_bucket_never_syncs_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch = pytest.importorskip("torch")
        synced = []
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: synced.append(1))

        with prof.bucket("eval.forward"):
            pass

        assert synced == [], "disabled bucket() must not call torch.cuda.synchronize"


# ---------------------------------------------------------------------------
# Enabled accumulation
# ---------------------------------------------------------------------------


class TestEnabledAccumulation:
    def test_single_bucket_records_time(self) -> None:
        prof.enable()
        with prof.bucket("eval.forward"):
            time.sleep(0.02)

        buckets, _ = prof.snapshot()
        assert "eval.forward" in buckets
        assert buckets["eval.forward"] >= 0.01  # at least half the sleep

    def test_two_blocks_accumulate_into_one_key(self) -> None:
        prof.enable()
        with prof.bucket("eval.rle_encode"):
            time.sleep(0.01)
        with prof.bucket("eval.rle_encode"):
            time.sleep(0.01)

        buckets, _ = prof.snapshot()
        assert buckets["eval.rle_encode"] >= 0.015  # two sleeps accumulated

    def test_longer_sleep_yields_larger_bucket(self) -> None:
        prof.enable()
        with prof.bucket("a"):
            time.sleep(0.01)
        with prof.bucket("b"):
            time.sleep(0.05)

        buckets, _ = prof.snapshot()
        assert buckets["b"] > buckets["a"]

    def test_note_records_metadata(self) -> None:
        prof.enable()
        prof.note(n_images=8, split="val")

        _, meta = prof.snapshot()
        assert meta["n_images"] == 8
        assert meta["split"] == "val"

    def test_note_last_value_wins(self) -> None:
        prof.enable()
        prof.note(x=1)
        prof.note(x=99)

        _, meta = prof.snapshot()
        assert meta["x"] == 99

    def test_incr_accumulates_and_defaults_to_1(self) -> None:
        prof.enable()
        prof.incr("k")
        prof.incr("k", by=2)

        _, meta = prof.snapshot()
        assert meta["k"] == 3

    def test_incr_starts_from_zero(self) -> None:
        prof.enable()
        prof.incr("fresh")

        _, meta = prof.snapshot()
        assert meta["fresh"] == 1

    def test_multiple_different_buckets(self) -> None:
        prof.enable()
        for name in ("eval.forward", "eval.rle_encode", "train.loss"):
            with prof.bucket(name):
                time.sleep(0.001)

        buckets, _ = prof.snapshot()
        assert set(buckets.keys()) == {"eval.forward", "eval.rle_encode", "train.loss"}


# ---------------------------------------------------------------------------
# snapshot() / snapshot_json() shape and copy isolation
# ---------------------------------------------------------------------------


class TestSnapshotShape:
    def test_snapshot_json_has_buckets_and_meta_keys(self) -> None:
        prof.enable()
        with prof.bucket("predict.forward"):
            pass
        prof.note(info="test")

        data = json.loads(prof.snapshot_json())
        assert "buckets" in data
        assert "meta" in data

    def test_snapshot_json_default_indent(self) -> None:
        prof.enable()
        prof.note(x=1)
        raw = prof.snapshot_json()
        # Indented JSON has newlines
        assert "\n" in raw

    def test_snapshot_json_custom_indent(self) -> None:
        prof.enable()
        raw = prof.snapshot_json(indent=4)
        data = json.loads(raw)
        assert "buckets" in data

    def test_snapshot_buckets_copy_isolation(self) -> None:
        prof.enable()
        with prof.bucket("eval.forward"):
            pass

        buckets, _ = prof.snapshot()
        original_val = buckets["eval.forward"]
        buckets["eval.forward"] = 9999.0  # mutate the returned copy

        buckets2, _ = prof.snapshot()
        assert buckets2["eval.forward"] == original_val  # internal state unchanged

    def test_snapshot_meta_copy_isolation(self) -> None:
        prof.enable()
        prof.note(key="original")

        _, meta = prof.snapshot()
        meta["key"] = "mutated"  # mutate the returned copy

        _, meta2 = prof.snapshot()
        assert meta2["key"] == "original"  # internal state unchanged


# ---------------------------------------------------------------------------
# dump()
# ---------------------------------------------------------------------------


class TestDump:
    def test_dump_writes_valid_json(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[override]
        prof.enable()
        with prof.bucket("eval.forward"):
            pass
        prof.note(n=3)

        out = prof.dump(tmp_path / "p.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert "buckets" in data
        assert "meta" in data

    def test_dump_returns_path(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[override]
        prof.enable()
        result = prof.dump(tmp_path / "snap.json")
        assert isinstance(result, type(tmp_path))

    def test_dump_creates_parent_dirs(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[override]
        nested = tmp_path / "a" / "b" / "c.json"
        prof.enable()
        prof.dump(nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_buckets(self) -> None:
        prof.enable()
        with prof.bucket("eval.forward"):
            time.sleep(0.001)

        prof.reset()
        buckets, _ = prof.snapshot()
        assert buckets == {}

    def test_reset_clears_meta(self) -> None:
        prof.enable()
        prof.note(x=1)
        prof.incr("counter")

        prof.reset()
        _, meta = prof.snapshot()
        assert meta == {}

    def test_reset_does_not_affect_enabled_state(self) -> None:
        prof.enable()
        prof.reset()
        assert prof.is_enabled()


# ---------------------------------------------------------------------------
# enable() / disable() toggle
# ---------------------------------------------------------------------------


class TestToggle:
    def test_enable_then_disable(self) -> None:
        prof.enable()
        assert prof.is_enabled()
        prof.disable()
        assert not prof.is_enabled()

    def test_data_collected_only_while_enabled(self) -> None:
        prof.disable()
        with prof.bucket("eval.forward"):
            pass

        prof.enable()
        with prof.bucket("eval.forward"):
            time.sleep(0.01)

        prof.disable()
        with prof.bucket("eval.forward"):
            pass

        buckets, _ = prof.snapshot()
        # Only the one enabled block should be counted
        assert buckets.get("eval.forward", 0) < 0.5  # not 3 blocks inflated
        assert "eval.forward" in buckets  # the enabled block was counted
