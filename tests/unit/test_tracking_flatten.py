"""Tests for flatten_metrics_report — duck-typed scalar flattening."""

from __future__ import annotations

from dataclasses import dataclass, field

from custom_sam_peft.tracking import flatten_metrics_report


@dataclass
class _FakeReport:
    overall: dict[str, float] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)


def test_flatten_overall_only() -> None:
    r = _FakeReport(overall={"mAP": 0.42, "mAP_50": 0.6}, per_class={})
    out = flatten_metrics_report(r)
    assert out == {"eval/mAP": 0.42, "eval/mAP_50": 0.6}


def test_flatten_with_per_class() -> None:
    r = _FakeReport(
        overall={"mAP": 0.5},
        per_class={
            "cat": {"AP": 0.7, "AP_50": 0.8},
            "dog": {"AP": 0.3, "AP_50": 0.4},
        },
    )
    out = flatten_metrics_report(r)
    assert out == {
        "eval/mAP": 0.5,
        "eval/per_class/cat/AP": 0.7,
        "eval/per_class/cat/AP_50": 0.8,
        "eval/per_class/dog/AP": 0.3,
        "eval/per_class/dog/AP_50": 0.4,
    }


def test_flatten_sanitizes_slash_in_class_name() -> None:
    r = _FakeReport(overall={}, per_class={"animals/cat": {"AP": 0.9}})
    out = flatten_metrics_report(r)
    assert out == {"eval/per_class/animals_cat/AP": 0.9}


def test_flatten_custom_prefix() -> None:
    r = _FakeReport(overall={"mAP": 0.1}, per_class={"cat": {"AP": 0.2}})
    out = flatten_metrics_report(r, prefix="val")
    assert out == {"val/mAP": 0.1, "val/per_class/cat/AP": 0.2}


def test_flatten_returns_floats() -> None:
    r = _FakeReport(overall={"mAP": 1}, per_class={"cat": {"AP": 2}})  # ints in
    out = flatten_metrics_report(r)
    assert all(isinstance(v, float) for v in out.values())
