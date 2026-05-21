"""Tests for the convs.3 missing-keys filter logic in custom_sam_peft.models.sam3.

This test suite exercises ``_classify_missing_keys`` in isolation — no sam3,
torch, or GPU required.  The function is a pure helper that decides whether a
(missing_keys, unexpected_keys) pair from load_state_dict should be silently
suppressed ("ok") or cause a loud RuntimeError ("fail").

Background: the released sam3.1_multiplex.pt is built from a 3-scale neck;
our load_sam31 instantiates a 4-scale neck.  convs[3] is dropped by the
scalp=1 trim in vl_combiner so its random init never participates in training.
The filter detects exactly that pattern and suppresses the noise.
"""

from __future__ import annotations

import ast

import pytest

from custom_sam_peft.models.sam3 import (
    _KNOWN_MISSING_KEYS,
    _SAM3_MISSING_KEYS_RE,
    _classify_missing_keys,
)

# ---------------------------------------------------------------------------
# Happy-path: exactly the known set → "ok"
# ---------------------------------------------------------------------------


def test_classify_exactly_known_set_is_ok() -> None:
    """The four convs.3 keys and no unexpected keys → "ok" (harmless noise)."""
    result = _classify_missing_keys(
        missing=set(_KNOWN_MISSING_KEYS),
        unexpected=set(),
    )
    assert result == "ok"


def test_classify_empty_missing_is_ok() -> None:
    """No missing keys at all (e.g. a future sam3 ships convs.3) → "ok".

    If the released checkpoint starts shipping convs[3] weights the missing set
    shrinks to empty.  That is strictly safer — the neck is now fully
    initialised — so we accept it.
    """
    result = _classify_missing_keys(missing=set(), unexpected=set())
    assert result == "ok"


def test_classify_subset_of_known_is_ok() -> None:
    """Proper subset of known missing keys and no unexpected → "ok".

    A new sam3 release might ship some (but not all) of the convs.3 keys.
    Fewer missing keys can only be safer, so a subset is accepted.
    """
    subset = {
        "backbone.vision_backbone.convs.3.conv_1x1.weight",
        "backbone.vision_backbone.convs.3.conv_1x1.bias",
    }
    assert subset < _KNOWN_MISSING_KEYS  # sanity: really a proper subset
    result = _classify_missing_keys(missing=subset, unexpected=set())
    assert result == "ok"


# ---------------------------------------------------------------------------
# Failure: known set PLUS extra keys → "fail"
# ---------------------------------------------------------------------------


def test_classify_known_set_plus_extra_key_is_fail() -> None:
    """Known set + one extra missing key → "fail" (could be checkpoint regression)."""
    extended = set(_KNOWN_MISSING_KEYS) | {"backbone.some_new_layer.weight"}
    result = _classify_missing_keys(missing=extended, unexpected=set())
    assert result == "fail"


def test_classify_entirely_unknown_missing_key_is_fail() -> None:
    """A single missing key not in the known set → "fail"."""
    result = _classify_missing_keys(
        missing={"backbone.transformer.encoder.layers.0.weight"},
        unexpected=set(),
    )
    assert result == "fail"


# ---------------------------------------------------------------------------
# Failure: any unexpected key → "fail"
# ---------------------------------------------------------------------------


def test_classify_unexpected_key_alone_is_fail() -> None:
    """Any unexpected key → "fail", even if missing keys look fine."""
    result = _classify_missing_keys(
        missing=set(_KNOWN_MISSING_KEYS),
        unexpected={"backbone.some_extra_layer.weight"},
    )
    assert result == "fail"


def test_classify_empty_missing_with_unexpected_is_fail() -> None:
    """No missing keys but unexpected keys present → "fail"."""
    result = _classify_missing_keys(
        missing=set(),
        unexpected={"backbone.unexpected.weight"},
    )
    assert result == "fail"


# ---------------------------------------------------------------------------
# Interaction: both missing-outside-known and unexpected → "fail"
# ---------------------------------------------------------------------------


def test_classify_both_extra_missing_and_unexpected_is_fail() -> None:
    """Both extra missing and unexpected keys → "fail"."""
    result = _classify_missing_keys(
        missing=set(_KNOWN_MISSING_KEYS) | {"backbone.new.weight"},
        unexpected={"backbone.extra.weight"},
    )
    assert result == "fail"


# ---------------------------------------------------------------------------
# ast.literal_eval safety: hostile-input rejection
# ---------------------------------------------------------------------------


def test_literal_eval_rejects_hostile_call_expression() -> None:
    """ast.literal_eval must reject a repr that contains a bare call expression.

    The canonical attack surface: if the captured text were anything other than
    a list-of-strings repr (e.g. due to a future sam3 format change that
    includes a computed value or if the regex over-captures), eval() would
    execute arbitrary code while ast.literal_eval would refuse.

    We test the property directly: ast.literal_eval rejects any input that
    contains a function call or other non-literal expression.  This confirms
    that swapping eval() → ast.literal_eval is the correct mitigation.
    """
    # This is NOT a list repr — it is a raw call expression.  eval() would
    # execute __import__ and run a shell command; ast.literal_eval must refuse.
    hostile_expr = "__import__('os').system('echo pwned')"
    with pytest.raises((ValueError, SyntaxError)):
        ast.literal_eval(hostile_expr)


def test_literal_eval_rejects_call_expression_in_key_name() -> None:
    """ast.literal_eval must raise on any non-literal token inside the list.

    Even when the attack is embedded among legitimate-looking keys the parser
    must refuse to evaluate the expression.
    """
    # The element after the first key is a call expression, not a string.
    hostile_repr = "['backbone.layer.weight', __import__('os').system('x')]"
    with pytest.raises((ValueError, SyntaxError)):
        ast.literal_eval(hostile_repr)


# ---------------------------------------------------------------------------
# Regex robustness: _SAM3_MISSING_KEYS_RE
# ---------------------------------------------------------------------------


def test_regex_matches_typical_sam3_output() -> None:
    """Regex matches the exact format sam3's _load_checkpoint prints."""
    text = (
        "loaded /path/to/sam3.1_multiplex.pt and found missing and/or unexpected keys:\n"
        "missing_keys=['backbone.vision_backbone.convs.3.conv_1x1.weight']\n"
    )
    m = _SAM3_MISSING_KEYS_RE.search(text)
    assert m is not None
    parsed = ast.literal_eval(m.group(1))
    assert parsed == ["backbone.vision_backbone.convs.3.conv_1x1.weight"]


def test_regex_does_not_consume_output_after_list() -> None:
    """Group 1 must not include text that follows the closing ']' of the list.

    With re.DOTALL and a greedy .+ group the old regex consumed everything to
    the end of the string, so ast.literal_eval would receive trailing text and
    raise.  The fixed regex uses a non-greedy \\[.*?\\] group so only the list
    repr is captured.
    """
    text = (
        "loaded /path/to/ckpt.pt and found missing and/or unexpected keys:\n"
        "missing_keys=['backbone.layer.weight']\n"
        "Some other sam3 progress output\n"
    )
    m = _SAM3_MISSING_KEYS_RE.search(text)
    assert m is not None
    # group(1) must be parseable — no trailing noise
    parsed = ast.literal_eval(m.group(1))
    assert parsed == ["backbone.layer.weight"]


def test_regex_matches_multikey_list() -> None:
    """Regex captures all four convs.3 keys as a list."""
    keys = sorted(_KNOWN_MISSING_KEYS)
    text = (
        "loaded /path/to/ckpt.pt and found missing and/or unexpected keys:\n"
        f"missing_keys={keys!r}\n"
    )
    m = _SAM3_MISSING_KEYS_RE.search(text)
    assert m is not None
    parsed = ast.literal_eval(m.group(1))
    assert set(parsed) == _KNOWN_MISSING_KEYS
