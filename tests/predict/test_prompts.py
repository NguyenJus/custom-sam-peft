"""Tests for predict/inputs.py — parse_prompts."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from custom_sam_peft.predict.inputs import parse_prompts


def test_parse_prompts_comma_string() -> None:
    """Comma-separated string splits into the expected list."""
    result = parse_prompts("cat,dog,person")
    assert result == ["cat", "dog", "person"]


def test_parse_prompts_one_per_line_file(prompts_file: Path) -> None:
    """UTF-8 file with one class per line → list in file order."""
    result = parse_prompts(str(prompts_file))
    assert result == ["cat", "dog", "person"]


def test_parse_prompts_strips_whitespace() -> None:
    """Leading/trailing whitespace is stripped from every entry."""
    result = parse_prompts("  cat  ,dog   ")
    assert result == ["cat", "dog"]


def test_parse_prompts_drops_empty_entries() -> None:
    """Empty tokens from doubled commas or trailing commas are dropped."""
    result = parse_prompts("cat,,dog,")
    assert result == ["cat", "dog"]


def test_parse_prompts_dedupes_first_seen() -> None:
    """Duplicate entries are removed, keeping first occurrence."""
    result = parse_prompts("cat,dog,cat,bird,dog")
    assert result == ["cat", "dog", "bird"]


def test_parse_prompts_empty_raises() -> None:
    """Empty string raises typer.BadParameter with the spec message."""
    with pytest.raises(typer.BadParameter) as exc_info:
        parse_prompts("")

    assert "--prompts must resolve to at least one non-empty class name" in str(exc_info.value)


def test_parse_prompts_all_whitespace_raises() -> None:
    """A string of only whitespace/commas raises typer.BadParameter."""
    with pytest.raises(typer.BadParameter):
        parse_prompts("  ,  ,  ")


def test_parse_prompts_empty_file_raises(tmp_path: Path) -> None:
    """An empty file raises typer.BadParameter."""
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")

    with pytest.raises(typer.BadParameter) as exc_info:
        parse_prompts(str(f))

    assert "--prompts must resolve to at least one non-empty class name" in str(exc_info.value)


def test_parse_prompts_file_with_blanks_and_whitespace(tmp_path: Path) -> None:
    """File entries are stripped; blank lines dropped; deduplication applied."""
    f = tmp_path / "classes.txt"
    f.write_text("  cat  \n\ndog\ncat\n  \n", encoding="utf-8")

    result = parse_prompts(str(f))
    assert result == ["cat", "dog"]


def test_parse_prompts_preserves_order(tmp_path: Path) -> None:
    """File-based prompts preserve the order they appear in the file."""
    f = tmp_path / "order.txt"
    f.write_text("zebra\napple\nmango\n", encoding="utf-8")

    result = parse_prompts(str(f))
    assert result == ["zebra", "apple", "mango"]
