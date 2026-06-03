"""Attribution report reader for CSP_PROFILE snapshots (issue #256 / #273).

Consumes the harness snapshot JSON (produced by ``profiling.dump`` /
``snapshot_json``) and produces:

- Bucket ranking by share of timed wall-time.
- Dominant-path identification and CPU/GPU/sync/IO split.
- Structural facts extracted from meta (N, forwards, dtype, HW, n_images).
- Lever GO/NO-GO heuristics per the #250 documented rules.
- Regression detection against a baseline snapshot.
- Markdown report skeleton pre-filled from the data.

Pure snapshot-in / report-out.  No GPU, no torch import required.

Module home: ``src/custom_sam_peft/profiling_report.py``
Runner: ``scripts/attribute_profile.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Bucket classification table
# (data-driven so new buckets are easy to add)
# ---------------------------------------------------------------------------

#: Map from fully-qualified bucket name → kind string.
#: Unknown buckets (not in this mapping) → "unclassified".
BUCKET_KINDS: dict[str, str] = {
    # GPU spans
    "eval.forward": "gpu",
    "eval.mask_upsample": "gpu",
    "eval.proxy_iou": "gpu",
    "semantic_eval.forward": "gpu",
    "semantic_eval.upsample": "gpu",
    "train.forward": "gpu",
    "predict.forward": "gpu",
    # CPU spans
    "eval.rle_encode": "cpu",
    "eval.gt_rle_encode": "cpu",
    "eval.coco_aggregate": "cpu",
    "eval.pair_iou": "cpu",
    "semantic_eval.confusion": "cpu",
    "train.matcher": "cpu",
    "train.loss": "cpu",
    "train.backward": "cpu",
    "train.optim_step": "cpu",
    "predict.postprocess": "cpu",
    "predict.write": "cpu",
    # Sync (host↔device transfer) spans
    "eval.transfer_binarize": "sync",
    "eval.box_transfer": "sync",
    "semantic_eval.transfer": "sync",
    # I/O spans
    "eval.dataset_load": "io",
}

#: The set of bucket names that are *parent* wall-clock wrappers, not leaf spans.
#: When present, a parent bucket is the denominator; it is NOT listed as a leaf row.
_TOTAL_SUFFIXES = (".total",)


def _is_total(name: str) -> bool:
    return any(name.endswith(s) for s in _TOTAL_SUFFIXES)


# Known surface prefixes, longest-first to avoid ambiguity.
_SURFACE_PREFIXES = ["semantic_eval", "eval", "train", "predict"]


def _detect_surface(bucket_names: list[str]) -> str:
    """Detect the profiling surface from bucket names."""
    for name in bucket_names:
        for prefix in _SURFACE_PREFIXES:
            if name.startswith(prefix + ".") or name == prefix:
                return prefix
    return "unknown"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BucketRow:
    """One ranked leaf bucket."""

    name: str
    seconds: float
    share: float  # fraction of denominator [0, 1]
    kind: str  # "gpu" | "cpu" | "sync" | "io" | "unclassified"


@dataclass
class DominantPath:
    """Summary of the dominant bucket and the CPU/GPU/sync/IO breakdown."""

    name: str
    kind: str
    share: float
    gpu_share: float
    cpu_share: float
    sync_share: float
    io_share: float
    unclassified_share: float


@dataclass
class GoNoGoVerdict:
    """One GO/NO-GO heuristic verdict."""

    rule_id: str
    verdict: str  # "GO" | "NO-GO"
    reason: str
    citation: str  # #NNN reference


@dataclass
class RegressionFlag:
    """One bucket flagged as regressed vs. baseline."""

    name: str
    baseline_share: float
    current_share: float
    share_delta: float  # current - baseline
    baseline_seconds: float
    current_seconds: float
    seconds_delta: float  # current - baseline


@dataclass
class ReportData:
    """Full attributed report data, ready for rendering."""

    surface: str
    rows: list[BucketRow]  # leaf buckets, sorted descending by share
    dominant: DominantPath
    wall_seconds: float  # denominator used
    residual_seconds: float  # total - sum(leaves); 0 when no *.total parent
    pct_label: str  # "% of wall" | "% of timed"
    total_label: str  # "TOTAL(wall)" | "TOTAL(timed)"
    facts: dict[str, Any]  # structural facts; missing keys → "unknown"
    verdicts: list[GoNoGoVerdict]
    raw_meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Core attribution logic
# ---------------------------------------------------------------------------


def _extract_facts(meta: dict[str, Any], surface: str) -> dict[str, Any]:
    """Pull structural facts from meta; degrade to "unknown" on missing keys."""
    facts: dict[str, Any] = {}

    facts["N"] = meta.get("N", "unknown")
    # forwards key is surface-dependent
    fwd_key = f"{surface}.forwards" if surface != "unknown" else None
    if fwd_key and fwd_key in meta:
        facts["forwards"] = meta[fwd_key]
    else:
        facts["forwards"] = meta.get("forwards", "unknown")

    facts["n_images"] = meta.get("n_images", "unknown")
    facts["mask_logit_hw"] = meta.get("mask_logit_hw", "unknown")
    facts["sem_forward_dtype"] = meta.get("sem_forward_dtype", "unknown")
    facts["K"] = meta.get("K", "unknown")  # semantic class count (§3b semantic meta)

    # Derive forwards_per_image when both are available and numeric
    fwd = facts["forwards"]
    n_img = facts["n_images"]
    if isinstance(fwd, (int, float)) and isinstance(n_img, (int, float)) and n_img > 0:
        facts["forwards_per_image"] = fwd / n_img
    else:
        facts["forwards_per_image"] = "unknown"

    return facts


def _compute_go_nogo(
    rows: list[BucketRow],
    facts: dict[str, Any],
    surface: str,
    rle_threshold: float,
) -> list[GoNoGoVerdict]:
    """Apply documented #250 lever GO/NO-GO heuristics."""
    verdicts: list[GoNoGoVerdict] = []

    # Rule: postprocess/RLE dominates AND N > 100 → top-100 query-filter GO
    # Only meaningful on eval surface (exact/full path, not lite proxy).
    share_by_name = {r.name: r.share for r in rows}
    rle_share = share_by_name.get("eval.rle_encode", 0.0)
    N = facts.get("N", "unknown")
    N_numeric = N if isinstance(N, (int, float)) else None

    rle_dominant = rle_share >= rle_threshold
    n_large = N_numeric is not None and N_numeric > 100

    if rle_dominant and n_large:
        rle_verdict = "GO"
        rle_reason = (
            f"eval.rle_encode share {rle_share:.1%} ≥ threshold {rle_threshold:.1%}"
            f" AND N={N_numeric} > 100"
        )
    elif rle_dominant and N_numeric is None:
        # RLE dominates but we can't confirm the N>100 precondition — do NOT
        # claim N ≤ 100 (false fact in a triage record); flag N as unknown.
        rle_verdict = "NO-GO"
        rle_reason = (
            f"eval.rle_encode share {rle_share:.1%} ≥ threshold {rle_threshold:.1%}"
            " but N unknown — cannot confirm the N>100 precondition for the top-100 filter"
        )
    elif rle_dominant and not n_large:
        rle_verdict = "NO-GO"
        rle_reason = (
            f"eval.rle_encode share {rle_share:.1%} ≥ threshold but N={N_numeric} ≤ 100"
            " — top-100 filter does not help when N already ≤ 100"
        )
    elif not rle_dominant and N_numeric is not None:
        rle_verdict = "NO-GO"
        rle_reason = f"eval.rle_encode share {rle_share:.1%} < threshold {rle_threshold:.1%}"
    else:
        rle_verdict = "NO-GO"
        rle_reason = (
            f"eval.rle_encode share {rle_share:.1%} < threshold {rle_threshold:.1%}; N unknown"
        )

    verdicts.append(
        GoNoGoVerdict(
            rule_id="postprocess_rle_top100",
            verdict=rle_verdict,
            reason=rle_reason,
            citation=(
                "#250 (#273 §6): postprocess/RLE dominates (share >= threshold)"
                " AND N > 100 -> top-100 query filter is GO"
            ),
        )
    )

    return verdicts


def attribute_snapshot(
    snap: dict[str, Any],
    rle_threshold: float = 0.05,
) -> ReportData:
    """Attribute a profiler snapshot dict and return a :class:`ReportData`.

    Parameters
    ----------
    snap:
        Dict with ``"buckets"`` (name → seconds) and ``"meta"`` (free-form).
    rle_threshold:
        Minimum ``eval.rle_encode`` share to trigger the top-100 query-filter
        GO verdict (spec §6 default: 0.05 = 5%).
    """
    buckets: dict[str, float] = snap.get("buckets", {})
    meta: dict[str, Any] = snap.get("meta", {})

    bucket_names = list(buckets.keys())
    surface = _detect_surface(bucket_names)

    # Separate total (parent) buckets from leaf buckets.
    total_name: str | None = None
    leaf_buckets: dict[str, float] = {}
    for name, secs in buckets.items():
        if _is_total(name):
            total_name = name
        else:
            leaf_buckets[name] = secs

    leaf_total = sum(leaf_buckets.values())
    has_total = total_name is not None
    wall_seconds: float
    if has_total:
        wall_seconds = buckets[total_name] or 1.0  # type: ignore[index]
        pct_label = "% of wall"
        total_label = "TOTAL(wall)"
    else:
        wall_seconds = leaf_total or 1.0
        pct_label = "% of timed"
        total_label = "TOTAL(timed)"

    residual_seconds = (wall_seconds - leaf_total) if has_total else 0.0

    # Build ranked leaf rows.
    rows: list[BucketRow] = []
    for name, secs in leaf_buckets.items():
        share = secs / wall_seconds
        kind = BUCKET_KINDS.get(name, "unclassified")
        rows.append(BucketRow(name=name, seconds=secs, share=share, kind=kind))
    rows.sort(key=lambda r: r.share, reverse=True)

    # Dominant path + split.
    dominant_row = (
        rows[0] if rows else BucketRow(name="(none)", seconds=0.0, share=0.0, kind="unclassified")
    )
    gpu_share = sum(r.share for r in rows if r.kind == "gpu")
    cpu_share = sum(r.share for r in rows if r.kind == "cpu")
    sync_share = sum(r.share for r in rows if r.kind == "sync")
    io_share = sum(r.share for r in rows if r.kind == "io")
    unclassified_share = sum(r.share for r in rows if r.kind == "unclassified")

    dominant = DominantPath(
        name=dominant_row.name,
        kind=dominant_row.kind,
        share=dominant_row.share,
        gpu_share=gpu_share,
        cpu_share=cpu_share,
        sync_share=sync_share,
        io_share=io_share,
        unclassified_share=unclassified_share,
    )

    # Structural facts.
    facts = _extract_facts(meta, surface)

    # GO/NO-GO verdicts.
    verdicts = _compute_go_nogo(rows, facts, surface, rle_threshold)

    return ReportData(
        surface=surface,
        rows=rows,
        dominant=dominant,
        wall_seconds=wall_seconds,
        residual_seconds=residual_seconds,
        pct_label=pct_label,
        total_label=total_label,
        facts=facts,
        verdicts=verdicts,
        raw_meta=dict(meta),
    )


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def compare_snapshots(
    baseline: dict[str, Any],
    current: dict[str, Any],
    share_tolerance: float = 0.05,
    seconds_tolerance: float | None = None,
) -> list[RegressionFlag]:
    """Diff two snapshot dicts; return buckets whose share grew beyond tolerance.

    A bucket is flagged when its **share delta** exceeds *share_tolerance*.
    If *seconds_tolerance* is also supplied, the bucket is additionally flagged
    when its absolute seconds increase exceeds that threshold (OR logic).

    Parameters
    ----------
    baseline:
        Earlier snapshot dict (same shape as ``attribute_snapshot`` input).
    current:
        Newer snapshot dict to compare against the baseline.
    share_tolerance:
        Minimum share increase (e.g. 0.05 = 5 percentage points) to flag.
        Primary gate — share growth is the canonical regression signal.
    seconds_tolerance:
        If provided, also flag when absolute seconds grew by more than this
        amount (independent of share).  Default ``None`` — seconds growth alone
        does not trigger a flag.

    Returns a list of :class:`RegressionFlag`, one per flagged bucket.
    Empty list means no regressions detected above the tolerance.
    """
    base_data = attribute_snapshot(baseline)
    curr_data = attribute_snapshot(current)

    base_shares = {r.name: r.share for r in base_data.rows}
    curr_shares = {r.name: r.share for r in curr_data.rows}
    base_secs = {r.name: r.seconds for r in base_data.rows}
    curr_secs = {r.name: r.seconds for r in curr_data.rows}

    flags: list[RegressionFlag] = []
    # Check all buckets that appear in either snapshot.
    all_names = set(base_shares) | set(curr_shares)
    for name in all_names:
        b_share = base_shares.get(name, 0.0)
        c_share = curr_shares.get(name, 0.0)
        b_secs = base_secs.get(name, 0.0)
        c_secs = curr_secs.get(name, 0.0)
        share_delta = c_share - b_share
        secs_delta = c_secs - b_secs
        share_grew = share_delta > share_tolerance
        secs_grew = seconds_tolerance is not None and secs_delta > seconds_tolerance
        if share_grew or secs_grew:
            flags.append(
                RegressionFlag(
                    name=name,
                    baseline_share=b_share,
                    current_share=c_share,
                    share_delta=share_delta,
                    baseline_seconds=b_secs,
                    current_seconds=c_secs,
                    seconds_delta=secs_delta,
                )
            )

    return flags


# ---------------------------------------------------------------------------
# Markdown report rendering
# ---------------------------------------------------------------------------


def _fmt_share(share: float) -> str:
    return f"{share * 100:.1f}%"


def _fact_str(val: Any) -> str:
    if val == "unknown":
        return "unknown"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def render_report(
    data: ReportData,
    regression_flags: list[RegressionFlag] | None = None,
) -> str:
    """Render a docs/research-style Markdown attribution report.

    Parameters
    ----------
    data:
        :class:`ReportData` from :func:`attribute_snapshot`.
    regression_flags:
        Optional list from :func:`compare_snapshots`; included in the
        Regression section when provided.

    Returns the Markdown string.  The caller writes it to disk.
    """
    lines: list[str] = []

    def h(level: int, title: str) -> None:
        lines.append(f"{'#' * level} {title}")
        lines.append("")

    def para(text: str) -> None:
        lines.append(text)
        lines.append("")

    def table_row(*cols: str) -> str:
        return "| " + " | ".join(cols) + " |"

    def table_sep(n: int) -> str:
        return "| " + " | ".join(["---"] * n) + " |"

    # Title
    h(1, f"Attribution Report — `{data.surface}` surface")
    para(
        f"**Surface:** `{data.surface}`  \n"
        f"**Wall denominator:** {data.wall_seconds:.4f}s ({data.total_label})  \n"
        f"**Residual (unbucketed):** {data.residual_seconds:.4f}s"
        f" ({_fmt_share(data.residual_seconds / data.wall_seconds if data.wall_seconds else 0)})"
    )

    # Bucket Ranking
    h(2, "Bucket Ranking")
    if data.rows:
        lines.append(table_row("Bucket", "Seconds", data.pct_label, "Kind"))
        lines.append(table_sep(4))
        for r in data.rows:
            lines.append(
                table_row(
                    f"`{r.name}`",
                    f"{r.seconds:.4f}",
                    _fmt_share(r.share),
                    r.kind,
                )
            )
        lines.append("")
    else:
        para("*(no leaf buckets)*")

    # Residual row (when total parent present)
    if data.residual_seconds > 0:
        residual_share = data.residual_seconds / data.wall_seconds if data.wall_seconds else 0.0
        para(
            f"**Residual** (total - sum of leaves): {data.residual_seconds:.4f}s"
            f" ({_fmt_share(residual_share)})"
        )

    # Dominant path
    h(2, "Dominant Path")
    dp = data.dominant
    para(
        f"**Top bucket:** `{dp.name}` ({_fmt_share(dp.share)}, kind=`{dp.kind}`)  \n"
        f"**GPU share:** {_fmt_share(dp.gpu_share)}  \n"
        f"**CPU share:** {_fmt_share(dp.cpu_share)}  \n"
        f"**Sync share:** {_fmt_share(dp.sync_share)}  \n"
        f"**I/O share:** {_fmt_share(dp.io_share)}  \n"
        f"**Unclassified share:** {_fmt_share(dp.unclassified_share)}"
    )

    # Structural facts
    h(2, "Structural Facts")
    lines.append(table_row("Key", "Value"))
    lines.append(table_sep(2))
    fact_keys = [
        ("N (queries/forward)", "N"),
        ("forwards", "forwards"),
        ("n_images", "n_images"),
        ("mask_logit_hw", "mask_logit_hw"),
        ("forwards_per_image", "forwards_per_image"),
        ("sem_forward_dtype", "sem_forward_dtype"),
        ("K (classes)", "K"),
    ]
    for label, key in fact_keys:
        val = data.facts.get(key, "unknown")
        lines.append(table_row(label, _fact_str(val)))
    lines.append("")

    # GO / NO-GO verdicts
    h(2, "Lever GO/NO-GO Verdicts")
    if data.verdicts:
        lines.append(table_row("Rule", "Verdict", "Reason"))
        lines.append(table_sep(3))
        for v in data.verdicts:
            lines.append(table_row(v.rule_id, f"**{v.verdict}**", v.reason))
        lines.append("")
        h(3, "Citations")
        for v in data.verdicts:
            para(f"- `{v.rule_id}`: {v.citation}")
    else:
        para("*(no verdicts computed)*")

    # Regression section
    h(2, "Regression Detection")
    if regression_flags is None:
        para("*(no baseline snapshot provided — run `compare_snapshots()` to populate)*")
    elif not regression_flags:
        para("No regressions detected above tolerance.")
    else:
        lines.append(
            table_row("Bucket", "Baseline share", "Current share", "Delta share", "Delta seconds")
        )
        lines.append(table_sep(5))
        for f_ in regression_flags:
            lines.append(
                table_row(
                    f"`{f_.name}`",
                    _fmt_share(f_.baseline_share),
                    _fmt_share(f_.current_share),
                    f"+{_fmt_share(f_.share_delta)}",
                    f"{f_.seconds_delta:+.4f}s",
                )
            )
        lines.append("")

    return "\n".join(lines)
