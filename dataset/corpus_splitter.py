"""Split an already-rendered corpus into train/test corpora (post-render split).

Building splits presets *before* rendering (``dexed_preset_loader``); this module
splits a corpus that is *already* rendered, so a held-out test set can be carved
out of an existing corpus without re-rendering everything (e.g. the ~30k-sample
preset-gen corpus, built all-train in-process). See D-SPLIT in docs/DECISIONS.md.

The split is a seeded row-partition (``split_indices``, identical determinism to
the build-time split). The **train** partition is copied verbatim -- its render
context is irrelevant to training. The **test** partition must be re-rendered in
fresh processes at position 0 (D-REPRO), which needs the VST, so that half lives
in ``scripts/split_corpus.py`` via :class:`~dataset.preset_sources.CorpusPresetSource`
and the ``DatasetBuilder``. Everything here is VST-free and unit-testable.

Hybrid corpora are refused: their augmented children (and repeated blend parents)
would straddle train and test -- train/test leakage. Split their human source
cartridges at build time instead.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config


def load_corpus(corpus_dir: Path) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Read a corpus's ``run_summary.json`` and ``metadata.csv``."""
    corpus_dir = Path(corpus_dir)
    summary = json.loads((corpus_dir / "run_summary.json").read_text())
    df_metadata = pd.read_csv(corpus_dir / "metadata.csv")
    return summary, df_metadata


def source_method(summary: Dict[str, object]) -> Optional[str]:
    """The construction method a corpus was built with (``source.method``)."""
    source = summary.get("source") or {}
    return source.get("method") if isinstance(source, dict) else None


def assert_splittable(summary: Dict[str, object]) -> None:
    """Refuse hybrid corpora (train/test leakage); other methods are fine."""
    if source_method(summary) == "hybrid":
        raise ValueError(
            "Refusing to split a hybrid corpus: its augmented children and repeated "
            "blend parents would straddle train and test (train/test leakage). Split "
            "the human source cartridges at build time instead (build_dataset.py human)."
        )


def split_source_description(
    summary: Dict[str, object],
    partition: str,
    split_from: str,
    test_fraction: float,
    split_seed: int,
    count: int,
) -> Dict[str, object]:
    """The run-summary ``source`` block for one split partition.

    Built in one place so both the copied-train and re-rendered-test summaries
    carry an identical provenance shape; ``method`` preserves the source corpus's
    construction method.
    """
    return {
        "method": source_method(summary),
        "partition": partition,
        "split_from": split_from,
        "split_test_fraction": float(test_fraction),
        "split_seed": int(split_seed),
        "count": int(count),
    }


def clean_records(df_partition: pd.DataFrame) -> List[Dict[str, object]]:
    """Metadata rows as plain dicts with NaN coerced to ``None`` (for CorpusPresetSource).

    Casts to ``object`` first: ``None`` can't survive in a float column, so the cast
    is what lets missing provenance (empty ``voice_index`` / ``parent_id``) come back
    as ``None`` rather than ``NaN``.
    """
    cleaned = df_partition.astype(object).where(pd.notna(df_partition), None)
    return cleaned.to_dict(orient="records")


def _method_counts(df_partition: pd.DataFrame) -> Dict[str, int]:
    return {str(method): int(count) for method, count in df_partition["method"].value_counts().items()}


def derive_summary(
    source_summary: Dict[str, object],
    run_name: str,
    df_partition: pd.DataFrame,
    render_process: str,
    description: Dict[str, object],
) -> Dict[str, object]:
    """A partition's ``run_summary.json``: the source summary with the per-run fields
    overridden. Keeps ``parameter_space`` / ``render_settings`` / ``subset_names`` /
    ``default_params`` / ``sample_rate`` / ``renderer`` intact so the corpus stays
    self-describing (D-SELFDESC)."""
    near_silent_count = (
        int(df_partition["near_silent"].sum()) if "near_silent" in df_partition else 0
    )
    summary = dict(source_summary)
    summary["run_name"] = run_name
    summary["num_samples"] = int(len(df_partition))
    summary["near_silent_count"] = near_silent_count
    summary["method_counts"] = _method_counts(df_partition)
    summary["render_process"] = render_process
    summary["source"] = description
    summary["git_revision"] = _git_revision()
    return summary


def write_copied_partition(
    source_dir: Path,
    df_partition: pd.DataFrame,
    out_dir: Path,
    source_summary: Dict[str, object],
    description: Dict[str, object],
    partition: str = "train",
) -> Dict[str, object]:
    """Write ``df_partition`` as a corpus at ``out_dir`` by copying its WAVs verbatim.

    Re-indexes samples to ``sample_000000...`` in row order, rewrites ``sample_id`` /
    ``audio_path`` / ``partition``, preserves every other column, and derives a
    self-describing ``run_summary.json``. The copied audio keeps the source's render
    process. Returns the run-summary dict.
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=False)

    original_paths = df_partition["audio_path"].tolist()
    df_out = df_partition.copy().reset_index(drop=True)
    sample_ids = [f"sample_{index:06d}" for index in range(len(df_out))]
    df_out["sample_id"] = sample_ids
    df_out["audio_path"] = [f"audio/{sample_id}.wav" for sample_id in sample_ids]
    df_out["partition"] = partition

    for original_path, sample_id in zip(original_paths, sample_ids):
        shutil.copyfile(source_dir / original_path, audio_dir / f"{sample_id}.wav")

    df_out.to_csv(out_dir / "metadata.csv", index=False)

    render_process = str(source_summary.get("render_process", "in-process"))
    summary = derive_summary(source_summary, out_dir.name, df_out, render_process, description)
    with open(out_dir / "run_summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    return summary


def _git_revision() -> Optional[str]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(config.BASE_DIR),
            stderr=subprocess.DEVNULL,
        )
        return revision.decode().strip()
    except Exception:
        return None
