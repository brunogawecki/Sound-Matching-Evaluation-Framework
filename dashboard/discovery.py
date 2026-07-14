"""Scan the framework's on-disk artefacts to populate dashboard dropdowns.

All reads only -- these list what the scripts have already produced so the
Split corpus / Train on cluster / Evaluate / Results pages can reference real
corpora, checkpoints, and result runs instead of asking the user to type paths.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import config

DATASET_DIR = Path(config.DATASET_DIR)
CHECKPOINTS_DIR = Path(config.BASE_DIR) / "checkpoints"
RESULTS_DIR = Path(config.BASE_DIR) / "results"


@dataclass(frozen=True)
class Corpus:
    name: str
    path: Path
    num_samples: Optional[int]
    fresh_process: bool  # True if any partition rendered fresh-process (eval-ready)
    method: Optional[str]  # construction method ("synthetic" | "human" | "hybrid" | ...)


@dataclass(frozen=True)
class ResultRun:
    corpus: str
    model: str
    summary_path: Path
    per_sample_path: Path


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def list_corpora() -> List[Corpus]:
    """Every ``<DATASET_DIR>/<name>/`` that carries a ``run_summary.json``."""
    corpora: List[Corpus] = []
    if not DATASET_DIR.exists():
        return corpora
    for entry in sorted(DATASET_DIR.iterdir()):
        summary_file = entry / "run_summary.json"
        if not (entry.is_dir() and summary_file.is_file()):
            continue
        summary = _read_json(summary_file)
        # run_summary.json records how it was rendered ("in-process" | "fresh").
        # Fresh-process corpora are the eval-ready ones (D-REPRO).
        fresh = str(summary.get("render_process", "")).lower().startswith("fresh")
        source = summary.get("source") or {}
        method = source.get("method") if isinstance(source, dict) else None
        corpora.append(
            Corpus(
                name=entry.name,
                path=entry,
                num_samples=summary.get("num_samples"),
                fresh_process=fresh,
                method=method,
            )
        )
    return corpora


def list_checkpoints() -> List[Path]:
    """Saved model checkpoints under ``checkpoints/``.

    Covers both layouts: flat files (locally trained models, and pulls that fell
    back to the pre-job-scoping path) and one level of nesting, which is where a
    job-scoped pull puts them (``checkpoints/<job_id>/<model>.pt``).
    """
    if not CHECKPOINTS_DIR.exists():
        return []
    found: List[Path] = []
    for extension in ("json", "ckpt", "pt"):
        found.extend(CHECKPOINTS_DIR.glob(f"*.{extension}"))
        found.extend(CHECKPOINTS_DIR.glob(f"*/*.{extension}"))
    return sorted(found)


def list_result_runs() -> List[ResultRun]:
    """Every ``results/<corpus>/<model>/`` holding an ``eval_summary.json``."""
    runs: List[ResultRun] = []
    if not RESULTS_DIR.exists():
        return runs
    for corpus_dir in sorted(RESULTS_DIR.iterdir()):
        if not corpus_dir.is_dir():
            continue
        for model_dir in sorted(corpus_dir.iterdir()):
            summary = model_dir / "eval_summary.json"
            if model_dir.is_dir() and summary.is_file():
                runs.append(
                    ResultRun(
                        corpus=corpus_dir.name,
                        model=model_dir.name,
                        summary_path=summary,
                        per_sample_path=model_dir / "per_sample.csv",
                    )
                )
    return runs


def load_summary(path: Path) -> Dict:
    """Read an eval_summary.json / run_summary.json (``{}`` on failure)."""
    return _read_json(path)


def list_saved_audio_samples(corpus: str, model: str) -> List[str]:
    """Sample ids with a saved prediction WAV for one eval run (``--save-audio``).

    Sorted stems of ``results/<corpus>/<model>/audio/*.wav`` -- empty if the run
    predates ``--save-audio`` or was evaluated without it (D-EVAL update).
    """
    audio_dir = RESULTS_DIR / corpus / model / "audio"
    if not audio_dir.exists():
        return []
    return sorted(path.stem for path in audio_dir.glob("*.wav"))


def original_audio_path(corpus: str, sample_id: str) -> Path:
    """Where a corpus sample's target WAV lives (``dataset/<corpus>/audio/<id>.wav``)."""
    return DATASET_DIR / corpus / "audio" / f"{sample_id}.wav"


def predicted_audio_path(corpus: str, model: str, sample_id: str) -> Path:
    """Where an eval run's saved prediction WAV lives, if ``--save-audio`` wrote one."""
    return RESULTS_DIR / corpus / model / "audio" / f"{sample_id}.wav"
