"""The Evaluator (Layer 4): scores a fitted model on a held-out corpus.

The metric panel (``evaluation.registry``) only *holds* per-sample callables; the
Evaluator is what *consumes* them. Given a fitted :class:`~models.base_model.BaseModel`
and a loaded :class:`~dataset.torch_dataset.RenderedCorpusDataset`, it produces the
project's results table: for every corpus sample it predicts parameters, re-renders
the prediction, runs the whole panel, and aggregates.

Its defining constraint is **D-REPRO**: predictions are re-rendered in a *fresh OS
process at sequence position 0* (:class:`~dataset.render_backends.FreshProcessRenderBackend`),
identically to how the eval corpus's targets were rendered, so Dexed's hidden
per-voice-state leak does not dominate the benchmark. Audio metrics then compare the
re-render against the **stored target WAV** (the target is never re-rendered); the
target and the prediction therefore share an identical clean pos-0 context, so a
perfect prediction floors the audio metrics at ~0.

The render contract (note, velocity, durations, renderer, sample rate, default
parameters) is read **from the corpus's own** ``run_summary.json`` -- never from
``config.py``, which could have drifted and would silently re-render every prediction
under the wrong contract. A missing render field is a hard error, not a fallback.

Each eval run is a self-describing folder mirroring the corpus convention:
``results/<corpus_name>/<model_name>/`` holding ``per_sample.csv`` (the N x M matrix,
``NaN``s intact -- the source of truth for the metric-panel pruning analysis) and
``eval_summary.json`` (render contract echoed from the corpus, checkpoint fingerprint,
and per-metric mean / std / valid-count). The Evaluator both writes these files and
returns the in-memory :class:`EvaluationResult`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.io import wavfile

import config
from dataset.render_backends import FreshProcessRenderBackend, RenderSettings
from dataset.torch_dataset import RenderedCorpusDataset
from evaluation.registry import METRIC_PANEL

# The run_summary.json fields the render contract is reconstructed from. Each is
# required: the Evaluator hard-fails rather than fall back to config.py (which may
# have drifted), since a wrong contract would silently corrupt every re-render.
_REQUIRED_CONTRACT_FIELDS = ("render_settings", "renderer", "sample_rate", "default_params")


@dataclass(frozen=True)
class EvaluationResult:
    """The outcome of one eval run: the per-sample matrix, the summary, and where they live."""

    per_sample_metrics: pd.DataFrame
    summary: Dict[str, object]
    per_sample_metrics_path: Path
    summary_path: Path


class Evaluator:
    """Scores a fitted model on one corpus through the full metric panel.

    The Evaluator only ever calls :meth:`BaseModel.predict` -- never ``fit``
    (training is cluster-side). It owns the :class:`FreshProcessRenderBackend`
    lifecycle: it builds the backend from the corpus's render contract and tears it
    down when :meth:`evaluate` returns.
    """

    def __init__(self, corpus: RenderedCorpusDataset):
        self._corpus = corpus
        self._corpus_summary = self._load_corpus_summary(corpus.corpus_dir)
        self._render_settings = RenderSettings(**self._corpus_summary["render_settings"])
        self._renderer = str(self._corpus_summary["renderer"])
        self._sample_rate = int(self._corpus_summary["sample_rate"])
        self._default_params: Dict[str, float] = {
            name: float(value) for name, value in self._corpus_summary["default_params"].items()
        }

    @staticmethod
    def _load_corpus_summary(corpus_dir: Path) -> Dict[str, object]:
        """Read the corpus run_summary.json and verify it carries the render contract."""
        summary_path = Path(corpus_dir) / "run_summary.json"
        with open(summary_path) as summary_file:
            summary = json.load(summary_file)
        missing = [field for field in _REQUIRED_CONTRACT_FIELDS if field not in summary]
        if missing:
            raise ValueError(
                f"{summary_path} is missing render-contract fields {missing}; the Evaluator "
                "reads the contract from the corpus and never falls back to config.py. "
                "Rebuild this corpus with the current DatasetBuilder."
            )
        return summary

    def evaluate(
        self,
        model,
        *,
        checkpoint_path: Optional[Union[str, Path]] = None,
        out_dir: Optional[Union[str, Path]] = None,
        save_audio: bool = False,
        save_audio_n: int = 20,
        save_audio_seed: int = 0,
    ) -> EvaluationResult:
        """Score ``model`` on the corpus and persist the result.

        Args:
            model: a **fitted** (or loaded) :class:`BaseModel`. Only ``predict`` is
                called.
            checkpoint_path: the file ``model`` was loaded from, hashed into the
                summary's checkpoint fingerprint. ``None`` records a null hash with a
                note (e.g. a model fitted in-memory with no file).
            out_dir: results root; defaults to ``<project>/results``. The run is
                written to ``<out_dir>/<corpus_name>/<model_name>/``.
            save_audio: if ``True``, persist the re-rendered prediction WAV for a
                seeded random subset of samples under ``<run_dir>/audio/`` (D-EVAL
                update), so target vs. prediction can be A/B-played later. Off by
                default -- a full benchmark sweep shouldn't pay to write audio nobody
                listens to.
            save_audio_n: cap on how many samples get their prediction saved. Ignored
                when ``save_audio`` is ``False``.
            save_audio_seed: seed for the random sample selection, so which samples get
                saved is reproducible but not biased by corpus ordering.

        Returns:
            The :class:`EvaluationResult`, whose two files are also written to disk.
        """
        model_name = type(model).__name__
        results_root = Path(out_dir) if out_dir is not None else Path(config.BASE_DIR) / "results"
        run_dir = results_root / self._corpus.corpus_dir.name / model_name
        run_dir.mkdir(parents=True, exist_ok=True)

        audio_sample_indices = (
            self._select_audio_sample_indices(save_audio_n, save_audio_seed) if save_audio else frozenset()
        )
        per_sample_rows = self._score_all_samples(model, run_dir, audio_sample_indices)
        per_sample_metrics = pd.DataFrame(per_sample_rows, columns=["sample_id"] + [spec.name for spec in METRIC_PANEL])

        per_sample_metrics_path = run_dir / "per_sample.csv"
        per_sample_metrics.to_csv(per_sample_metrics_path, index=False)

        summary = self._build_eval_summary(model_name, per_sample_metrics, checkpoint_path)
        summary_path = run_dir / "eval_summary.json"
        with open(summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2)

        return EvaluationResult(per_sample_metrics, summary, per_sample_metrics_path, summary_path)

    # -- per-sample scoring --------------------------------------------------
    def _score_all_samples(self, model) -> List[Dict[str, object]]:
        """Predict + re-render + run the panel for every corpus sample.

        The fresh-process backend is built here and always closed, even if a render
        or metric raises midway.
        """
        target_matrix = self._corpus.targets.numpy()
        backend = FreshProcessRenderBackend(self._render_settings, renderer=self._renderer)
        rows: List[Dict[str, object]] = []
        try:
            for index in range(len(self._corpus)):
                target_audio, _ = self._corpus[index]
                target_waveform = target_audio.numpy()
                target_vector = target_matrix[index]

                predicted_dict = model.predict(target_audio)
                predicted_vector = self._corpus.parameter_space.synth_dict_to_ml_vector(predicted_dict)
                prediction_waveform = backend.render({**self._default_params, **predicted_dict})

                row: Dict[str, object] = {"sample_id": self._corpus.metadata.iloc[index]["sample_id"]}
                for spec in METRIC_PANEL:
                    if spec.input_type == "audio":
                        row[spec.name] = spec.compute(
                            target_waveform, prediction_waveform, sample_rate=self._sample_rate
                        )
                    else:
                        row[spec.name] = spec.compute(
                            target_vector, predicted_vector, self._corpus.parameter_space
                        )
                rows.append(row)
        finally:
            backend.close()
        return rows

    # -- aggregation + summary -----------------------------------------------
    def _build_eval_summary(
        self,
        model_name: str,
        per_sample_metrics: pd.DataFrame,
        checkpoint_path: Optional[Union[str, Path]],
    ) -> Dict[str, object]:
        """Aggregate the per-sample matrix and assemble the self-describing summary."""
        per_metric: Dict[str, Dict[str, object]] = {}
        for spec in METRIC_PANEL:
            values = per_sample_metrics[spec.name].to_numpy(dtype=float)
            valid = values[~np.isnan(values)]
            per_metric[spec.name] = {
                "mean": float(np.mean(valid)) if valid.size else float("nan"),
                "std": float(np.std(valid)) if valid.size else float("nan"),
                "valid_count": int(valid.size),
                "higher_is_better": spec.higher_is_better,
            }

        return {
            "model_class": model_name,
            "checkpoint": _checkpoint_fingerprint(checkpoint_path),
            "corpus": {
                "path": str(self._corpus.corpus_dir),
                "name": self._corpus.corpus_dir.name,
                "git_revision": self._corpus_summary.get("git_revision"),
            },
            "render_contract": {
                "render_settings": self._corpus_summary["render_settings"],
                "renderer": self._renderer,
                "sample_rate": self._sample_rate,
            },
            "metrics": [spec.name for spec in METRIC_PANEL],
            "per_metric": per_metric,
            "num_samples": int(len(per_sample_metrics)),
            "git_revision": _git_revision(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def _checkpoint_fingerprint(checkpoint_path: Optional[Union[str, Path]]) -> Dict[str, object]:
    """A sha256 fingerprint of the checkpoint file, or a null hash with a note."""
    if checkpoint_path is None:
        return {"path": None, "sha256": None, "note": "no checkpoint file (model fitted in-memory)"}
    path = Path(checkpoint_path)
    if not path.exists():
        return {"path": str(path), "sha256": None, "note": "checkpoint file not found at eval time"}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "note": None}


def _git_revision() -> Optional[str]:
    """The current commit of the framework repo, or ``None`` outside a checkout."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(config.BASE_DIR),
            stderr=subprocess.DEVNULL,
        )
        return revision.decode().strip()
    except Exception:
        return None
