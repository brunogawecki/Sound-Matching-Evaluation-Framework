"""Out-of-domain (audio-only) corpora and the evaluation-corpus loader (Layer 2).

An **out-of-domain corpus** holds target audio the synthesizer never produced -- NSynth
instrument notes, for instance -- so it carries no ground-truth parameters. Everything
else about it is an ordinary corpus: the same ``audio/`` + ``metadata.csv`` +
``run_summary.json`` layout, read by the same Evaluator, written to the same
``results/<corpus>/<model>/``.

The one conceptual shift is what the render contract in ``run_summary.json`` describes.
On a rendered corpus it describes how the targets were made *and* how predictions must be
re-rendered. Here the targets were never rendered at all, so the contract describes **the
prediction render only** -- which is why ``scripts/build_ood_corpus.py`` copies it verbatim
from an in-domain reference corpus rather than inventing one.

Consequences, both deliberate (see D-OOD in ``docs/DECISIONS.md``):

- :attr:`AudioOnlyCorpusDataset.targets` is ``None``, and the Evaluator reports ``NaN``
  for the three parameter metrics -- the panel's established "undefined for this sample"
  convention, surfaced by the ``valid_count`` it already prints beside every mean.
- The audio metrics no longer floor at ~0 for a perfect prediction, because the target is
  generally unreachable by the synth. Out-of-domain scores rank models against each other;
  they are not absolute fidelity numbers.

The :class:`~synth.parameter_space.ParameterSpace` is still needed (the Evaluator encodes
each *prediction* through it), so it is still carried in the summary and still rebuilt with
no live VST (D-SELFDESC).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.torch_dataset import (
    RenderedCorpusDataset,
    _expected_num_samples,
    read_corpus_audio,
)
from synth.parameter_space import ParameterSpace

# run_summary.json marker distinguishing an audio-only corpus from a rendered one.
NO_TARGETS = "none"


class AudioOnlyCorpusDataset(Dataset):
    """A corpus of target audio with no ground-truth parameters.

    Exposes the surface :class:`~evaluation.evaluator.Evaluator` consumes --
    :attr:`corpus_dir`, :attr:`metadata`, :attr:`parameter_space`, :attr:`targets`,
    ``__len__`` and ``__getitem__`` -- so it drops into the eval path unchanged.

    Args:
        corpus_dir: the corpus directory, containing ``metadata.csv`` and ``audio/``.
        parameter_space: the space predictions are encoded through. It comes from the
            in-domain reference corpus (copied into this corpus's ``run_summary.json``
            at build time), not from the targets, which have no parameters.
        expected_num_samples: the audio length every WAV is checked against. An
            out-of-domain corpus is built to the reference contract's exact length, so a
            mismatch is still a corrupt file. ``None`` disables the check.
    """

    def __init__(
        self,
        corpus_dir: Union[str, Path],
        parameter_space: ParameterSpace,
        expected_num_samples: Optional[int] = None,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.parameter_space = parameter_space
        self.expected_num_samples = expected_num_samples
        self.metadata = pd.read_csv(self.corpus_dir / "metadata.csv")

    @classmethod
    def load(cls, corpus_dir: Union[str, Path]) -> "AudioOnlyCorpusDataset":
        """Load an audio-only corpus, parameter space and all (no VST needed)."""
        corpus_dir = Path(corpus_dir)
        with open(corpus_dir / "run_summary.json") as summary_file:
            summary = json.load(summary_file)
        if "parameter_space" not in summary:
            raise ValueError(
                f"{corpus_dir / 'run_summary.json'} has no 'parameter_space'. An "
                "out-of-domain corpus copies one from its reference corpus; rebuild it "
                "with scripts/build_ood_corpus.py."
            )
        parameter_space = ParameterSpace.from_dict(summary["parameter_space"])
        return cls(corpus_dir, parameter_space, _expected_num_samples(summary))

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, None]:
        """The target audio and ``None`` -- there is no target vector to return."""
        relative_path = self.metadata.iloc[index]["audio_path"]
        audio = read_corpus_audio(self.corpus_dir / relative_path, self.expected_num_samples)
        return audio, None

    @property
    def targets(self) -> None:
        """``None``: out-of-domain targets have no parameters (D-OOD)."""
        return None


def load_eval_corpus(
    corpus_dir: Union[str, Path],
) -> Union[RenderedCorpusDataset, AudioOnlyCorpusDataset]:
    """Load a corpus for evaluation, picking the flavour from its own summary.

    Returns an :class:`AudioOnlyCorpusDataset` when ``run_summary.json`` marks the corpus
    as carrying no targets, otherwise a
    :class:`~dataset.torch_dataset.RenderedCorpusDataset`. This is the single entry point
    ``scripts/evaluate.py`` uses, so an out-of-domain run needs no extra flag -- the corpus
    describes itself (D-SELFDESC).
    """
    corpus_dir = Path(corpus_dir)
    with open(corpus_dir / "run_summary.json") as summary_file:
        summary = json.load(summary_file)
    if summary.get("targets") == NO_TARGETS:
        return AudioOnlyCorpusDataset.load(corpus_dir)
    return RenderedCorpusDataset.load(corpus_dir)
