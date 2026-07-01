"""LightningDataModule over a rendered corpus (issue #22).

Wraps a :class:`~dataset.torch_dataset.RenderedCorpusDataset` (which already emits
``(audio, target)`` pairs -- raw fixed-length waveform per D-REPR, ML-side target
vector) in the DataLoader plumbing Lightning expects. Default collation is correct
because every sample is the same length on both sides.

Validation source, in priority order:
1. an explicit ``validation_dataset`` (``fit`` already accepts one); else
2. a seeded sample-level split carved from the train corpus when
   ``data_config.val_fraction`` is set; else
3. no validation (``val_dataloader`` returns nothing).

The held-out human test set is never used here -- training-time validation is for
model selection only; final scoring is the Evaluator's job (D-EVAL, Phase 6).

Imports Lightning: a training-only module, never touched by the Mac eval path.
"""
from __future__ import annotations

from typing import List, Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from dataset.torch_dataset import RenderedCorpusDataset
from models.training.config import DataConfig


class CorpusDataModule(pl.LightningDataModule):
    """Train (+ optional validation) DataLoaders over rendered corpora."""

    def __init__(
        self,
        train_dataset: RenderedCorpusDataset,
        validation_dataset: Optional[RenderedCorpusDataset] = None,
        data_config: Optional[DataConfig] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._full_train_dataset = train_dataset
        self._explicit_validation_dataset = validation_dataset
        self._data_config = data_config or DataConfig()
        self._seed = seed
        self._train_split: Dataset = train_dataset
        self._val_split: Optional[Dataset] = validation_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if self._explicit_validation_dataset is not None:
            self._train_split = self._full_train_dataset
            self._val_split = self._explicit_validation_dataset
            return

        val_fraction = self._data_config.val_fraction
        if val_fraction is None:
            self._train_split = self._full_train_dataset
            self._val_split = None
            return

        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"data.val_fraction must be in (0, 1), got {val_fraction}.")
        total = len(self._full_train_dataset)
        val_size = int(round(total * val_fraction))
        if not 0 < val_size < total:
            raise ValueError(
                f"val_fraction={val_fraction} on a corpus of {total} samples yields a "
                f"degenerate split (val_size={val_size}); pick a fraction that leaves both "
                "splits non-empty."
            )
        lengths: List[int] = [total - val_size, val_size]
        generator = torch.Generator().manual_seed(self._seed)
        self._train_split, self._val_split = random_split(
            self._full_train_dataset, lengths, generator=generator
        )

    @property
    def has_validation(self) -> bool:
        return self._val_split is not None

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_split, shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self._val_split is None:
            return None
        return self._make_loader(self._val_split, shuffle=False)

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        config = self._data_config
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers and config.num_workers > 0,
        )
