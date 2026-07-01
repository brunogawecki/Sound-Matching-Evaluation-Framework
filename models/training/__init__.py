"""Layer 3 -- deep-model training harness.

The orchestration layer every deep family shares: a typed
:class:`~models.training.config.TrainingConfig`, a :class:`ParameterLoss`, a
:class:`CorpusDataModule`, a generic :class:`LightningRegressor`, a ``build_trainer``
factory, and plain ``torch`` checkpoint I/O. PyTorch Lightning is imported only by
this package's training modules, never by the eval path.
"""
