"""Layer 3 -- deep-model training harness (issue #22).

The reusable orchestration layer every deep family shares: a typed
:class:`~models.training.config.TrainingConfig`, a :class:`ParameterLoss` routed
by the corpus's :class:`~synth.parameter_space.ParameterSpace`, a
:class:`CorpusDataModule`, a generic :class:`LightningRegressor` LightningModule, a
``build_trainer`` factory, and the plain ``torch`` checkpoint I/O consumed by
``BaseModel.load``.

**PyTorch Lightning lives only here** (D-FRAMEWORK). The Mac-side eval path
(``BaseModel.load`` / ``predict`` via :class:`~models.base_deep_model.BaseDeepModel`)
imports nothing from this package, so Lightning never becomes a Mac dependency
(D-SELFDESC / D-EVAL). ``config.py``, ``loss.py`` and ``checkpoint.py`` are pure
``torch`` (no Lightning); ``data_module.py``, ``lightning_module.py`` and
``trainer_factory.py`` import ``lightning`` and are used only during training.
"""
