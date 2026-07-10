"""preset-gen-vae port (Le Vaillant et al., DAFx 2021; paper_repos/preset-gen-vae).

A staged reimplementation of the paper's ``FlVAE2`` model as ``BaseDeepModel``
families that predict this framework's D1 parameter space through ``ParameterSpace``
(not the paper's 144-param / ``all<=32`` scheme), so they stay comparable to the other
model families and train through the existing harness and ``ParameterLoss``.
The stage map lives in ``docs/PRESETGEN_VAE_PORT.md``.

One paper, one package, one file per role:

- ``network.py`` -- the mel-dB front-end and the paper's ``speccnn8l1_bn`` VAE
  (:class:`PresetGenVAENetwork`), plus the D-MELNORM corpus measurement.
- ``realnvp.py`` -- the paper's ``CustomRealNVP`` flow, ported as plain torch.
- ``families.py`` -- the benchmark wrappers: :class:`PresetGenVAEMLPRegressor` and
  :class:`PresetGenVAEFlowRegressor` (the paper's MLP-vs-Flow regression comparison).
- ``lightning_module.py`` -- the training-only VAE loss recipe
  (``LightningVAERegressor``). Never imported here: it stays behind the families'
  lazy import so the eval path needs no Lightning (D-FRAMEWORK).
"""
from models.presetgen_vae.families import (
    BasePresetGenVAERegressor,
    PresetGenVAEFlowRegressor,
    PresetGenVAEMLPRegressor,
)
from models.presetgen_vae.network import (
    PresetGenVAENetwork,
    VAENetworkOutput,
    measure_corpus_mel_db_range,
)
from models.presetgen_vae.realnvp import RealNVP

__all__ = [
    "BasePresetGenVAERegressor",
    "PresetGenVAEFlowRegressor",
    "PresetGenVAEMLPRegressor",
    "PresetGenVAENetwork",
    "RealNVP",
    "VAENetworkOutput",
    "measure_corpus_mel_db_range",
]
