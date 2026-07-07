"""The single source of truth for which model families exist (Layer 3).

Analogous to :data:`evaluation.registry.METRIC_PANEL`: one table that names every
model family and maps it to the class the fit/eval scripts construct. Adding a new
family (search-based, generative, ...) is one entry here -- it then becomes both
trainable (``scripts/fit_model.py --model``) and loadable at eval time
(``scripts/evaluate.py --model``) with no other wiring.

Each entry also carries the default checkpoint filename the fit script writes when
``--out`` is omitted, because families use different serialization formats (the
baseline saves JSON, deep families save a torch artifact).

Importing this pulls in ``torch`` (via the deep families), so it stays on the
CLI/pipeline side. The dashboard never imports it -- it mirrors the model names as
plain strings instead, to keep the "dashboard never imports the pipeline" rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from models.base_model import BaseModel
from models.mean_parameter_baseline import MeanParameterBaseline
from models.sound2synth import Sound2SynthSpectrogramRegressor


@dataclass(frozen=True)
class ModelRegistration:
    """A model family: the class to construct and its default checkpoint filename."""

    model_class: Type[BaseModel]
    default_checkpoint_filename: str


# Name -> registration. The name is the public identifier used on the ``--model``
# flag of both fit and eval scripts and mirrored (as a plain string) in the dashboard.
MODEL_REGISTRY: Dict[str, ModelRegistration] = {
    "MeanParameterBaseline": ModelRegistration(
        MeanParameterBaseline, "mean_parameter_baseline.json"
    ),
    "Sound2SynthSpectrogramRegressor": ModelRegistration(
        Sound2SynthSpectrogramRegressor, "spectrogram_cnn.pt"
    ),
}
