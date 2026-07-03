"""Layer 3 -- sound-matching models.

``BaseModel`` is the contract every family implements; ``BaseDeepModel`` adds the
shared ``save``/``load``/``predict`` for deep families (Lightning-free, eval-path
safe); ``MeanParameterBaseline`` is the trivial floor used to validate the
end-to-end pipeline; ``SpectrogramConvolutionalRegressor`` is the first real deep
family (issue #19, Sound2Synth lineage).
"""
from models.base_deep_model import BaseDeepModel
from models.base_model import BaseModel
from models.mean_parameter_baseline import MeanParameterBaseline
from models.Sound2Synth import SpectrogramConvolutionalRegressor

__all__ = [
    "BaseModel",
    "BaseDeepModel",
    "MeanParameterBaseline",
    "SpectrogramConvolutionalRegressor",
]
