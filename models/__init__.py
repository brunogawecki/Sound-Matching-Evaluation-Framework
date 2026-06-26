"""Layer 3 -- sound-matching models.

``BaseModel`` is the contract every family implements; ``MeanParameterBaseline``
is the trivial floor used to validate the end-to-end pipeline.
"""
from models.base_model import BaseModel
from models.mean_parameter_baseline import MeanParameterBaseline

__all__ = ["BaseModel", "MeanParameterBaseline"]
