"""Layer 3 -- sound-matching models.

``BaseModel`` is the contract every family implements; ``BaseDeepModel`` adds the
shared ``save``/``load``/``predict`` for deep families (Lightning-free, eval-path
safe); ``MeanParameterBaseline`` is the trivial floor used to validate the
end-to-end pipeline.
"""
from models.base_deep_model import BaseDeepModel
from models.base_model import BaseModel
from models.mean_parameter_baseline import MeanParameterBaseline

__all__ = ["BaseModel", "BaseDeepModel", "MeanParameterBaseline"]
