"""Sound2Synth-lineage models (Chen et al., 2022; github.com/Sound2Synth/Sound2Synth).

The framework's first real deep family lives here: ``SpectrogramConvolutionalRegressor``
recreates Sound2Synth's STFT ``ConvBackbone`` (its ``main`` branch) but emits through
this framework's own ``ParameterSpace`` contract. Future Sound2Synth variants (e.g. the
paper's oscillator-attention head) belong in this package too.

The module stays importable without Lightning (the training-only dependency is imported
lazily inside ``fit``), so importing this package on the eval path is safe (D-FRAMEWORK).
"""
from models.Sound2Synth.spectrogram_convolutional_regressor import (
    SpectrogramConvolutionalNetwork,
    SpectrogramConvolutionalRegressor,
)

__all__ = ["SpectrogramConvolutionalNetwork", "SpectrogramConvolutionalRegressor"]
