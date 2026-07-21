"""SynthRL port (Shin & Lee, IJCAI-25) -- transformer + reinforcement-learning
synthesizer sound matching, ported as a Layer 3 model family.

One paper, one package, one file per role (grows as the port lands):

- ``representation.py`` -- the SynthRL-local per-parameter **class-index**
  representation: continuous parameters binned into ordinal classes, categorical
  parameters kept native, plus the Gaussian-smoothed cross-entropy targets. Wraps
  (never modifies) the shared :class:`ParameterSpace`.

See ``docs/SYNTHRL_PORT.md`` for the paper->package mapping.
"""
from models.synthrl.representation import SynthRLRepresentation

__all__ = ["SynthRLRepresentation"]
