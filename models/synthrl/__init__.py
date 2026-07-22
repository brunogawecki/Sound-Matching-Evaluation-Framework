"""SynthRL port (Shin & Lee, IJCAI-25) -- transformer + reinforcement-learning
synthesizer sound matching, ported as a Layer 3 model family.

One paper, one package, one file per role (grows as the port lands):

- ``representation.py`` -- the SynthRL-local per-parameter **class-index**
  representation: continuous parameters binned into ordinal classes, categorical
  parameters kept native, plus the Gaussian-smoothed cross-entropy targets. Wraps
  (never modifies) the shared :class:`ParameterSpace`.
- ``network.py`` -- :class:`SynthRLNetwork`: mel front-end -> strided conv reducer ->
  transformer encoder -> DETR decoder (one query per parameter) -> per-parameter class
  heads. Emits the flat class-logit vector the representation lays out.
- ``families.py`` -- the benchmark wrappers. :class:`SynthRLp` (stage 1, parameter loss)
  and :class:`SynthRLi` (stage 2, in-domain RL, warm-started from ``-p``).
- ``lightning_module.py`` -- the two training recipes (Gaussian-smoothed per-parameter
  cross-entropy; REINFORCE + reward PER curriculum). Training-only, imported lazily.
- ``reward.py`` / ``reward_buffer.py`` -- the RL stage's audio-similarity reward and its
  per-target reward-based prioritized replay buffer.

See ``docs/SYNTHRL_PORT.md`` for the paper->package mapping.
"""
from models.synthrl.families import BaseSynthRLModel, SynthRLi, SynthRLp
from models.synthrl.network import SynthRLNetwork
from models.synthrl.representation import SynthRLRepresentation

__all__ = ["BaseSynthRLModel", "SynthRLp", "SynthRLi", "SynthRLNetwork", "SynthRLRepresentation"]
