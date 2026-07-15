"""InverSynth II port (Barkan et al., ISMIR 2023; paper_repos/InverSynth2).

Fills the benchmark's **neural-proxy** model family -- a peer paper approach alongside the
discriminative (Sound2Synth) and generative (preset-gen-vae) families. The paper stacks three
models, built in stages under the paper's own names (note the "xITF = *excluding* ITF" trap):

- ``IS``      -- encoder, parameters-loss only (Stage 1, done).
- ``IS2xITF`` -- ``IS`` + a training-only differentiable synthesizer-proxy + audio loss, no
  inference-time finetuning (Stage 2).
- ``IS2``     -- ``IS2xITF`` + per-sample inference-time finetuning (Stage 3).

One paper, one package, one file per role:

- ``network.py``  -- the mel-dB front-end (reused from the preset-gen-vae port) and the ported
  strided-CNN encoder (:class:`InverSynthEncoderNetwork`). Later stages add the proxy decoder.
- ``families.py`` -- the benchmark wrappers (:class:`IS`; later ``IS2xITF`` / ``IS2``).
- ``lightning_module.py`` (Stage 2+) -- the training-only proxy loss recipe. It will stay behind
  the families' lazy import so the eval path needs no Lightning (D-FRAMEWORK), and so is never
  imported here.
"""
from models.inversynth2.families import IS
from models.inversynth2.network import InverSynthEncoderNetwork

__all__ = [
    "IS",
    "InverSynthEncoderNetwork",
]
