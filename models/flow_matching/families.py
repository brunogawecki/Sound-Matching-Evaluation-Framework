"""The flow-matching benchmark families: ``BaseDeepModel`` wrappers around
:class:`FlowMatchingNetwork`.

The conditional-generative family from Hayes et al. (ISMIR 2025, "Audio Synthesizer
Inversion in Symmetric Parameter Spaces with Approximately Equivariant Flow
Matching"): instead of regressing parameters, sample them from a continuous
normalizing flow ``p(params | audio)`` trained by flow matching. Stage 1 lands the
paper's CNF (MLP) model -- :class:`FlowMatchingMLP`, the non-equivariant residual-MLP
vector field; the equivariant CNF (Param2Tok) family follows on the same base.

Unlike the regression families, ``predict`` cannot be a single forward pass: it draws
one *sample* by integrating the learned ODE (CFG-guided RK4, the paper's test-time
protocol: 200 steps, guidance strength 2). The draw is seeded per call, so a model's
prediction for a clip is deterministic and the Evaluator's re-render is reproducible.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import torch
from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.flow_matching.network import FlowMatchingNetwork, measure_corpus_mel_db_statistics
from models.training.config import TrainingConfig
from models.training.loss import ParameterLoss
from synth.parameter_space import ParameterSpace


class BaseFlowMatchingModel(BaseDeepModel):
    """Shared :class:`BaseDeepModel` family wrapping :class:`FlowMatchingNetwork`.

    Constructor knobs default to the paper's Surge configs (``surge_flowmlp.yaml`` /
    ``encoder/ast.yaml``). ``ml_dimension``, ``num_audio_samples`` and ``sample_rate``
    are read from the corpus at ``fit`` time (D-SELFDESC), and the mel-dB
    standardization statistics are measured over the train corpus, so ``load``
    rebuilds the exact network offline. Concrete families pin the vector field via
    ``_vector_field_architecture``. Not registered itself.
    """

    _vector_field_architecture: str

    def __init__(
        self,
        n_mels: int = 128,
        window_duration_ms: float = 25.0,
        hop_duration_ms: float = 10.0,
        encoder_d_model: int = 512,
        encoder_num_heads: int = 8,
        encoder_num_layers: int = 8,
        patch_size: int = 16,
        patch_stride: int = 10,
        field_d_model: int = 768,
        field_num_layers: int = 9,
        field_num_heads: int = 8,
        num_parameter_tokens: int = 128,
        projection_penalty: float = 0.01,
        time_encoding_dimension: int = 256,
        cfg_dropout_rate: float = 0.1,
        rectified_sigma_min: float = 0.0,
        ot_pairing: bool = True,
        sample_steps: int = 200,
        sample_cfg_strength: float = 2.0,
        validation_sample_steps: int = 50,
        validation_cfg_strength: float = 2.0,
        predict_seed: int = 0,
        default_root_dir: str = "lightning_logs",
    ) -> None:
        super().__init__(default_root_dir=default_root_dir)
        self._n_mels = n_mels
        self._window_duration_ms = window_duration_ms
        self._hop_duration_ms = hop_duration_ms
        self._encoder_d_model = encoder_d_model
        self._encoder_num_heads = encoder_num_heads
        self._encoder_num_layers = encoder_num_layers
        self._patch_size = patch_size
        self._patch_stride = patch_stride
        self._field_d_model = field_d_model
        self._field_num_layers = field_num_layers
        self._field_num_heads = field_num_heads
        self._num_parameter_tokens = num_parameter_tokens
        self._projection_penalty = projection_penalty
        self._time_encoding_dimension = time_encoding_dimension
        self._cfg_dropout_rate = cfg_dropout_rate
        self._rectified_sigma_min = rectified_sigma_min
        self._ot_pairing = ot_pairing
        self._sample_steps = sample_steps
        self._sample_cfg_strength = sample_cfg_strength
        self._validation_sample_steps = validation_sample_steps
        self._validation_cfg_strength = validation_cfg_strength
        self._predict_seed = int(predict_seed)

    def _num_conditioning_outputs(self) -> int:
        """One conditioning token per vector-field layer (the paper's encoder configs)."""
        return self._field_num_layers

    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        return FlowMatchingNetwork(**architecture_hparams)

    @staticmethod
    def _corpus_sample_rate(train_dataset: RenderedCorpusDataset) -> int:
        """Read the render sample rate from the corpus's ``run_summary.json``."""
        with open(train_dataset.corpus_dir / "run_summary.json") as summary_file:
            return int(json.load(summary_file)["sample_rate"])

    def _build_architecture_hparams(
        self, train_dataset: RenderedCorpusDataset, parameter_space: ParameterSpace
    ) -> Dict[str, Any]:
        example_audio, _ = train_dataset[0]
        sample_rate = self._corpus_sample_rate(train_dataset)
        mel_mean_db, mel_std_db = measure_corpus_mel_db_statistics(
            train_dataset,
            sample_rate=sample_rate,
            n_mels=self._n_mels,
            window_duration_ms=self._window_duration_ms,
            hop_duration_ms=self._hop_duration_ms,
        )
        return {
            "ml_dimension": parameter_space.ml_dimension,
            "num_audio_samples": int(example_audio.shape[-1]),
            "sample_rate": sample_rate,
            "n_mels": self._n_mels,
            "window_duration_ms": self._window_duration_ms,
            "hop_duration_ms": self._hop_duration_ms,
            "mel_mean_db": mel_mean_db,
            "mel_std_db": mel_std_db,
            "encoder_d_model": self._encoder_d_model,
            "encoder_num_heads": self._encoder_num_heads,
            "encoder_num_layers": self._encoder_num_layers,
            "num_conditioning_outputs": self._num_conditioning_outputs(),
            "patch_size": self._patch_size,
            "patch_stride": self._patch_stride,
            "vector_field_architecture": self._vector_field_architecture,
            "field_d_model": self._field_d_model,
            "field_num_layers": self._field_num_layers,
            "field_num_heads": self._field_num_heads,
            "num_parameter_tokens": self._num_parameter_tokens,
            "projection_penalty": self._projection_penalty,
            "time_encoding_dimension": self._time_encoding_dimension,
            "rectified_sigma_min": self._rectified_sigma_min,
            "sample_steps": self._sample_steps,
            "sample_cfg_strength": self._sample_cfg_strength,
        }

    def _build_lightning_module(
        self, network: nn.Module, parameter_loss: ParameterLoss, training_config: TrainingConfig
    ):
        # Lazy: the training-only Lightning stack (D-FRAMEWORK) stays off the eval path.
        # parameter_loss is unused: the objective is the flow-matching velocity MSE
        # (see models/flow_matching/lightning_module.py).
        from models.flow_matching.lightning_module import LightningFlowMatching

        return LightningFlowMatching(
            network,
            training_config.optimizer,
            cfg_dropout_rate=self._cfg_dropout_rate,
            rectified_sigma_min=self._rectified_sigma_min,
            ot_pairing=self._ot_pairing,
            validation_sample_steps=self._validation_sample_steps,
            validation_cfg_strength=self._validation_cfg_strength,
        )

    def predict(self, audio: torch.Tensor) -> Dict[str, float]:
        """Sample a synth-side dict for one waveform ``[num_samples]``.

        Overrides the base single-forward ``predict``: draws one seeded sample from the
        flow (the paper's single-sample test protocol), maps it from flow space
        ``[-1, 1]`` back to the ML-side ``[0, 1]``, and decodes it with
        ``ml_vector_to_synth_dict``. The per-call seeded generator makes repeated
        predictions of the same clip identical.
        """
        if self._network is None or self._parameter_space is None:
            raise RuntimeError("Model must be fit (or loaded) before predict.")
        self._network.eval()
        audio = audio.to(next(self._network.parameters()).device)
        generator = torch.Generator().manual_seed(self._predict_seed)
        with torch.no_grad():
            flow_sample = self._network.sample(audio.unsqueeze(0), generator=generator)
        vector = ((flow_sample.squeeze(0) + 1.0) / 2.0).cpu().numpy()
        return self._parameter_space.ml_vector_to_synth_dict(vector)


class FlowMatchingMLP(BaseFlowMatchingModel):
    """The paper's CNF (MLP) model: the non-equivariant residual-MLP vector field.

    A 9-block conditional residual MLP (``d_model`` 768) with per-block Ada-LN-style
    conditioning from the shared AST encoder -- the paper's ``surge_flowmlp.yaml``.
    The non-equivariant member of the CNF pair; its comparison against CNF (Param2Tok)
    isolates the value of building the synth's permutation symmetry into the field.
    """

    _vector_field_architecture = "mlp"


class FlowMatchingParam2Tok(BaseFlowMatchingModel):
    """The paper's CNF (Param2Tok) model: the approximately-equivariant DiT vector field.

    Param2Tok maps the ML-side vector to 128 tokens, an 8-block Diffusion Transformer
    (``d_model`` 512) with no positional encoding processes them permutation-equivariantly,
    and Param2Tok maps back -- the paper's ``surge_flow.yaml``. Its assignment matrix picks
    up an L1 penalty during training, so the learned parameters-to-token routing is pushed
    to be sparse. This is the paper's headline model: the same corpus, encoder, loss and
    sampler as :class:`FlowMatchingMLP`, differing only in the field, so the pair isolates
    what the symmetry-aware architecture is worth.

    Constructor defaults differ from the base's MLP-shaped ones (8 layers at 512 rather
    than 9 at 768), matching the paper's config.
    """

    _vector_field_architecture = "param2tok"

    def __init__(
        self,
        field_d_model: int = 512,
        field_num_layers: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            field_d_model=field_d_model, field_num_layers=field_num_layers, **kwargs
        )
