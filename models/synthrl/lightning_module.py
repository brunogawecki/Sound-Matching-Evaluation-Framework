"""LightningModules for the SynthRL training stages (training-only, D-FRAMEWORK).

Two recipes, both lazily imported by the SynthRL families so the eval path needs no
Lightning:

1. :class:`SynthRLParameterRegressor` -- the parameter stage (SynthRL-p, paper §3.3):
   the per-parameter classification loss, cross-entropy with **Gaussian label smoothing**
   over the network's class heads.
2. :class:`SynthRLReinforceRegressor` -- the in-domain RL stage (SynthRL-i, paper §3.4):
   REINFORCE over the discrete class policy, rewarded by re-rendering the sampled patch
   and scoring audio similarity, with a per-target reward-based PER buffer and a
   parameter-loss -> RL curriculum ramp.

Both share :class:`SmoothedClassCrossEntropy`, the parameter loss as a small ``nn.Module``:
it maps the framework's ML-side target vector to per-head class indices (continuous snapped
to the nearest level, categorical argmax) and returns the mean soft cross-entropy plus class
accuracy.

The RL stage renders inside the training loop with the real Dexed via the parallel
fresh-process backend, so a SynthRL-i *training* run is no longer VST-free (a deliberate,
documented deviation from D-SELFDESC -- training-only; the eval path is unchanged).
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_info
from torch import nn
from torch.distributions import Categorical

from dataset.render_backends import ParallelFreshProcessRenderBackend, RenderSettings
from models.synthrl.representation import SynthRLRepresentation
from models.synthrl.reward import DEFAULT_REWARD_WEIGHTS, RewardWeights, sound_matching_reward
from models.synthrl.reward_buffer import RewardPrioritizedReplayBuffer
from models.training.config import OptimizerConfig
from models.training.lightning_module import build_optimizers

# Softmax temperatures applied to the network's [0, 1] class scores. The repo sets the
# loss one in config (`cat_softmax_t: 0.2`) and leaves the policy one at the
# PresetProcessor default (0.1). Both cap how sharp the resulting distribution can get:
# the within-head score gap is at most 1.0, so the odds ratio is at most exp(1/T).
LOSS_SOFTMAX_TEMPERATURE = 0.2
POLICY_SOFTMAX_TEMPERATURE = 0.1
# The repo weights its categorical cross-entropy by `categorical_loss_factor` (0.2) before
# mixing it with the RL term. Kept so the curriculum ramp balances the two the same way.
CATEGORICAL_LOSS_FACTOR = 0.2

# A Dexed operator at output level 0 is silent, so the rest of that operator's parameters do
# not affect the sound and carry no learnable signal. The repo drops them from the parameter
# loss per sample (data/preset.py: get_useless_learned_params_indexes); the groups below are
# derived from the D-NAMING parameter names, so this is a no-op on any synth that does not
# use the `OP<n> <param>` convention.
_OPERATOR_PREFIX = re.compile(r"^(OP\d+) ")
_GATE_SUFFIX = "OUTPUT LEVEL"


def gated_parameter_groups(names: List[str]) -> List[Tuple[int, List[int]]]:
    """``(gate position, gated positions)`` per operator, read off the parameter names.

    Empty when the names do not follow the ``OP<n> ...`` / ``OP<n> OUTPUT LEVEL`` convention.
    """
    gates: Dict[str, int] = {}
    gated: Dict[str, List[int]] = {}
    for position, name in enumerate(names):
        match = _OPERATOR_PREFIX.match(name)
        if match is None:
            continue
        operator = match.group(1)
        if name == f"{operator} {_GATE_SUFFIX}":
            gates[operator] = position
        else:
            gated.setdefault(operator, []).append(position)
    return [
        (gates[operator], positions)
        for operator, positions in sorted(gated.items())
        if operator in gates
    ]


class SmoothedClassCrossEntropy(nn.Module):
    """The SynthRL parameter loss: Gaussian-smoothed per-parameter cross-entropy.

    Maps the framework's ML-side target vector (continuous floats + one-hot categorical
    blocks) to per-head target class indices -- continuous values snapped to their
    nearest level, categorical blocks argmax-decoded -- gathers the smoothed targets from
    :meth:`SynthRLRepresentation.smoothing_matrices`, and returns the mean soft
    cross-entropy over the network's flat class scores together with the class accuracy.
    Heads belonging to a silent operator are dropped per sample (see
    :func:`gated_parameter_groups`).
    """

    def __init__(
        self,
        representation: SynthRLRepresentation,
        softmax_temperature: float = LOSS_SOFTMAX_TEMPERATURE,
        loss_factor: float = CATEGORICAL_LOSS_FACTOR,
    ) -> None:
        super().__init__()
        self._class_slices = representation.class_slices
        self._num_parameters = len(representation.class_counts)
        self._num_bins = representation.num_bins
        self._softmax_temperature = float(softmax_temperature)
        self._loss_factor = float(loss_factor)

        # position -> the position of the parameter gating it (None when ungated).
        self._gate_position: List[Optional[int]] = [None] * self._num_parameters
        for gate, gated in gated_parameter_groups(representation.names):
            for position in gated:
                self._gate_position[position] = gate

        # Per-parameter ML-side layout for the target -> class-index mapping.
        self._parameter_kinds: List[str] = []
        self._ml_slices: List[slice] = []
        continuous_low = torch.zeros(self._num_parameters)
        continuous_high = torch.ones(self._num_parameters)
        specs = representation.parameter_space.parameter_specs
        for position, (ml_slice, kind, _name) in enumerate(
            representation.parameter_space.loss_slices
        ):
            self._parameter_kinds.append(kind)
            self._ml_slices.append(ml_slice)
            if kind == "continuous":
                continuous_low[position], continuous_high[position] = specs[position].bounds
        self.register_buffer("_continuous_low", continuous_low, persistent=False)
        self.register_buffer("_continuous_high", continuous_high, persistent=False)

        # Per-head Gaussian-smoothing lookup tables (derived; kept off the checkpoint).
        for position, matrix in enumerate(representation.smoothing_matrices()):
            self.register_buffer(
                f"_smoothing_{position}",
                torch.tensor(matrix, dtype=torch.float32),
                persistent=False,
            )

    def target_class_indices(self, ml_targets: torch.Tensor) -> torch.Tensor:
        """Per-parameter target class indices ``[batch, num_parameters]`` from ML targets."""
        columns = [self._target_class_index(position, ml_targets) for position in range(self._num_parameters)]
        return torch.stack(columns, dim=1)

    def _target_class_index(self, position: int, ml_targets: torch.Tensor) -> torch.Tensor:
        ml_slice = self._ml_slices[position]
        if self._parameter_kinds[position] == "categorical":
            return ml_targets[:, ml_slice].argmax(dim=1)
        value = ml_targets[:, ml_slice.start]
        low = self._continuous_low[position]
        high = self._continuous_high[position]
        fraction = (value - low) / (high - low)
        if self._num_bins == 1:
            return torch.zeros_like(value, dtype=torch.long)
        # Mirror SynthRLRepresentation._level_index: snap to the nearest of num_bins levels.
        return (fraction * (self._num_bins - 1)).round().long().clamp(0, self._num_bins - 1)

    def forward(self, logits: torch.Tensor, ml_targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean soft cross-entropy, mean class accuracy)`` over the batch.

        A head gated by a silent operator contributes nothing for that sample; its loss is
        averaged over the samples where the gate is open. Accuracy is left unmasked, as in
        the repo's monitoring metric.
        """
        class_indices = self.target_class_indices(ml_targets)  # [batch, num_parameters]
        total_loss = logits.new_zeros(())
        total_correct = logits.new_zeros(())
        for position in range(self._num_parameters):
            class_index = class_indices[:, position]  # [batch]
            block_logits = logits[:, self._class_slices[position]] / self._softmax_temperature
            smoothing = getattr(self, f"_smoothing_{position}")
            soft_target = smoothing[class_index]  # [batch, count]
            cross_entropy = -(soft_target * F.log_softmax(block_logits, dim=1)).sum(dim=1)  # [batch]
            gate = self._gate_position[position]
            if gate is None:
                total_loss = total_loss + cross_entropy.mean()
            else:
                # Gate class 0 is the parameter's minimum, i.e. the operator is silent.
                open_gate = (class_indices[:, gate] != 0).to(logits.dtype)
                total_loss = total_loss + (open_gate * cross_entropy).sum() / open_gate.sum().clamp(min=1.0)
            total_correct = total_correct + (block_logits.argmax(dim=1) == class_index).float().mean()
        loss = self._loss_factor * total_loss / self._num_parameters
        return loss, total_correct / self._num_parameters


class SynthRLParameterRegressor(pl.LightningModule):
    """Trains a :class:`SynthRLNetwork` with the Gaussian-smoothed classification loss.

    ``network`` maps ``audio [batch, num_samples]`` to flat class logits
    ``[batch, total_class_dimension]``; ``representation`` supplies the class layout and
    the parameter loss. This is the RL-free stage (SynthRL-p).
    """

    def __init__(
        self,
        network: nn.Module,
        representation: SynthRLRepresentation,
        optimizer_config: OptimizerConfig,
    ) -> None:
        super().__init__()
        self.network = network
        self._optimizer_config = optimizer_config
        self.parameter_loss = SmoothedClassCrossEntropy(representation)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.network(audio)

    def training_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def _shared_step(self, batch: List[torch.Tensor], stage: str) -> torch.Tensor:
        audio, ml_targets = batch
        logits = self.network(audio)
        loss, accuracy = self.parameter_loss(logits, ml_targets)
        log = dict(on_step=False, on_epoch=True, batch_size=audio.shape[0])
        self.log(f"{stage}_loss", loss, prog_bar=True, **log)
        self.log(f"{stage}_class_accuracy", accuracy, prog_bar=True, **log)
        return loss

    def configure_optimizers(self):
        return build_optimizers(self.network, self._optimizer_config)


class SynthRLReinforceRegressor(pl.LightningModule):
    """Trains a :class:`SynthRLNetwork` with in-domain reinforcement learning (SynthRL-i).

    The network's per-parameter class heads are a stochastic policy over discrete actions
    (one class per parameter, sampled at ``policy_temperature``). Before the first gradient
    step, :meth:`on_train_start` fills every target's buffer over ``prefill_epochs`` render
    passes. Then each training step:

    1. samples an action per batch item, decodes it to a synth-side patch, **renders** it
       with the real Dexed (parallel fresh-process backend), scores audio similarity to the
       target with :func:`sound_matching_reward`, and stores ``(action, reward)`` in the
       per-target reward-based PER buffer;
    2. samples experiences from each target's buffer and forms the REINFORCE objective
       ``-mean(m * reward * mean_log_pi(action))``, ``m = buffer_capacity`` being the paper's
       uniform ``1/m`` importance weight;
    3. blends in the parameter loss during a curriculum ramp: the parameter weight falls
       ``1 -> 0`` and the RL weight rises ``0 -> 1`` over ``ramp_epochs``, then RL-only.

    Validation renders the greedy (argmax) prediction and logs its mean reward as
    ``val_reward``, plus ``val_loss = -val_reward`` so best-checkpoint selection (min mode)
    picks the highest-reward epoch.
    """

    def __init__(
        self,
        network: nn.Module,
        representation: SynthRLRepresentation,
        optimizer_config: OptimizerConfig,
        render_settings: RenderSettings,
        sample_rate: int,
        renderer: str = "dawdreamer",
        num_render_workers: Optional[int] = None,
        reward_weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
        buffer_capacity: int = 5,
        samples_per_target: int = 1,
        prefill_epochs: int = 5,
        ramp_epochs: int = 0,
        policy_temperature: float = POLICY_SOFTMAX_TEMPERATURE,
        seed: int = 0,
        backend_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        super().__init__()
        self.network = network
        self._representation = representation
        self._optimizer_config = optimizer_config
        self._class_slices = representation.class_slices
        self.parameter_loss = SmoothedClassCrossEntropy(representation)

        self._render_settings = render_settings
        self._sample_rate = int(sample_rate)
        self._renderer = renderer
        self._num_render_workers = num_render_workers
        self._reward_weights = reward_weights
        self._buffer_capacity = int(buffer_capacity)
        self._samples_per_target = int(samples_per_target)
        self._prefill_epochs = int(prefill_epochs)
        self._ramp_epochs = int(ramp_epochs)
        self._policy_temperature = float(policy_temperature)
        self._rng = np.random.default_rng(seed)
        self._buffer = RewardPrioritizedReplayBuffer(buffer_capacity)
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backend = None

    # -- render backend lifecycle -------------------------------------------
    def _default_backend_factory(self):
        return ParallelFreshProcessRenderBackend(
            self._render_settings, renderer=self._renderer, num_workers=self._num_render_workers
        )

    def setup(self, stage: str) -> None:
        if stage == "fit" and self._backend is None:
            self._backend = self._backend_factory()

    def teardown(self, stage: str) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    # -- buffer pre-fill -----------------------------------------------------
    def on_train_start(self) -> None:
        """Fill every target's PER buffer before the first gradient step (repo ``finetune.py``).

        The repo runs ``per_capacity`` gradient-free passes over the training set -- the first
        greedy, the rest sampled -- so that when training begins each target already has a full
        buffer of ``m`` experiences to draw its best action from. Without this the buffer holds
        one entry per target for the first epochs and the objective degenerates to plain
        on-policy REINFORCE, exactly while the parameter loss is being ramped out.

        Each pass renders the whole training set, so this is expensive by construction; set
        ``prefill_epochs`` to 0 to skip it.
        """
        if self._prefill_epochs <= 0:
            return
        rank_zero_info(
            f"[SynthRL-i] pre-filling the reward buffer: {self._prefill_epochs} render passes "
            f"over the training set."
        )
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            for pass_index in range(self._prefill_epochs):
                # Repo: the first pass is deterministic, the remaining ones sample.
                greedy = pass_index == 0
                for batch in self.trainer.train_dataloader:
                    audio, ml_targets = (tensor.to(self.device) for tensor in batch)
                    scores = self.network(audio)
                    actions = self._greedy_actions(scores) if greedy else self._sample_actions(scores)
                    self._collect_experiences(actions, audio, ml_targets)
        self.network.train(was_training)

    # -- policy helpers ------------------------------------------------------
    def _policy(self, scores: torch.Tensor, block: slice) -> Categorical:
        """The per-head action distribution: the head's scores at the policy temperature."""
        return Categorical(logits=scores[:, block] / self._policy_temperature)

    def _sample_actions(self, scores: torch.Tensor) -> torch.Tensor:
        """Sample one class per parameter from the policy: ``[batch, num_parameters]``."""
        columns = [self._policy(scores, block).sample() for block in self._class_slices]
        return torch.stack(columns, dim=1)

    def _greedy_actions(self, scores: torch.Tensor) -> torch.Tensor:
        """Argmax class per parameter: ``[batch, num_parameters]``. Temperature-independent."""
        columns = [scores[:, block].argmax(dim=1) for block in self._class_slices]
        return torch.stack(columns, dim=1)

    def _mean_log_prob(self, scores: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Policy log-prob of ``actions [rows, num_parameters]``, **averaged** over parameters.

        The repo divides the summed per-parameter log-probabilities by the parameter count
        (``get_mean_log_probs``). Averaging rather than summing keeps the REINFORCE term on a
        scale the curriculum ramp can blend against the parameter loss.
        """
        total = scores.new_zeros(scores.shape[0])
        for position, block in enumerate(self._class_slices):
            total = total + self._policy(scores, block).log_prob(actions[:, position])
        return total / len(self._class_slices)

    # -- rendering + reward --------------------------------------------------
    def _render_and_reward(
        self, actions: torch.Tensor, target_audio: torch.Tensor
    ) -> np.ndarray:
        """Render each action's patch and score its reward against the target audio."""
        actions_np = actions.detach().cpu().numpy()
        targets_np = target_audio.detach().cpu().numpy()
        patches = [
            self._representation.class_indices_to_synth_dict(actions_np[row])
            for row in range(actions_np.shape[0])
        ]
        rendered = self._backend.render_batch(patches)
        return np.array(
            [
                sound_matching_reward(
                    targets_np[row], rendered[row],
                    sample_rate=self._sample_rate, weights=self._reward_weights,
                )
                for row in range(actions_np.shape[0])
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _target_key(ml_targets: torch.Tensor, row: int) -> bytes:
        """Stable per-target key: the target's ML-side vector bytes (same patch, same key)."""
        return ml_targets[row].detach().cpu().numpy().tobytes()

    def _collect_experiences(
        self, actions: torch.Tensor, target_audio: torch.Tensor, ml_targets: torch.Tensor
    ) -> float:
        """Render sampled actions, reward them, and add them to the per-target buffer."""
        rewards = self._render_and_reward(actions, target_audio)
        actions_np = actions.detach().cpu().numpy()
        for row in range(actions_np.shape[0]):
            self._buffer.add(self._target_key(ml_targets, row), actions_np[row], float(rewards[row]))
        return float(rewards.mean())

    # -- objective -----------------------------------------------------------
    def _reinforce_loss(self, scores: torch.Tensor, ml_targets: torch.Tensor) -> torch.Tensor:
        """REINFORCE loss from PER-buffer experiences of the batch's targets.

        The ``buffer_capacity`` factor is the paper's importance weight under the uniform
        proposal ``mu = 1/m`` (Eq. 6, repo ``per_capacity * rewards * mean_log_probs``); the
        policy-ratio part of that weight is dropped, as it is in the released code.
        """
        per_target_losses: List[torch.Tensor] = []
        for row in range(scores.shape[0]):
            key = self._target_key(ml_targets, row)
            if key not in self._buffer:
                continue
            experiences = self._buffer.sample(key, self._samples_per_target, self._rng)
            actions = torch.as_tensor(
                np.stack([experience.action for experience in experiences]),
                device=scores.device, dtype=torch.long,
            )
            rewards = torch.as_tensor(
                [experience.reward for experience in experiences],
                device=scores.device, dtype=scores.dtype,
            )
            repeated_scores = scores[row : row + 1].expand(len(experiences), -1)
            mean_log_prob = self._mean_log_prob(repeated_scores, actions)
            per_target_losses.append(-(self._buffer_capacity * rewards * mean_log_prob).mean())
        if not per_target_losses:
            return scores.new_zeros(())
        return torch.stack(per_target_losses).mean()

    def _ramp_progress(self) -> float:
        if self._ramp_epochs <= 0:
            return 1.0
        return min(1.0, self.current_epoch / self._ramp_epochs)

    def training_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        audio, ml_targets = batch
        logits = self.network(audio)

        with torch.no_grad():
            sampled = self._sample_actions(logits)
        mean_reward = self._collect_experiences(sampled, audio, ml_targets)

        rl_weight = self._ramp_progress()
        parameter_weight = 1.0 - rl_weight
        policy_loss = self._reinforce_loss(logits, ml_targets)
        if parameter_weight > 0.0:
            parameter_loss, _accuracy = self.parameter_loss(logits, ml_targets)
        else:
            parameter_loss = logits.new_zeros(())
        total = rl_weight * policy_loss + parameter_weight * parameter_loss

        log = dict(on_step=False, on_epoch=True, batch_size=audio.shape[0])
        self.log("train_loss", total, prog_bar=True, **log)
        self.log("train_reward", mean_reward, prog_bar=True, **log)
        self.log("train_policy_loss", policy_loss, **log)
        return total

    def validation_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        audio, _ml_targets = batch
        with torch.no_grad():
            logits = self.network(audio)
            greedy = self._greedy_actions(logits)
            mean_reward = float(self._render_and_reward(greedy, audio).mean())
        log = dict(on_step=False, on_epoch=True, batch_size=audio.shape[0])
        self.log("val_reward", mean_reward, prog_bar=True, **log)
        # min-mode checkpoint monitor: higher reward = lower loss.
        self.log("val_loss", -mean_reward, prog_bar=True, **log)
        return torch.tensor(-mean_reward)

    def configure_optimizers(self):
        return build_optimizers(self.network, self._optimizer_config)
