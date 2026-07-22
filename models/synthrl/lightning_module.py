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
it maps the framework's ML-side target vector to per-head class indices (continuous binned,
categorical argmax) and returns the mean soft cross-entropy plus class accuracy.

The RL stage renders inside the training loop with the real Dexed via the parallel
fresh-process backend, so a SynthRL-i *training* run is no longer VST-free (a deliberate,
documented deviation from D-SELFDESC -- training-only; the eval path is unchanged).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from dataset.render_backends import ParallelFreshProcessRenderBackend, RenderSettings
from models.synthrl.representation import SynthRLRepresentation
from models.synthrl.reward import DEFAULT_REWARD_WEIGHTS, RewardWeights, sound_matching_reward
from models.synthrl.reward_buffer import RewardPrioritizedReplayBuffer
from models.training.config import OptimizerConfig
from models.training.lightning_module import build_optimizers


class SmoothedClassCrossEntropy(nn.Module):
    """The SynthRL parameter loss: Gaussian-smoothed per-parameter cross-entropy.

    Maps the framework's ML-side target vector (continuous floats + one-hot categorical
    blocks) to per-head target class indices -- continuous values binned, categorical
    blocks argmax-decoded -- gathers the Gaussian-smoothed soft targets from
    :meth:`SynthRLRepresentation.smoothing_matrices`, and returns the mean soft
    cross-entropy over the network's flat class logits together with the class accuracy.
    """

    def __init__(self, representation: SynthRLRepresentation) -> None:
        super().__init__()
        self._class_slices = representation.class_slices
        self._num_parameters = len(representation.class_counts)
        self._num_bins = representation.num_bins

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
        return (fraction * self._num_bins).floor().long().clamp(0, self._num_bins - 1)

    def forward(self, logits: torch.Tensor, ml_targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean soft cross-entropy, mean class accuracy)`` over the batch."""
        total_loss = logits.new_zeros(())
        total_correct = logits.new_zeros(())
        for position in range(self._num_parameters):
            class_index = self._target_class_index(position, ml_targets)  # [batch]
            block_logits = logits[:, self._class_slices[position]]  # [batch, count]
            smoothing = getattr(self, f"_smoothing_{position}")
            soft_target = smoothing[class_index]  # [batch, count]
            total_loss = total_loss - (soft_target * F.log_softmax(block_logits, dim=1)).sum(dim=1).mean()
            total_correct = total_correct + (block_logits.argmax(dim=1) == class_index).float().mean()
        return total_loss / self._num_parameters, total_correct / self._num_parameters


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
    (one class per parameter). Each training step:

    1. samples an action per batch item, decodes it to a synth-side patch, **renders** it
       with the real Dexed (parallel fresh-process backend), scores audio similarity to the
       target with :func:`sound_matching_reward`, and stores ``(action, reward)`` in the
       per-target reward-based PER buffer;
    2. samples experiences from each target's buffer and forms the REINFORCE objective
       ``-mean(reward * log pi(action))`` (the paper's uniform ``1/m`` importance weight);
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
        buffer_capacity: int = 8,
        samples_per_target: int = 4,
        ramp_epochs: int = 0,
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
        self._samples_per_target = int(samples_per_target)
        self._ramp_epochs = int(ramp_epochs)
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

    # -- policy helpers ------------------------------------------------------
    def _sample_actions(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample one class per parameter from the policy: ``[batch, num_parameters]``."""
        columns = [Categorical(logits=logits[:, block]).sample() for block in self._class_slices]
        return torch.stack(columns, dim=1)

    def _greedy_actions(self, logits: torch.Tensor) -> torch.Tensor:
        """Argmax class per parameter: ``[batch, num_parameters]``."""
        columns = [logits[:, block].argmax(dim=1) for block in self._class_slices]
        return torch.stack(columns, dim=1)

    def _log_prob(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Policy log-prob of ``actions [rows, num_parameters]`` under ``logits [rows, total]``."""
        total = logits.new_zeros(logits.shape[0])
        for position, block in enumerate(self._class_slices):
            total = total + Categorical(logits=logits[:, block]).log_prob(actions[:, position])
        return total

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
    def _reinforce_loss(self, logits: torch.Tensor, ml_targets: torch.Tensor) -> torch.Tensor:
        """REINFORCE loss from PER-buffer experiences of the batch's targets."""
        per_target_losses: List[torch.Tensor] = []
        for row in range(logits.shape[0]):
            key = self._target_key(ml_targets, row)
            if key not in self._buffer:
                continue
            experiences = self._buffer.sample(key, self._samples_per_target, self._rng)
            actions = torch.as_tensor(
                np.stack([experience.action for experience in experiences]),
                device=logits.device, dtype=torch.long,
            )
            rewards = torch.as_tensor(
                [experience.reward for experience in experiences],
                device=logits.device, dtype=logits.dtype,
            )
            repeated_logits = logits[row : row + 1].expand(len(experiences), -1)
            log_prob = self._log_prob(repeated_logits, actions)
            per_target_losses.append(-(rewards * log_prob).mean())
        if not per_target_losses:
            return logits.new_zeros(())
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
