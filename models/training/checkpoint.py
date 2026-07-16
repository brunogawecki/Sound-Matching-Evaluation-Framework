"""The exported inference checkpoint -- a single self-contained ``torch`` artifact.

One ``torch.save`` dict carrying everything ``load`` needs to reconstruct a model with
no training data and no VST: the network ``state_dict``, the ``architecture_hparams``
``_build_network`` consumes, and the serialized :class:`ParameterSpace`. Pure ``torch``
(no Lightning), so the eval path can load a checkpoint without the training framework.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from synth.parameter_space import ParameterSpace

CHECKPOINT_FORMAT_VERSION = 1


def export_checkpoint(
    network: nn.Module,
    architecture_hparams: Dict[str, Any],
    parameter_space: ParameterSpace,
    path: Union[str, Path],
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the network + hparams + ParameterSpace as one ``torch`` file.

    Args:
        network: the trained network (a plain ``nn.Module``); its
            ``state_dict`` is saved on CPU.
        architecture_hparams: the kwargs ``_build_network`` consumes to
            rebuild ``network``'s structure (must be JSON-safe / picklable plain data).
        parameter_space: the corpus's space, serialized for offline decoding.
        path: destination file (parent directories are created).
        extra_state: optional family-specific payload (plain data / tensors) that
            ``load`` hands back to the family. Additive and optional -- families that
            do not use it write ``None`` and older checkpoints simply lack the key.
            InverSynth II's ``IS2`` stores its cached ITF training pool here.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": {key: value.cpu() for key, value in network.state_dict().items()},
        "architecture_hparams": dict(architecture_hparams),
        "parameter_space": parameter_space.to_dict(),
        "extra_state": extra_state,
    }
    torch.save(payload, path)


def network_state_dict_from_lightning_checkpoint(
    path: Union[str, Path], network_prefix: str = "network."
) -> Dict[str, Any]:
    """Extract just the network's weights from a Lightning ``.ckpt``.

    The end-of-``fit`` export step: a family reads the best ``.ckpt`` Lightning wrote
    (whose ``state_dict`` keys are prefixed by the LightningModule's attribute name,
    e.g. ``network.``), strips that prefix, and feeds the result into
    :func:`export_checkpoint`. Pure ``torch.load`` -- no Lightning import needed to
    read its own file format.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    network_state_dict = {
        key[len(network_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(network_prefix)
    }
    if not network_state_dict:
        raise ValueError(
            f"No keys prefixed '{network_prefix}' in {path}; the LightningModule's "
            "network attribute name does not match network_prefix."
        )
    return network_state_dict


def load_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a checkpoint written by :func:`export_checkpoint`.

    Returns the raw payload dict (``state_dict`` / ``architecture_hparams`` /
    ``parameter_space``); reconstructing the network from it is
    :class:`~models.base_deep_model.BaseDeepModel`'s job. ``weights_only=False`` is
    intentional -- this is our own trusted artifact and the payload holds plain
    Python containers alongside the tensors.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version {version!r} at {path}; "
            f"this build writes/reads version {CHECKPOINT_FORMAT_VERSION}."
        )
    return payload
