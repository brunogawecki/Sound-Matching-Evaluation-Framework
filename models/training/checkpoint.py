"""The exported inference checkpoint -- a single, self-contained ``torch`` artifact.

D-FRAMEWORK requires that ``BaseModel.save``/``load`` round-trip a plain ``torch``
``state_dict`` (+ minimal hparams), **never** a raw Lightning ``.ckpt``. Lightning's
own ``.ckpt`` files stay cluster-side for crash/requeue only; at the end of training
a family exports the best weights through :func:`export_checkpoint` into the clean
artifact this module defines.

The artifact is one ``torch.save`` dict carrying everything ``load`` needs to fully
reconstruct a model with **no training data and no VST**:

- ``state_dict``            -- the inference core's weights;
- ``architecture_hparams``  -- the hyperparameters ``_build_inference_core`` needs to
  rebuild the core's structure before loading the weights;
- ``parameter_space``       -- the serialized :class:`ParameterSpace`
  (:meth:`ParameterSpace.to_dict`), so ``predict`` can decode ML-side vectors back to
  synth-side dicts with no live synth.

Pure ``torch`` -- no Lightning import, so the Mac eval path can load a checkpoint
without the training framework installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import torch
from torch import nn

from synth.parameter_space import ParameterSpace

CHECKPOINT_FORMAT_VERSION = 1


def export_checkpoint(
    core: nn.Module,
    architecture_hparams: Dict[str, Any],
    parameter_space: ParameterSpace,
    path: Union[str, Path],
) -> None:
    """Write the inference core + hparams + ParameterSpace as one ``torch`` file.

    Args:
        core: the trained inference core (a plain ``nn.Module``); its
            ``state_dict`` is saved on CPU.
        architecture_hparams: the kwargs ``_build_inference_core`` consumes to
            rebuild ``core``'s structure (must be JSON-safe / picklable plain data).
        parameter_space: the corpus's space, serialized for offline decoding.
        path: destination file (parent directories are created).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": {key: value.cpu() for key, value in core.state_dict().items()},
        "architecture_hparams": dict(architecture_hparams),
        "parameter_space": parameter_space.to_dict(),
    }
    torch.save(payload, path)


def core_state_dict_from_lightning_checkpoint(
    path: Union[str, Path], core_prefix: str = "inference_core."
) -> Dict[str, Any]:
    """Extract just the inference core's weights from a Lightning ``.ckpt``.

    The end-of-``fit`` export step: a family reads the best ``.ckpt`` Lightning wrote
    (whose ``state_dict`` keys are prefixed by the LightningModule's attribute name,
    e.g. ``inference_core.``), strips that prefix, and feeds the result into
    :func:`export_checkpoint`. Pure ``torch.load`` -- no Lightning import needed to
    read its own file format.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    return {
        key[len(core_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(core_prefix)
    }


def load_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a checkpoint written by :func:`export_checkpoint`.

    Returns the raw payload dict (``state_dict`` / ``architecture_hparams`` /
    ``parameter_space``); reconstructing the core from it is
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
