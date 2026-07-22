"""Fit a model on a training corpus and save a checkpoint.

Generic over the model family: ``--model`` selects any class registered in
``models.registry.MODEL_REGISTRY``. The baseline ignores ``--config``/``--val``;
deep families drive the training harness (config -> DataModule -> LightningRegressor
-> Trainer -> exported checkpoint) and need only the corpus's rendered audio +
ML-side targets, no VST. Produces the checkpoint ``scripts/evaluate.py`` loads.

    python scripts/fit_model.py --model Sound2SynthSpectrogramRegressor \
        --corpus dataset/run_A_train --config training_config.yaml
    python scripts/fit_model.py --model MeanParameterBaseline --corpus dataset/run_A_train
"""
import argparse
import inspect
import sys
from pathlib import Path
from typing import Optional, Tuple

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, dataset, models) import from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.registry import MODEL_REGISTRY
from models.training.config import TrainingConfig


def resolve_run_paths(model_name: str, run_id: Optional[str]) -> Tuple[Path, Path]:
    """Where this run writes its checkpoint and its Lightning logs.

    A ``run_id`` (the SLURM job id on the cluster) scopes both under a directory
    of their own, so repeat runs of one family never overwrite each other and a
    pull can name exactly one job's artifacts. Without one, the flat layout is
    unchanged.
    """
    base_dir = Path(config.BASE_DIR)
    filename = MODEL_REGISTRY[model_name].default_checkpoint_filename
    if run_id:
        return (
            base_dir / "checkpoints" / run_id / filename,
            base_dir / "lightning_logs" / run_id,
        )
    return base_dir / "checkpoints" / filename, base_dir / "lightning_logs"


def _accepts_init_from_checkpoint(model_class: type) -> bool:
    """Whether the family takes ``init_from_checkpoint`` anywhere up its MRO.

    Staged families (SynthRLi) declare it on a base ``__init__`` and forward it through
    ``**kwargs`` on the leaf, so the leaf signature alone does not show it.
    """
    for klass in model_class.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        if "init_from_checkpoint" in inspect.signature(init).parameters:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and save a sound-matching model.")
    parser.add_argument(
        "--model", required=True, choices=sorted(MODEL_REGISTRY),
        help="model family to train",
    )
    parser.add_argument(
        "--corpus", default=str(Path(config.DATASET_DIR) / "run_A_train"),
        help="training corpus directory",
    )
    parser.add_argument(
        "--out", default=None,
        help="checkpoint path to write (default: checkpoints/<model default filename>)",
    )
    parser.add_argument(
        "--config", default=None,
        help="training_config.yaml with harness knobs (ignored by the baseline; omit for defaults)",
    )
    parser.add_argument(
        "--val", default=None,
        help="optional explicit validation corpus directory (ignored by the baseline)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="scope this run's checkpoint and logs under checkpoints/<run-id>/ and "
             "lightning_logs/<run-id>/ (the cluster passes $SLURM_JOB_ID)",
    )
    parser.add_argument(
        "--init-from", default=None,
        help="checkpoint to warm-start from, for staged families (e.g. SynthRLi from a "
             "SynthRLp checkpoint); only families whose constructor accepts "
             "init_from_checkpoint support it",
    )
    args = parser.parse_args()

    registration = MODEL_REGISTRY[args.model]
    default_out_path, log_root = resolve_run_paths(args.model, args.run_id)
    out_path = Path(args.out) if args.out else default_out_path

    training_config = (
        TrainingConfig.from_yaml(args.config).to_dict() if args.config else None
    )

    corpus = RenderedCorpusDataset.load(args.corpus)
    validation_corpus = RenderedCorpusDataset.load(args.val) if args.val else None
    print(f"--- Fitting {args.model} on '{corpus.corpus_dir.name}' "
          f"({len(corpus)} samples) ---")

    # Only the deep families log through the Lightning harness; the baseline
    # takes no default_root_dir.
    model_kwargs = (
        {"default_root_dir": str(log_root)}
        if issubclass(registration.model_class, BaseDeepModel)
        else {}
    )
    if args.init_from:
        if not _accepts_init_from_checkpoint(registration.model_class):
            parser.error(f"{args.model} does not support --init-from (no warm-start hook)")
        model_kwargs["init_from_checkpoint"] = args.init_from
    model = registration.model_class(**model_kwargs)
    model.fit(corpus, validation_dataset=validation_corpus, config=training_config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
