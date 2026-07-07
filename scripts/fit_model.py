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
import sys
from pathlib import Path

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, dataset, models) import from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from dataset.torch_dataset import RenderedCorpusDataset
from models.registry import MODEL_REGISTRY
from models.training.config import TrainingConfig


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
    args = parser.parse_args()

    registration = MODEL_REGISTRY[args.model]
    out_path = Path(
        args.out
        if args.out
        else Path(config.BASE_DIR) / "checkpoints" / registration.default_checkpoint_filename
    )

    training_config = (
        TrainingConfig.from_yaml(args.config).to_dict() if args.config else None
    )

    corpus = RenderedCorpusDataset.load(args.corpus)
    validation_corpus = RenderedCorpusDataset.load(args.val) if args.val else None
    print(f"--- Fitting {args.model} on '{corpus.corpus_dir.name}' "
          f"({len(corpus)} samples) ---")

    model = registration.model_class()
    model.fit(corpus, validation_dataset=validation_corpus, config=training_config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
