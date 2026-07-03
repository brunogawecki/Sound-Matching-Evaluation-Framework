"""Fit the spectrogram-CNN discriminative regressor on a corpus and save a checkpoint.

The first real deep family (issue #19, Sound2Synth lineage): a VGG11-BN conv net over
a log-power STFT of the target audio. Fitting drives the training harness end-to-end
(config -> DataModule -> LightningRegressor -> Trainer -> exported checkpoint) and needs
no VST -- only the corpus's rendered audio + ML-side targets. This mirrors the real flow
(train + save on the cluster, then load + eval locally) and produces the checkpoint
``scripts/evaluate.py`` loads.

    python scripts/fit_model.py --corpus dataset/run_A_train --config training_config.yaml

    --corpus  training corpus directory                            [default: dataset/run_A_train]
    --out     checkpoint path to write       [default: <project>/checkpoints/spectrogram_cnn.pt]
    --config  training_config.yaml (harness knobs); omit for harness defaults  [optional]
    --val     optional explicit validation corpus directory                    [optional]
"""
import argparse
import sys
from pathlib import Path

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, dataset, models) import from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from dataset.torch_dataset import RenderedCorpusDataset
from models.sound2synth import Sound2SynthSpectrogramRegressor
from models.training.config import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and save the spectrogram-CNN regressor.")
    parser.add_argument(
        "--corpus", default=str(Path(config.DATASET_DIR) / "run_A_train"),
        help="training corpus directory",
    )
    parser.add_argument(
        "--out", default=str(Path(config.BASE_DIR) / "checkpoints" / "spectrogram_cnn.pt"),
        help="checkpoint path to write",
    )
    parser.add_argument(
        "--config", default=None,
        help="training_config.yaml with harness knobs (omit for defaults)",
    )
    parser.add_argument(
        "--val", default=None,
        help="optional explicit validation corpus directory",
    )
    args = parser.parse_args()

    training_config = (
        TrainingConfig.from_yaml(args.config).to_dict() if args.config else None
    )

    corpus = RenderedCorpusDataset.load(args.corpus)
    validation_corpus = RenderedCorpusDataset.load(args.val) if args.val else None
    print(f"--- Fitting Sound2SynthSpectrogramRegressor on '{corpus.corpus_dir.name}' "
          f"({len(corpus)} samples) ---")

    model = Sound2SynthSpectrogramRegressor(
        default_root_dir=str(Path(config.BASE_DIR) / "lightning_logs"),
    )
    model.fit(corpus, validation_dataset=validation_corpus, config=training_config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
