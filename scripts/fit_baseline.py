"""Fit the mean-parameter baseline on a training corpus and save a checkpoint.

The baseline ignores audio and predicts the training set's mean parameter vector
(issue #7); fitting needs only the corpus's ML-side target matrix -- no VST. This
mirrors the real flow (train + save on the cluster, then load + eval locally) and
produces the checkpoint ``scripts/evaluate.py`` loads.

    python scripts/fit_baseline.py --corpus dataset/run_A_train

    --corpus  training corpus directory                  [default: dataset/run_A_train]
    --out     checkpoint path to write   [default: <project>/checkpoints/mean_parameter_baseline.json]
"""
import argparse
import sys
from pathlib import Path

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, dataset, models) import from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from dataset.torch_dataset import RenderedCorpusDataset
from models.mean_parameter_baseline import MeanParameterBaseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and save the mean-parameter baseline.")
    parser.add_argument(
        "--corpus", default=str(Path(config.DATASET_DIR) / "run_A_train"),
        help="training corpus directory",
    )
    parser.add_argument(
        "--out", default=str(Path(config.BASE_DIR) / "checkpoints" / "mean_parameter_baseline.json"),
        help="checkpoint path to write",
    )
    args = parser.parse_args()

    corpus = RenderedCorpusDataset.load(args.corpus)
    print(f"--- Fitting MeanParameterBaseline on '{corpus.corpus_dir.name}' ({len(corpus)} samples) ---")
    model = MeanParameterBaseline()
    model.fit(corpus)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
