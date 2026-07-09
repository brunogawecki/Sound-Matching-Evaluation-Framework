"""Evaluate a fitted model checkpoint on a corpus through the metric panel.

Loads a model from its checkpoint and a corpus from disk, runs the
:class:`~evaluation.evaluator.Evaluator` (which re-renders each prediction in a fresh
process at position 0 -- D-REPRO -- so it needs the Dexed VST locally), and writes
``results/<corpus_name>/<model_name>/{per_sample.csv, eval_summary.json}``.

Pair with ``scripts/fit_model.py``, which produces the checkpoint::

    python scripts/fit_model.py --model MeanParameterBaseline --corpus dataset/run_A_train
    python scripts/evaluate.py --checkpoint checkpoints/mean_parameter_baseline.json \
        --corpus dataset/run_A_test --model MeanParameterBaseline

    --checkpoint      the saved model file to load and fingerprint        [REQUIRED]
    --corpus          the eval corpus directory (must be fresh-process)    [REQUIRED]
    --model           model class to load the checkpoint into              [REQUIRED]
    --out             results root                                  [default: <project>/results]
    --save-audio      persist prediction WAVs for a seeded random sample subset
    --save-audio-n    cap on how many samples get saved             [default: 20]
"""
import argparse
import os
import sys
from pathlib import Path

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, evaluation, dataset, models) import from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from dataset.torch_dataset import RenderedCorpusDataset
from evaluation.evaluator import EvaluationResult, Evaluator
from models.registry import MODEL_REGISTRY


def _require_dexed() -> None:
    """The re-render path needs the local VST; fail early with a clear message."""
    plugin_path = os.path.expanduser(config.DEXED_PATH)
    if not os.path.exists(plugin_path):
        print(f"Could not find Dexed plugin at: {plugin_path}")
        print("The Evaluator re-renders predictions, which needs the VST (D-REPRO).")
        print("Please update DEXED_PATH in your .env file.")
        sys.exit(1)


def _print_table(result: EvaluationResult) -> None:
    print("\nResults")
    print(f"  model:  {result.summary['model_class']}")
    print(f"  corpus: {result.summary['corpus']['name']} ({result.summary['num_samples']} samples)")
    print(f"  {'metric':<28}{'mean':>14}{'std':>14}{'valid':>8}")
    for name, stats in result.summary["per_metric"].items():
        arrow = "(higher better)" if stats["higher_is_better"] else ""
        print(f"  {name:<28}{stats['mean']:>14.6g}{stats['std']:>14.6g}{stats['valid_count']:>8}  {arrow}")
    print(f"\nWritten to:\n  {result.per_sample_metrics_path}\n  {result.summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint on a corpus.")
    parser.add_argument("--checkpoint", required=True, help="saved model file to load and fingerprint")
    parser.add_argument("--corpus", required=True, help="eval corpus directory (fresh-process)")
    parser.add_argument(
        "--model", required=True, choices=sorted(MODEL_REGISTRY),
        help="model class to load the checkpoint into",
    )
    parser.add_argument("--out", default=None, help="results root (default: <project>/results)")
    parser.add_argument(
        "--save-audio", action="store_true",
        help="persist the re-rendered prediction WAV for a seeded random sample subset",
    )
    parser.add_argument(
        "--save-audio-n", type=int, default=20,
        help="cap on how many samples get their prediction audio saved (default: 20)",
    )
    args = parser.parse_args()

    _require_dexed()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model = MODEL_REGISTRY[args.model].model_class()
    model.load(checkpoint_path)
    corpus = RenderedCorpusDataset.load(args.corpus)

    print(f"--- Evaluating {args.model} on '{corpus.corpus_dir.name}' ({len(corpus)} samples) ---")
    result = Evaluator(corpus).evaluate(
        model,
        checkpoint_path=checkpoint_path,
        out_dir=args.out,
        save_audio=args.save_audio,
        save_audio_n=args.save_audio_n,
    )
    _print_table(result)


if __name__ == "__main__":
    main()
