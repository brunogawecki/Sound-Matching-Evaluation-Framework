"""Measure RL-reward render throughput against worker count.

The SynthRLi training step renders one batch of patches per step, so render throughput
sets the epoch cost when the run is render-bound rather than GPU-bound. This sweeps
``num_render_workers`` and reports renders/sec plus the epoch time it implies, so
``--cpus-per-task`` can be sized from a measurement instead of a guess.

CPU-only -- no GPU needed, so it can run on any partition with free cores:

    srun -p pmem -c 96 --pty python scripts/benchmark_render_throughput.py \
        --corpus ~/corpora/full_preset-gen-vae_train --workers 8,16,32,64

Reads the render contract from the corpus (never config.py -- D-EVAL) and DEXED_PATH from
config.py (cluster.env / .env). Uses the reuse backend, matching synthrl_i_config.yaml.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.render_backends import ParallelInProcessRenderBackend, RenderSettings
from synth.parameter_space import ParameterSpace


def load_corpus(corpus_dir: Path) -> Dict:
    summary_path = corpus_dir / "run_summary.json"
    if not summary_path.exists():
        print(f"No run_summary.json at {summary_path}.")
        sys.exit(1)
    with open(summary_path) as summary_file:
        return json.load(summary_file)


def time_worker_count(
    patches: List[Dict[str, float]],
    settings: RenderSettings,
    renderer: str,
    num_workers: int,
    batch_size: int,
    num_batches: int,
) -> float:
    """Renders/sec at ``num_workers``, excluding pool spawn and plugin load."""
    backend = ParallelInProcessRenderBackend(
        settings, renderer=renderer, num_workers=num_workers
    )
    try:
        # Warm-up batch: pays the spawn + Dexed load once, outside the timed region.
        backend.render_batch(patches[:batch_size])

        start = time.perf_counter()
        for batch_index in range(num_batches):
            offset = (batch_index * batch_size) % (len(patches) - batch_size)
            backend.render_batch(patches[offset : offset + batch_size])
        elapsed = time.perf_counter() - start
    finally:
        backend.close()
    return (num_batches * batch_size) / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus dir (run_summary.json).")
    parser.add_argument("--workers", default="8,16,32,64", help="Comma-separated worker counts.")
    parser.add_argument("--batch-size", type=int, default=32, help="Patches per render_batch.")
    parser.add_argument("--batches", type=int, default=5, help="Timed batches per worker count.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = load_corpus(args.corpus)
    settings = RenderSettings(**summary["render_settings"])
    renderer = summary["renderer"]
    space = ParameterSpace.from_dict(summary["parameter_space"])
    num_samples = int(summary["num_samples"])

    worker_counts = [int(value) for value in args.workers.split(",")]
    rng = np.random.default_rng(args.seed)
    patches = [space.sample_uniform(rng) for _ in range(args.batch_size * (args.batches + 1))]

    # Mirrors the RL DataLoader: val_fraction 0.1 held out, drop-last not applied.
    steps_per_epoch = int(np.ceil(num_samples * 0.9 / args.batch_size))
    renders_per_epoch = steps_per_epoch * args.batch_size

    print(f"corpus            {args.corpus}")
    print(f"renderer          {renderer}  |  {settings.duration_sec}s per render")
    print(f"batch size        {args.batch_size}  ({args.batches} timed batches per setting)")
    print(f"steps/epoch       {steps_per_epoch}  ({renders_per_epoch} renders/epoch)")
    print()
    print(f"{'workers':>8}  {'renders/sec':>12}  {'sec/render':>11}  {'render-only epoch':>18}")
    print("-" * 56)

    for num_workers in worker_counts:
        renders_per_sec = time_worker_count(
            patches, settings, renderer, num_workers, args.batch_size, args.batches
        )
        epoch_minutes = renders_per_epoch / renders_per_sec / 60.0
        print(
            f"{num_workers:>8}  {renders_per_sec:>12.2f}  {1.0 / renders_per_sec:>11.3f}"
            f"  {epoch_minutes:>15.1f} min"
        )

    print()
    print("Job 1006799 ran at 34.4 min/epoch (3.13 s/step) with 8 workers. Rendering measured at")
    print("~29 ms per 32-patch batch there, so it is ~1% of a step: SynthRLi is NOT render-bound,")
    print("and raising num_render_workers alone will not move epoch cost. The step is dominated by")
    print("the per-sample reward loop and the REINFORCE loop over 103 parameter heads.")


if __name__ == "__main__":
    main()
