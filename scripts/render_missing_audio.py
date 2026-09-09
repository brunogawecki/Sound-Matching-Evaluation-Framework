"""Re-render a corpus's missing WAVs from its own metadata.csv.

Repair tool for a corpus whose metadata.csv and run_summary.json are complete but
whose audio/ directory is short -- an interrupted transfer, or a partial copy. Every
parameter vector is already recorded per row, so nothing is re-drawn: this renders
exactly the rows whose WAV is absent, under the render contract the corpus itself
records (D-SELFDESC). No preset source, no seed, no sampling.

``--verify N`` re-renders N rows that *do* have audio and reports how far the new
render lands from the stored one. Use it before a repair run: on a corpus built
in-process the reused wrapper leaks voice state between renders, so a re-render
started part-way through the corpus does not have to match bit for bit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, evaluation, dataset, models) import from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.render_backends import (
    DEFAULT_SYNTH,
    InProcessRenderBackend,
    RenderSettings,
    _synth_spec,
    make_wrapper,
)


def _load_corpus(corpus_dir: Path):
    with open(corpus_dir / "run_summary.json") as summary_file:
        summary = json.load(summary_file)
    df_metadata = pd.read_csv(corpus_dir / "metadata.csv")
    return summary, df_metadata


def _missing_rows(corpus_dir: Path, df_metadata: pd.DataFrame) -> List[int]:
    return [
        index for index, relative_path in enumerate(df_metadata["audio_path"])
        if not (corpus_dir / relative_path).exists()
    ]


def _row_parameters(row: pd.Series, subset_names: List[str], defaults: Dict[str, float]) -> Dict[str, float]:
    return {**defaults, **{name: float(row[name]) for name in subset_names}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="corpus directory to repair")
    parser.add_argument(
        "--verify", type=int, default=0, metavar="N",
        help="re-render the first N rows that already have audio and report the difference, then exit",
    )
    parser.add_argument(
        "--verify-ids", default=None, metavar="ID[,ID...]",
        help="re-render these sample_ids instead of the first N. Use a LATE id to test the "
             "case a repair actually hits: the stored render carried thousands of renders of "
             "leaked in-process voice state, a repair render carries none.",
    )
    parser.add_argument("--limit", type=int, default=None, help="render at most this many missing rows")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    summary, df_metadata = _load_corpus(corpus_dir)
    subset_names = summary["subset_names"]
    defaults = summary["default_params"]
    settings = RenderSettings(**summary["render_settings"])
    synth_name = summary.get("synth") or DEFAULT_SYNTH

    if summary.get("render_process") != "in-process":
        raise SystemExit(
            f"{corpus_dir} was built {summary.get('render_process')!r}; this tool only "
            "reproduces the in-process path. Rebuild it instead."
        )

    spec = _synth_spec(synth_name)
    with spec.open_output_suppressor():
        synth = make_wrapper(summary["renderer"], synth_name)
        backend = InProcessRenderBackend(synth, settings)

        if args.verify or args.verify_ids:
            present = [
                index for index, relative_path in enumerate(df_metadata["audio_path"])
                if (corpus_dir / relative_path).exists()
            ]
            if args.verify_ids:
                wanted = [name.strip() for name in args.verify_ids.split(",")]
                by_id = {df_metadata.iloc[index]["sample_id"]: index for index in present}
                absent = [name for name in wanted if name not in by_id]
                if absent:
                    raise SystemExit(f"No local audio for: {', '.join(absent)}")
                present = [by_id[name] for name in wanted]
            else:
                present = present[: args.verify]
            for index in present:
                row = df_metadata.iloc[index]
                rendered = backend.render(_row_parameters(row, subset_names, defaults))
                _, stored = wavfile.read(str(corpus_dir / row["audio_path"]))
                length = min(len(rendered), len(stored))
                difference = np.abs(rendered[:length] - stored[:length])
                peak = float(np.abs(stored).max()) or 1.0
                print(
                    f"{row['sample_id']}: max |diff| {difference.max():.3e} "
                    f"({100 * difference.max() / peak:.4f}% of peak), "
                    f"rms {np.sqrt((difference ** 2).mean()):.3e}"
                )
            return

        missing = _missing_rows(corpus_dir, df_metadata)
        if args.limit is not None:
            missing = missing[: args.limit]
        if not missing:
            print(f"{corpus_dir}: nothing missing, {len(df_metadata)} rows all have audio.")
            return

        print(f"{corpus_dir}: {len(missing)} of {len(df_metadata)} rows missing audio.")
        (corpus_dir / "audio").mkdir(parents=True, exist_ok=True)
        for index in tqdm(missing, desc="Rendering", unit="sample"):
            row = df_metadata.iloc[index]
            audio = backend.render(_row_parameters(row, subset_names, defaults))
            wavfile.write(
                str(corpus_dir / row["audio_path"]), int(summary["sample_rate"]),
                audio.astype(np.float32),
            )
        backend.close()
    print(f"Done. {len(_missing_rows(corpus_dir, df_metadata))} row(s) still missing audio.")


if __name__ == "__main__":
    main()
