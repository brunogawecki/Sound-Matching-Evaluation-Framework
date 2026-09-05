"""Build an out-of-domain (audio-only) evaluation corpus from real instrument recordings.

Every corpus in this project so far holds sounds the synthesizer made, so each target
comes with the parameters that produced it. This one does not: the targets are NSynth
instrument notes, which no synthesizer can be assumed to reach. What the corpus keeps is
the *render contract* -- copied verbatim from an in-domain reference corpus -- because
predictions still have to be re-rendered under exactly the contract the models were
trained against (D-EVAL / D-REPRO). See D-OOD in ``docs/DECISIONS.md``.

    python scripts/build_ood_corpus.py --nsynth-dir ~/nsynth \
        --reference-corpus full_preset-gen-vae_test_1500 --run-name nsynth_c4_dexed

    --nsynth-dir        directory holding nsynth-<split>/ subdirs   [default: NSYNTH_DIR]
    --reference-corpus  in-domain corpus to copy the render contract from     [REQUIRED]
    --run-name          output corpus name                                    [REQUIRED]
    --splits            NSynth splits to draw from            [default: train valid test]
    --pitch             MIDI pitch to keep                                    [default: 60]
    --velocity          NSynth velocities to keep (repeatable)               [default: 100]
    --out               dataset root                            [default: config.DATASET_DIR]

The pitch/velocity filter is not cosmetic: every model was trained under the D3 contract
(one note, C4, velocity 100), so a target at another pitch would measure a mismatch the
benchmark never intended to test.

All three NSynth splits are used by default. Their boundary exists to stop instrument
leakage between training *on NSynth* and evaluating *on NSynth*, and nothing here ever
trains on NSynth -- so it carries no weight, while the pitch/velocity filter is severe
enough (roughly one note per instrument) that `valid` + `test` alone yield only ~46 notes.

Build one corpus per synth -- the audio is identical, only the copied contract block
differs -- so each is self-describing and ``scripts/evaluate.py`` needs no extra flag.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import pandas as pd
import pyloudnorm
from scipy.io import wavfile
from tqdm import tqdm

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, dataset) import when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from dataset.ood_corpus import NO_TARGETS
from dataset.render_backends import DEFAULT_SYNTH

# Fields copied byte-for-byte from the reference corpus. The first four are the render
# contract the Evaluator hard-requires; the rest let the corpus rebuild its ParameterSpace
# with no live VST (D-SELFDESC). Copied rather than re-derived from config.py, which could
# have drifted since the reference corpus was built (D-EVAL).
_COPIED_FIELDS = (
    "render_settings",
    "renderer",
    "sample_rate",
    "default_params",
    "synth",
    "parameter_space",
    "subset_names",
)

# NSynth ships 16 kHz mono int16 WAVs, four seconds long, note held for the first three --
# which is the D3 render contract exactly. Asserted per file rather than assumed.
NSYNTH_SAMPLE_RATE = 16000
NSYNTH_NUM_SAMPLES = 64000

_METADATA_COLUMNS = [
    "sample_id",
    "audio_path",
    "nsynth_note_str",
    "nsynth_split",
    "instrument_family_str",
    "instrument_source_str",
    "pitch",
    "velocity",
    "rms",
    "loudness_lufs",
]


def _git_revision() -> Optional[str]:
    """The current commit of the framework repo, or None outside a checkout."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(config.BASE_DIR), stderr=subprocess.DEVNULL
        )
        return revision.decode().strip()
    except Exception:
        return None


def _resolve_corpus_dir(corpus: str) -> Path:
    """A run name under DATASET_DIR, or a path if one exists at that location."""
    candidate = Path(corpus)
    return candidate if candidate.exists() else Path(config.DATASET_DIR) / corpus


def _load_reference_summary(reference_dir: Path) -> Dict[str, object]:
    """Read the reference corpus's summary and verify it carries everything we copy."""
    summary_path = reference_dir / "run_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Reference corpus has no run_summary.json: {summary_path}")
    with open(summary_path) as summary_file:
        summary = json.load(summary_file)
    missing = [field for field in _COPIED_FIELDS if field not in summary and field != "synth"]
    if missing:
        raise SystemExit(
            f"{summary_path} is missing {missing}. An out-of-domain corpus copies the render "
            "contract from a reference corpus; pick one built by the current DatasetBuilder."
        )
    return summary


def record_from_note_str(note_str: str) -> Dict[str, object]:
    """Rebuild the fields we record from an NSynth filename.

    NSynth names every note ``<family>_<source>_<instrument>-<pitch>-<velocity>``, e.g.
    ``flute_acoustic_002-060-100``, which carries every column this corpus stores. Used
    when a split has audio but no ``examples.json`` -- the case when the archive was
    stream-extracted for the matching files rather than unpacked whole.
    """
    instrument_str, pitch, velocity = note_str.rsplit("-", 2)
    family, source, _ = instrument_str.split("_", 2)
    return {
        "note_str": note_str,
        "instrument_str": instrument_str,
        "instrument_family_str": family,
        "instrument_source_str": source,
        "pitch": int(pitch),
        "velocity": int(velocity),
    }


def _split_records(split_dir: Path, split: str) -> List[Dict[str, object]]:
    """Every note in one split, from ``examples.json`` if present, else from filenames."""
    examples_path = split_dir / "examples.json"
    if examples_path.exists():
        with open(examples_path) as examples_file:
            return list(json.load(examples_file).values())
    audio_dir = split_dir / "audio"
    if not audio_dir.is_dir():
        raise SystemExit(
            f"No NSynth {split} split at {split_dir}: neither examples.json nor audio/. "
            f"Download and extract nsynth-{split}.jsonwav.tar.gz into --nsynth-dir."
        )
    return [record_from_note_str(path.stem) for path in sorted(audio_dir.glob("*.wav"))]


def select_nsynth_notes(
    nsynth_dir: Path, splits: List[str], pitch: int, velocities: List[int]
) -> List[Dict[str, object]]:
    """Every NSynth note in ``splits`` matching the pitch/velocity filter.

    Returns the records with a ``nsynth_split`` key added, sorted by ``note_str`` so the
    corpus is a deterministic function of its arguments.
    """
    selected: List[Dict[str, object]] = []
    wanted = set(velocities)
    for split in splits:
        for record in _split_records(nsynth_dir / f"nsynth-{split}", split):
            if record["pitch"] == pitch and record["velocity"] in wanted:
                selected.append({**record, "nsynth_split": split})
    selected.sort(key=lambda record: record["note_str"])
    return selected


def load_and_resample(audio_path: Path, target_sample_rate: int) -> np.ndarray:
    """One NSynth WAV as float32 at ``target_sample_rate``.

    NSynth renders at 16 kHz; this project's contract is 22.05 kHz (D-METRIC-SR), so the
    audio is resampled up with a deterministic, anti-aliased resampler applied identically
    to every file. Nothing above 8 kHz is recovered -- the upsample only hands the metric
    panel the rate it expects. At the D3 contract 4.0 s maps to exactly 88200 samples, so
    the length is asserted rather than padded or truncated.
    """
    sample_rate, audio = wavfile.read(audio_path)
    if sample_rate != NSYNTH_SAMPLE_RATE or audio.shape[0] != NSYNTH_NUM_SAMPLES:
        raise ValueError(
            f"{audio_path} is {audio.shape[0]} samples at {sample_rate} Hz; expected "
            f"{NSYNTH_NUM_SAMPLES} at {NSYNTH_SAMPLE_RATE} Hz."
        )
    if audio.ndim != 1:
        raise ValueError(f"{audio_path} is not mono (shape {audio.shape}).")
    # NSynth ships int16; scale to [-1, 1) the way every other float32 WAV here is stored.
    # A float WAV (some mirrors repackage them) is already in range and passes through.
    if np.issubdtype(audio.dtype, np.integer):
        waveform = audio.astype(np.float32) / np.float32(np.iinfo(audio.dtype).max + 1)
    else:
        waveform = audio.astype(np.float32)
    return librosa.resample(
        waveform, orig_sr=sample_rate, target_sr=target_sample_rate, res_type="soxr_vhq"
    ).astype(np.float32)


def _integrated_loudness(meter: pyloudnorm.Meter, audio: np.ndarray) -> float:
    """Integrated loudness (LUFS), or -inf for audio the meter rejects as silent."""
    try:
        return float(meter.integrated_loudness(audio.astype(np.float64)))
    except Exception:
        return float("-inf")


def _corpus_median_loudness(corpus_dir: Path) -> Optional[float]:
    """The reference corpus's median loudness, read from its own metadata."""
    metadata_path = corpus_dir / "metadata.csv"
    if not metadata_path.exists():
        return None
    metadata = pd.read_csv(metadata_path)
    if "loudness_lufs" not in metadata.columns:
        return None
    finite = metadata["loudness_lufs"].replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.median()) if len(finite) else None


def build(
    nsynth_dir: Path,
    reference_dir: Path,
    run_dir: Path,
    splits: List[str],
    pitch: int,
    velocities: List[int],
) -> Dict[str, object]:
    """Write the out-of-domain corpus and return its run summary."""
    reference_summary = _load_reference_summary(reference_dir)
    sample_rate = int(reference_summary["sample_rate"])
    expected_num_samples = round(
        float(reference_summary["render_settings"]["duration_sec"]) * sample_rate
    )

    notes = select_nsynth_notes(nsynth_dir, splits, pitch, velocities)
    if not notes:
        raise SystemExit(
            f"No NSynth note in splits {splits} has pitch {pitch} and velocity in {velocities}."
        )

    (run_dir / "audio").mkdir(parents=True, exist_ok=True)
    meter = pyloudnorm.Meter(sample_rate)
    rows: List[Dict[str, object]] = []
    for note in tqdm(notes, desc="Converting", unit="note"):
        source_path = nsynth_dir / f"nsynth-{note['nsynth_split']}" / "audio" / f"{note['note_str']}.wav"
        audio = load_and_resample(source_path, sample_rate)
        if audio.shape[0] != expected_num_samples:
            raise ValueError(
                f"{source_path} resampled to {audio.shape[0]} samples, but the reference "
                f"contract is {expected_num_samples}. The reference corpus's render duration "
                "does not match NSynth's 4.0 s clips."
            )
        sample_id = f"sample_{len(rows):06d}"
        relative_path = f"audio/{sample_id}.wav"
        wavfile.write(str(run_dir / relative_path), sample_rate, audio)
        rows.append(
            {
                "sample_id": sample_id,
                "audio_path": relative_path,
                "nsynth_note_str": note["note_str"],
                "nsynth_split": note["nsynth_split"],
                "instrument_family_str": note["instrument_family_str"],
                "instrument_source_str": note["instrument_source_str"],
                "pitch": note["pitch"],
                "velocity": note["velocity"],
                "rms": float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))),
                "loudness_lufs": _integrated_loudness(meter, audio),
            }
        )

    pd.DataFrame(rows, columns=_METADATA_COLUMNS).to_csv(run_dir / "metadata.csv", index=False)

    # Audio metrics compare raw audio (D-METRIC-NORM), so the level gap between an
    # out-of-domain source and this synth's renders lands directly in the two loudness
    # metrics. Recorded here, not just printed, so the caveat stays a measured number.
    corpus_loudness = _corpus_median_loudness(run_dir)
    reference_loudness = _corpus_median_loudness(reference_dir)

    summary: Dict[str, object] = {
        "run_name": run_dir.name,
        "num_samples": len(rows),
        # The marker dataset.ood_corpus.load_eval_corpus dispatches on. Its presence is
        # what tells the Evaluator the parameter axis is undefined here (D-OOD).
        "targets": NO_TARGETS,
        "domain": "out_of_domain",
        "render_process": "fresh",
        **{field: reference_summary[field] for field in _COPIED_FIELDS if field in reference_summary},
        # A reference corpus predating the "synth" key is Dexed (D-SELFDESC). Resolve that
        # here rather than inheriting the gap: this corpus is built today and has no reason
        # to lean on the back-compat default.
        "synth": reference_summary.get("synth", DEFAULT_SYNTH),
        "source": {
            "method": "out_of_domain",
            "dataset": "nsynth",
            "splits": list(splits),
            "pitch": pitch,
            "velocities": list(velocities),
            "count": len(rows),
            "source_sample_rate": NSYNTH_SAMPLE_RATE,
            "resampler": "librosa.resample(res_type='soxr_vhq')",
            "reference_corpus": reference_dir.name,
            "reference_corpus_git_revision": reference_summary.get("git_revision"),
            "median_loudness_lufs": corpus_loudness,
            "reference_median_loudness_lufs": reference_loudness,
        },
        "git_revision": _git_revision(),
    }
    with open(run_dir / "run_summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an out-of-domain, audio-only evaluation corpus from NSynth."
    )
    parser.add_argument(
        "--nsynth-dir", default=config.NSYNTH_DIR,
        help="directory holding nsynth-<split>/ subdirs (default: config.NSYNTH_DIR)",
    )
    parser.add_argument(
        "--reference-corpus", required=True,
        help="in-domain corpus to copy the render contract from (run name under DATASET_DIR, or a path)",
    )
    parser.add_argument("--run-name", required=True, help="output corpus name")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"],
        help="NSynth splits to draw from (default: all three -- see D-OOD)",
    )
    parser.add_argument("--pitch", type=int, default=60, help="MIDI pitch to keep (default: 60, C4)")
    parser.add_argument(
        "--velocity", type=int, nargs="+", default=[100], dest="velocities",
        help="NSynth velocities to keep (default: 100)",
    )
    parser.add_argument("--out", default=None, help="dataset root (default: config.DATASET_DIR)")
    args = parser.parse_args()

    nsynth_dir = Path(args.nsynth_dir).expanduser()
    reference_dir = _resolve_corpus_dir(args.reference_corpus)
    if not reference_dir.exists():
        raise SystemExit(f"Reference corpus not found: {reference_dir}")
    output_root = Path(args.out) if args.out else Path(config.DATASET_DIR)
    run_dir = output_root / args.run_name

    print(
        f"--- Building '{args.run_name}' from NSynth {args.splits} "
        f"(pitch {args.pitch}, velocity {args.velocities}) ---"
    )
    summary = build(nsynth_dir, reference_dir, run_dir, args.splits, args.pitch, args.velocities)

    corpus_loudness = summary["source"]["median_loudness_lufs"]
    reference_loudness = summary["source"]["reference_median_loudness_lufs"]
    print(f"\nWrote {summary['num_samples']} samples to {run_dir}")
    print(f"  synth (from reference): {summary.get('synth', 'dexed')}")
    print(f"  render contract:        {summary['render_settings']} @ {summary['sample_rate']} Hz")
    if corpus_loudness is not None and reference_loudness is not None:
        print(
            f"  median loudness:        {corpus_loudness:.1f} LUFS "
            f"vs {reference_loudness:.1f} LUFS in {reference_dir.name} "
            f"(offset {corpus_loudness - reference_loudness:+.1f} dB)"
        )


if __name__ == "__main__":
    main()
