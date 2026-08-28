import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# VST Plugin Paths
DEXED_PATH = os.getenv("DEXED_PATH", "/Library/Audio/Plug-Ins/VST3/Dexed.vst3")
DIVA_PATH = os.getenv("DIVA_PATH", "/Library/Audio/Plug-Ins/VST3/Diva.vst3")
SURGE_XT_PATH = os.getenv("SURGE_XT_PATH", "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")

# Human preset collections. The Dexed database is Git LFS-backed (`git lfs pull` to
# materialize); the Diva corpus is a 1.4 GB manual download, see dataset/diva_preset_loader.py.
PRESETGEN_DB_PATH = os.getenv(
    "PRESETGEN_DB_PATH",
    str(Path(__file__).parent.absolute() / "paper_repos" / "preset-gen-vae" / "synth" / "dexed_presets.sqlite"),
)
DIVA_RAW_PATH = os.getenv(
    "DIVA_RAW_PATH",
    str(Path(__file__).parent.absolute() / "paper_repos" / "flow_synthesizer" / "diva_raw.zip"),
)

# Audio Rendering Defaults
SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", 22050))
BUFFER_SIZE = int(os.getenv("DEFAULT_BUFFER_SIZE", 128))
DURATION_SEC = float(os.getenv("DEFAULT_DURATION_SEC", 4.0))
NOTE_DURATION_SEC = float(os.getenv("DEFAULT_NOTE_DURATION_SEC", 3.0))

# MIDI Defaults
MIDI_NOTE = int(os.getenv("DEFAULT_MIDI_NOTE", 60))
VELOCITY = int(os.getenv("DEFAULT_VELOCITY", 100))

# File Export Paths
DATASET_DIR = BASE_DIR / "dataset"
AUDIO_OUT_DIR = DATASET_DIR / "audio"
METADATA_FILE = DATASET_DIR / "metadata.csv"

# Ensure directories exist
DATASET_DIR.mkdir(exist_ok=True)
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
