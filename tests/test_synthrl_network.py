"""Forward-pass shape tests for :class:`SynthRLNetwork` (no VST, no training).

A small STFT + few mels keep the strided conv reducer from collapsing the tiny
spectrogram and the run fast on CPU. Skips cleanly when ``torch`` / ``librosa`` are
absent (front-end dependencies).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")  # mel filterbank (front-end dependency)

from models.synthrl.network import SynthRLNetwork
from models.synthrl.representation import SynthRLRepresentation
from synth.parameter_space import ParameterSpace, ParameterSpecification

SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE  # 1 s

# Small front-end so the conv reducer survives on a tiny spectrogram.
TINY_KWARGS = dict(
    num_audio_samples=NUM_SAMPLES,
    sample_rate=SAMPLE_RATE,
    n_fft=512,
    hop_length=128,
    win_length=512,
    n_mels=64,
    mel_fmax=8000.0,
    d_model=32,
    num_conv_layers=3,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    feedforward_dim=64,
    dropout=0.0,
)


def make_representation() -> SynthRLRepresentation:
    space = ParameterSpace([
        ParameterSpecification(name="CONT A", kind="continuous", default=0.25),
        ParameterSpecification(name="CAT B", kind="categorical", options=[0.0, 0.5, 1.0], default=0.5),
        ParameterSpecification(name="CONT C", kind="continuous", bounds=(0.2, 0.8), default=0.5),
        ParameterSpecification(name="CAT D", kind="categorical", options=[0.0, 1.0], default=0.0),
    ])
    return SynthRLRepresentation(space, num_bins=25)


def test_forward_emits_flat_class_logits():
    representation = make_representation()
    network = SynthRLNetwork(class_counts=representation.class_counts, **TINY_KWARGS)
    network.eval()
    batch = torch.zeros(5, NUM_SAMPLES)
    with torch.no_grad():
        logits = network(batch)
    assert logits.shape == (5, representation.total_class_dimension)


def test_forward_logits_decode_through_the_representation():
    representation = make_representation()
    network = SynthRLNetwork(class_counts=representation.class_counts, **TINY_KWARGS)
    network.eval()
    with torch.no_grad():
        logits = network(torch.zeros(1, NUM_SAMPLES))
    # The single-sample logit vector decodes to a valid synth-side dict.
    synth_dict = representation.class_logits_to_synth_dict(logits.squeeze(0).numpy())
    assert set(synth_dict) == set(representation.names)


def test_one_query_per_parameter_and_head_widths_match():
    representation = make_representation()
    network = SynthRLNetwork(class_counts=representation.class_counts, **TINY_KWARGS)
    assert network.parameter_queries.shape[0] == len(representation.class_counts)
    assert [head.out_features for head in network.class_heads] == representation.class_counts


def test_positional_encoding_matches_conv_feature_map():
    representation = make_representation()
    network = SynthRLNetwork(class_counts=representation.class_counts, **TINY_KWARGS)
    with torch.no_grad():
        feature_map = network.conv_reducer(network.mel_db_spectrogram(torch.zeros(1, NUM_SAMPLES)))
    # PE is [d_model, height, width] and lines up with the conv reducer's output map.
    assert network._positional_encoding.shape[1:] == feature_map.shape[2:]
