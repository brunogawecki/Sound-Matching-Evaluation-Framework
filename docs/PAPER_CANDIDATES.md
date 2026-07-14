# Paper candidates (synth sound-matching / parameter estimation)

Working list of papers considered as architecture references for Layer 3 models, pulled from the
"Potentially for Evaluation" → "All info" table in the `master papers` Google Sheet (owned by Bruno).
Repo links are the load-bearing part of this doc — kept for quick access in future sessions, not a
substitute for reading the papers/repos before committing to an architecture.

Re-sync from the sheet: `docs/PAPER_CANDIDATES.md` mirrors
`https://docs.google.com/spreadsheets/d/1rahb64XF_uJjyoBkQC5RJuvR0y_gR5jZRhKExnp4v3o` (gid
`985398709`, "Potentially for Evaluation" tab, "All info" table, rows ~4–24).

## Discriminative / feed-forward (audio → params)

- **InverSynth II** (2023) — Dexed, TAL-NM, custom. Proxy + strided CNN, log-STFT spectrogram input,
  L2 + cross-entropy + mel-spectrogram-MAE loss. Direct lineage for issue #19 (InverSynth/preset-gen-vae
  lineage, lowest-risk). Repo: https://github.com/inversynth/InverSynth2
- **Sound2Synth** (2022) — Dexed FM. Multi-modal (STFT/mel/CQT/MFCC/stats) parallel NNs, cross-entropy +
  gradient-inspired weighting. Higher complexity/risk. Repo: https://github.com/Sound2Synth/Sound2Synth
- **Synthesizer Sound Matching Using Audio Spectrogram Transformer** (2024) — Massive synth, 1M random
  samples, plain Transformer over mel spectrogram, MSE loss. Repo not found (marked `?` in sheet).
- **Improving Semi-Supervised Differentiable Synth Sound Matching (SSSSM-DDSP)** (2023) — custom
  differentiable synth, CNN+RNN over mel spectrogram. Repo: https://github.com/hyakuchiki/SSSSM-DDSP
- **DiffMoog** (2024) — custom modular differentiable synth, MLP, signal-chain + param loss.
  Repo: https://github.com/aisynth/diffmoog

## Generative / VAE (relevant to Phase 5, D-FAMILIES)

- **preset-gen-vae** ("Improving Synthesizer Programming from VAE Latent Space", 2021) — Dexed, VAE over
  mel-spectrograms from log-magnitude STFT. Already vendored at `paper_repos/preset-gen-vae/`; source of
  the human DX7 corpus (`dexed_presets.sqlite`) used by Phase 4. Being ported into `models/presetgen_vae/`
  in stages — see `docs/PRESETGEN_VAE_PORT.md` for how its networks/models/modules map onto the paper.
  Repo: https://github.com/gwendal-lv/preset-gen-vae
- **Flow synthesizer** (2020) — Diva, VAE + normalizing flows, mel spectrogram, MSE + KL loss.
  Repo: https://github.com/acids-ircam/flow_synthesizer

## Transformer / RL / flow-matching (newer, higher-complexity references)

- **Neural Proxies for Sound Synthesizers** (2025) — Diva, Dexed, TAL-NM. RNN/MLP/Transformer proxy over
  params + deep audio embeddings (AudioMAE, CLAP, DAC, EfficientAT, OpenL3, Music2Latent, PaSST).
  Repo: https://github.com/pcmbs/synth-proxy
- **Audio Synthesizer Inversion in Symmetric Parameter Spaces w/ Equivariant Flow Matching** (2025) —
  Surge XT, normalizing flows / diffusion transformer over mel spectrogram.
  Repo: https://github.com/ben-hayes/synth-permutations
- **SynthRL** (2025) — Dexed + out-of-domain Surge XT. Transformer + RL fine-tuning, mel spectrogram,
  cross-entropy + spectrogram-based RL objective. Repo: https://github.com/argaaw/SynthRL
- **SynthCloner** (2025) — Serum preset conversion, factorized codec + Conv-BiLSTM. No repo found.

## Notes for the discriminative-model decision (issue #19)

ROADMAP Phase 4 names the target as "spectrogram→params, the InverSynth/preset-gen-vae lineage,
lowest-risk architecture" — points at **InverSynth II** and **preset-gen-vae** as the primary references,
with plain InverSynth (I) as the conceptual ancestor (CNN over log-STFT spectrogram, per-parameter
classification head). See [[project_layer3_started]] and `docs/ROADMAP.md` Phase 4 for current status.
