# The preset-gen-vae port — networks, models, and modules

How the paper *"Improving Synthesizer Programming from VAE Latent Space"* (Le Vaillant et al.,
DAFx 2021, `paper_repos/preset-gen-vae/`) maps onto the code in the `models/presetgen_vae/`
package -- one paper, one package, one file per role: `network.py` (front-end + VAE),
`realnvp.py` (the ported flow), `families.py` (the registered model wrappers), and
`lightning_module.py` (the training-only loss recipe). Read this
if the split between "network", "model", and "module" is unclear, or you want to know which class is
which part of the paper. Design rationale (D1, D-MELNORM, D-KIND, D-METRIC-SR) lives in
`docs/DECISIONS.md`; this doc is just the map.

## Three words that mean different things

The same net gets wrapped three times for three jobs. Keep them straight:

| Word | What it is | Examples |
|------|-----------|----------|
| **network** | The raw neural net — layers + a `forward`. Pure PyTorch. | `PresetGenVAENetwork` |
| **model / family** | The benchmark wrapper: `fit` / `save` / `load` / `predict`. What the pipeline scripts drive. | `PresetGenVAEMLPRegressor` |
| **(Lightning) module** | A *training-time-only* wrapper defining the loss + training step. Discarded after `fit`. | `LightningVAERegressor` |

One **model** contains one **network** and, only while training, hands it to one **module**.

## The paper builds ONE architecture

Not several models — one design (paper Figure 1), run in many configurations. Four pieces:

```
 audio  →  [ mel spectrogram x ]
                     │
                ┌────▼────┐
                │ ENCODER │   CNN, squeezes the spectrogram down
                └────┬────┘
                     │   produces a latent DISTRIBUTION q(z|x): a mean mu + spread sigma
                ┌────▼────┐
                │ LATENT  │   draw a sample z from that little Gaussian cloud
                └────┬────┘
              ┌──────┴───────┐
         ┌────▼────┐    ┌────▼─────┐
         │ DECODER │    │REGRESSOR │
         └────┬────┘    └────┬─────┘
              │              │
      reconstructed      predicted synth
      spectrogram x̂      parameters v̂
```

- **Encoder** — spectrogram → compact latent. Not a point but a Gaussian cloud (mean `mu`, spread
  `sigma`). This is the "variational" part.
- **Latent** — sample `z` from that cloud.
- **Decoder** — rebuild the *original spectrogram* from `z`. This is the "autoencoder" part; its job
  is to force the latent to actually capture the sound.
- **Regressor** — a *second* head off the same latent that predicts the synth parameters `v̂`. This
  is the part we care about for synth programming.

The paper's bet: forcing the latent to be good enough to reconstruct the sound (decoder) makes
predicting parameters *from that latent* better than predicting them straight from the spectrogram.

## Why it looks like "several models"

Because that one architecture is run with different knobs for the experiments (paper Tables 1–2).
These are settings, not separate models:

1. **Regressor type** — a **RealNVP flow** (their headline) or a plain **MLP** (simpler, also tested).
   **Both keep the full VAE** (encoder + latent + decoder + reconstruction + KL); only the head that
   reads the latent changes. The paper's "Flow" and "MLP" rows in Table 1 are *this* knob — neither is
   a no-VAE model.
2. **Parameter representation** — `Num` / `NumCat` / `NumCat++` (how categoricals are encoded).
3. **Input channels** — one spectrogram (Fig. 1) vs six-notes-at-once (Fig. 3, Section 4).

## Training vs inference flow

**Training** — all four pieces active:

```
x → encoder → (mu, sigma) → sample z → decoder → x̂      reconstruction loss: MSE(x, x̂)
                               │
                               └──→ regressor → v̂         controls loss vs true params
        plus a KL term pulling (mu, sigma) toward a standard Gaussian
   total loss = reconstruction + beta · KL + controls
```

**Inference** (program the synth from a sound) — the decoder is **dropped** (paper Section 5.1):

```
x → encoder → take mu (no sampling) → regressor → v̂ → load into Dexed
```

The decoder exists only to shape the latent during training. At predict time you go
audio → latent → parameters; you never reconstruct a spectrogram. (A third mode — *generation* —
samples new latents to invent presets; the VAE can do it, but the benchmark does not measure it.)

## How the code maps onto this

The port is built **in stages**. Stage 2 is the paper's VAE with an MLP regressor; Stage 3a swaps
the head for the RealNVP flow regressor (issue #35); Stage 3b (future) adds the latent RealNVP flow
completing the full model (issue #36). Both regressor variants share `PresetGenVAENetwork` (a
`regressor_architecture` knob) and `LightningVAERegressor`.

| Stage | Network | Model (family) | Trained by | = which paper piece |
|-------|---------|----------------|------------|---------------------|
| **2** | `PresetGenVAENetwork` (`mlp` head) | `PresetGenVAEMLPRegressor` | `LightningVAERegressor` | the paper's VAE + **MLP** regressor (≈ the "MLP" rows of Table 1), with closed-form KL |
| **3a** *(current)* | `PresetGenVAENetwork` (`flow` head, `realnvp.py`) | `PresetGenVAEFlowRegressor` | `LightningVAERegressor` | the paper's VAE + **flow** regressor (≈ the "Flow" rows), still closed-form KL |
| **3b** *(future, #36)* | *(+ latent RealNVP flow)* | *(same families)* | *(flow-KL module)* | latent flow z0 → zK + Monte-Carlo KL = full `FlVAE2` |

The two regressor variants are **separate registry entries**, so they stay separately trainable and
evaluable — mirroring the paper's MLP-vs-Flow comparison.

- **Stage 2 ≈ the paper's "MLP regression" model** — the full VAE (encoder + latent + decoder +
  reconstruction + KL) with an MLP head (`3l1024`) off the latent.
- **Stage 3a ≈ the paper's "Flow regression" model** — same VAE, but the head is a RealNVP flow
  (`realnvp_6l300`) used feed-forward. Invertible, so `latent_dimension` is pinned to
  `ml_dimension` at fit time (the paper's build-time assert). The nflows pieces the paper uses are
  ported as plain torch in `models/presetgen_vae/realnvp.py` (the head runs at predict time, so it cannot hide
  behind the training-only lazy imports).
- **Stage 3b** (their headline `FlVAE2`) — adds a RealNVP flow on the latent itself and replaces
  the closed-form KL with the Monte-Carlo estimate using the flow's log-det-Jacobian.

(Numbering starts at 2 because an earlier Stage 1 — a no-VAE encoder→regressor baseline, to isolate
what the VAE machinery buys — was built and then removed as out of scope. The discriminative slot in
the benchmark is covered by `Sound2SynthSpectrogramRegressor`.)

## The two regression architectures: MLP vs flow

The paper's Table 1 compares two configurations of the regressor head. Everything else is
shared: the same mel-dB front-end, the same `speccnn8l1_bn` encoder and decoder, the same
Gaussian latent, the same joint loss (reconstruction + beta·KL + controls), the same
optimizer recipe. Only the map latent → parameters changes. Our two registry entries
(`PresetGenVAEMLPRegressor` / `PresetGenVAEFlowRegressor`) mirror exactly this knob.

**MLP regression** (`mlp_3l1024`, `model/regression.py` `MLPRegression`): three
fully-connected layers of 1024 units, ReLU each, batch-norm + dropout on all but the two
deepest layers, then a plain linear to the parameter vector. A generic function
approximator: any latent size works, so `dim_z` stays at the paper's 256.

**Flow regression** (`flow_realnvp_6l300`, `model/regression.py` `FlowRegression`): a
RealNVP normalizing flow — six affine coupling layers, each conditioned by a 2-block
residual network with 300 hidden features. Used purely feed-forward here (the log-det
output is discarded). The trade the paper makes: a flow is *invertible*, so besides
predicting parameters from a latent it can also map a *known preset back into the latent
space*. The paper uses that inverse for its interactive preset-morphing interface
(Section 5.2); our benchmark never calls it. Invertibility costs a constraint: input and
output must have the same width, so `dim_z` must equal the learnable preset vector length
(the assert in `model/build.py:70`). That is why `PresetGenVAEFlowRegressor` has no
`latent_dimension` constructor knob and pins it to `ml_dimension` at fit time
(`families.py` `PresetGenVAEFlowRegressor._build_architecture_hparams`), while the MLP
family keeps the free default of 256.

Shared vs different, at a glance:

| | MLP regression | Flow regression |
|---|---|---|
| VAE (front-end, encoder, decoder, latent, loss) | identical | identical |
| Head | 3 FC layers × 1024, ReLU | 6 RealNVP couplings, 300 hidden features |
| Latent size | free (paper: 256) | forced = parameter vector length |
| Invertible (preset → latent) | no | yes (unused in our benchmark) |
| Our family | `PresetGenVAEMLPRegressor` | `PresetGenVAEFlowRegressor` |

One nuance for reading the paper: its headline model `FlVAE2` stacks a *second* RealNVP on
the latent itself (`FlowVAE`, `model/VAE.py:69`) on top of the flow regressor, replacing
the closed-form KL with a Monte-Carlo estimate. Both of our families currently sit on the
plain Gaussian latent (`BasicVAE`, closed-form KL); the latent flow is Stage 3b (issue #36).

## Walkthrough: our code ↔ the paper's code

Piece-by-piece counterpart map. Ours under `models/presetgen_vae/`, paper under
`paper_repos/preset-gen-vae/`. Line numbers are as of this writing; names are the stable
reference.

The map is backed by **parity tests** (`tests/test_paper_parity.py`): each network
component is built from both codebases, the paper's randomly-initialized weights are
transplanted into ours, and the outputs on identical inputs are asserted numerically
equal — proving the two implementations compute the same function. Encoder, decoder, and
beta warmup always run; the flow / MLP-head / KL tests need the paper's `nflows`
dependency and skip unless it is installed (`pip install nflows --no-deps`, dev-only).

| Piece | Ours | Paper |
|---|---|---|
| audio → mel-dB spectrogram | `network.py` `_compute_mel_db_spectrogram` + `_build_mel_filterbank` | `utils/audio.py` `MelSpectrogram` (73–87), applied per item in `data/abstractbasedataset.py` (121–133) |
| dB min/max normalization endpoints | `network.py` `measure_corpus_mel_db_range` (D-MELNORM) | dataset-wide `spec_stats` min/max, `data/abstractbasedataset.py` (129–131, 311) |
| encoder CNN (enc1–enc8) | `network.py` `_build_spectrogram_cnn` | `model/encoder.py` `SpectrogramCNN`, `'speccnn8l1_bn'` branch (233–259) + `SpectrogramEncoder.features_mixer_cnn` (54–70) |
| conv / tconv building block | `network.py` `_conv2d_block` / `_tconv2d_block` | `model/layer.py` `Conv2D` / `TConv2D` |
| encoder MLP → mu, logvar | `network.py` `encoder_mlp` + `_encode` | `model/encoder.py` (85, 102–108) |
| reparameterization (sample z; eval uses mu) | `network.py` `_reparameterize` | `model/VAE.py` `BasicVAE.forward` (51–58) |
| decoder MLP + 1×1 un-mixer + CNN (dec1–dec8) | `network.py` `decoder_mlp` + `_build_decoder_cnn` | `model/decoder.py` `SpectrogramDecoder` (57–92) + decoder `SpectrogramCNN` (199–220) |
| MLP regressor head | `network.py` `_build_regressor` | `model/regression.py` `MLPRegression` (61–102) |
| flow regressor head | `realnvp.py` `RealNVP` | `model/regression.py` `FlowRegression` (105–189) + `model/flows.py` `CustomRealNVP` (42–90) |
| flow internals (coupling, conditioner, flow BN) | `realnvp.py` `AffineCouplingLayer`, `ResidualNetwork`, `FlowBatchNorm` | the `nflows` package (`AffineCouplingTransform`, `nets.ResidualNet`, `transforms.normalization.BatchNorm`) — a dependency there, ported as plain torch here |
| flow ⇒ `dim_z` == parameter vector length | `families.py` `PresetGenVAEFlowRegressor._build_architecture_hparams` | `model/build.py` assert (70) |
| assembled network | `network.py` `PresetGenVAENetwork` | `model/VAE.py` `BasicVAE` + `model/extendedAE.py` `ExtendedAE`, wired in `model/build.py` `build_extended_ae_model` (55–80) |
| benchmark wrapper (fit/save/load/predict) | `families.py` (via `models/base_deep_model.py`) | none — `train.py` / `eval.py` drive the raw networks directly |
| training step + loss assembly | `lightning_module.py` `LightningVAERegressor._shared_step` | `train.py` epoch loop (222–248) |
| reconstruction loss (MSE) | `F.mse_loss` in `_shared_step` | `model/loss.py` `L2Loss` / `nn.MSELoss` via `train.py:106` |
| KL term | `models/training/loss.py` `gaussian_kl_divergence` | `model/loss.py` `GaussianDkl` (46–66) |
| controls (parameter) loss | `models/training/loss.py` `ParameterLoss` | `model/loss.py` `SynthParamsLoss` (73–186) |
| beta warmup | `lightning_module.py` `linear_warmup` | `LinearDynamicParam` schedule, `train.py:150` |
| hyperparameters | `cluster/training_configs/presetgen_full_config.yaml` | `config.py` (16–118) |
| latent RealNVP flow + Monte-Carlo KL | not ported yet (issue #36) | `model/VAE.py` `FlowVAE` (69–193) |

Notes on the non-obvious rows:

- **Front-end placement.** The paper precomputes mel-dB spectrograms offline and stores
  them; the dataset serves spectrograms. Our corpus stores raw audio (D-SELFDESC), so the
  same STFT → mel → dB → min-max math runs *inside the network* on every forward. The
  math is the same; the min-max endpoints come from one corpus pass at fit time
  (`measure_corpus_mel_db_range`) exactly as the paper's `spec_stats` come from a dataset
  pass, and the same `-1 + 2(x - min)/(max - min)` formula is applied.
- **Encoder split.** The paper splits enc1–enc6 into `SpectrogramCNN` and builds enc7 +
  enc8 in `SpectrogramEncoder.features_mixer_cnn` (machinery for its multi-channel
  six-note variant, Fig. 3). For the single-channel case we port, the composed layer
  sequence is one straight CNN, so ours builds all eight in `_build_spectrogram_cnn`.
  Caution when reading the paper's code: the raw `'speccnn8l1_bn'` listing in
  `model/encoder.py:258` ends in `Conv2D(512 -> 1024, 1x1)`, but that enc8 is dead code —
  under the shipped config (`stack_specs_deepest_features_mix = False`, single-channel)
  `SpectrogramEncoder` always rebuilds enc7/enc8 in `features_mixer_cnn`, whose 1×1 width
  is `mixer_1x1conv_ch = 2048` (`model/encoder.py:46, 59–70`), mirrored by the decoder
  (`model/decoder.py:31, 62, 72`). Our port originally copied the dead 1024 listing; the
  review below caught it and enc8 now emits 2048, matching the paper's actual runs (the
  decoder side follows automatically from the inferred encoder output shape).
- **Output activation.** Both paper heads end in `PresetActivation` (Hardtanh on
  numerical, optional softmax on categorical, `model/regression.py:20`). We drop it
  deliberately: the framework contract is raw outputs (continuous floats + categorical
  logits) into `ParameterLoss`, same as `Sound2SynthSpectrogramRegressor`.
- **Flow port fidelity.** `realnvp.py` reproduces the exact `CustomRealNVP` schedule:
  alternating checkerboard masks (`mask[::2] = -1`, flipped each layer), 2 residual
  blocks per conditioner, sigmoid-constrained scale (`sigmoid(x + 2) + 1e-3`), flow
  batch-norm between couplings, and no batch-norm / dropout on the last two couplings
  (`model/flows.py:81, 87`). Forward direction only — `FlowRegression` runs the flow
  feed-forward (`fast_forward_flow=True`) and discards the log-det, as we do; the
  per-sample log-det is still exposed (`forward_with_log_determinant`) for issue #36.
- **Loss parity.** `gaussian_kl_divergence` is a line-for-line port of `GaussianDkl`
  (sum over latent dims, mean over batch, optional division by `dim_z`);
  `normalize_latent_loss: true` + `beta: 0.2` matches the paper's
  `normalize_losses = True`, `beta = 0.2`, warmup 0.1 → 0.2 over 25 epochs.
  `ParameterLoss` plays `SynthParamsLoss`'s role (MSE on continuous + 0.2-weighted
  cross-entropy on categorical blocks) but is our own implementation over
  `ParameterSpace.loss_slices` — see the categorical-scheme caveat below.

### Deviations found in review (beyond the intentional ones)

- **Bottleneck width (found and fixed).** The port originally used
  `Conv2D(512 → 1024, 1×1)` for enc8, copied from the paper's raw `'speccnn8l1_bn'`
  listing — which turned out to be dead code; the paper's composed encoder actually runs
  a 2048-wide 1×1 mixer (see the encoder-split note above). Fixed: enc8 now emits 2048,
  so the encoder MLP reads 2048·3·4 = 24576 features exactly as the paper's runs did.
  Checkpoints trained before the fix cannot be loaded (the hparams-pinned shapes differ).
- **Mel band edges.** We pass `fmin=30, fmax=11000` — the values the paper's dataset
  *declares* (`data/abstractbasedataset.py:30`) but never implements (marked TODO there
  and in `utils/audio.py:76`; `librosa.feature.melspectrogram` is called without them,
  `utils/audio.py:85`). The paper's runs therefore used librosa defaults: 0 Hz to
  sr/2 = 11025 Hz. Negligible in practice (a few lowest/highest mel bins), but we match
  the paper's *intent*, not its execution.
- **Hann-window magnitude scaling omitted.** The paper divides STFT magnitude by
  `max |rfft(hann)|` before dB (`utils/audio.py:31, 46`) — a constant ≈ −54 dB offset.
  We skip it. Because both pipelines then min-max normalize with *measured* endpoints,
  the offset cancels out of the network input; only the −120 dB floor sits at a slightly
  different absolute signal level.
- **STFT window and padding mode.** The paper uses a symmetric Hann window
  (`torch.hann_window(n_fft, periodic=False)`, `utils/audio.py:30`) and
  `pad_mode='constant'`; ours uses the standard analysis choices, a periodic Hann and
  `pad_mode='reflect'`. Sub-percent magnitude differences across all frames (window) and
  in the two edge frames (padding). These, plus the two items above, are why the mel-dB
  front-end is documented rather than parity-tested (`tests/test_paper_parity.py`).
- **No useless-parameter masking in the controls loss.** `SynthParamsLoss` zeroes the
  loss on parameters that cannot affect the sound (e.g. an operator with output level 0,
  `model/loss.py:89, 119–126`). `ParameterLoss` has no such masking — every parameter in
  the D1 subset contributes equally. Part of the "our own categorical scheme" caveat.
- **Reconstruction size handling.** At the paper's fixed 257×347 contract the decoder
  lands on the input size exactly; ours renders any corpus length, so
  `_center_crop_or_pad` trims/pads the reconstruction by up to a few pixels to keep the
  MSE well-defined. At the paper's own geometry this is a no-op in height and ±2 frames
  in width.

## Caveats

- Reproduced numbers will **not** match the paper's tables, by design: **103** Dexed params (D1), not
  144; our own categorical scheme (close to `NumCat`, not `NumCat++`); a single train/test split, not
  5-fold; **dawdreamer** rendering, not RenderMan. We reproduce the *method*, not the *numbers*.
