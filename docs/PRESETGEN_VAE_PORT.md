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

Not several models — one design (paper Figure 1), run in many configurations. Five pieces:

```
 audio  →  [ mel spectrogram x ]
                     │
                ┌────▼────┐
                │ ENCODER │   CNN, squeezes the spectrogram down
                └────┬────┘
                     │   produces a latent DISTRIBUTION q(z0|x): a mean mu + spread sigma
                ┌────▼────┐
                │ LATENT  │   draw a sample z0 from that little Gaussian cloud
                └────┬────┘
                ┌────▼─────┐
                │LATENT    │  RealNVP: bend the Gaussian cloud into a richer
                │FLOW z0→zK│  distribution, and record how much it stretched space
                └────┬─────┘
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
- **Latent** — sample `z0` from that cloud.
- **Latent flow** — an invertible RealNVP that reshapes the simple Gaussian `z0` into a more
  expressive `zK` (paper §2.2.2). A plain Gaussian is a poor stand-in for the true posterior; the
  flow buys flexibility. The price: `q(zK|x)` has no closed form, so the KL term is replaced by a
  one-sample Monte-Carlo estimate that needs the flow's log-determinant (paper Eq. 5).
- **Decoder** — rebuild the *original spectrogram* from `zK`. This is the "autoencoder" part; its job
  is to force the latent to actually capture the sound.
- **Regressor** — a *second* head off the same `zK` that predicts the synth parameters `v̂`. This
  is the part we care about for synth programming.

The paper's bet: forcing the latent to be good enough to reconstruct the sound (decoder) makes
predicting parameters *from that latent* better than predicting them straight from the spectrogram.

**There are two RealNVPs in this model, doing different jobs.** The *latent* flow (`T`, above) makes
the posterior more expressive. The *regression* flow (`U`, one of the two head choices below) maps
the latent to synth parameters. Do not confuse them — they are separate networks with separate
weights, and the paper even builds them from different `nflows` classes.

## Why it looks like "several models"

Because that one architecture is run with different knobs for the experiments (paper Tables 1–2).
These are settings, not separate models:

1. **Regressor type** — a **RealNVP flow** (their headline) or a plain **MLP** (simpler, also tested).
   **Both keep the full VAE**, latent flow included; only the head that reads `zK` changes. The
   paper's "Flow" and "MLP" rows in Table 1 are *this* knob — neither is a no-VAE model, and neither
   drops the latent flow (§3.4.2: only the invertible transform `U` is replaced).
2. **Parameter representation** — `Num` / `NumCat` / `NumCat++` (how categoricals are encoded).
3. **Input channels** — one spectrogram (Fig. 1) vs six-notes-at-once (Fig. 3, Section 4).

Their repo also contains a `BasicVAE` — a plain Gaussian latent with a closed-form KL, no latent
flow. It is a code path, **not a model the paper reports**. Our network can still be built that way
(`latent_flow_layers=0`) as a free ablation, but no family registers it.

## Training vs inference flow

**Training** — all five pieces active:

```
x → encoder → (mu, sigma) → sample z0 → flow → zK → decoder → x̂   reconstruction loss: MSE(x, x̂)
                                                │
                                                └──→ regressor → v̂  controls loss vs true params
        plus a latent term pulling the latent toward a standard Gaussian
   total loss = reconstruction + beta · latent + controls
```

**Inference** (program the synth from a sound) — the decoder is **dropped** (paper Section 5.1):

```
x → encoder → take mu (no sampling) → flow → zK → regressor → v̂ → load into Dexed
```

The decoder exists only to shape the latent during training. At predict time you go
audio → latent → parameters; you never reconstruct a spectrogram. (A third mode — *generation* —
samples new latents to invent presets; the VAE can do it, but the benchmark does not measure it.)

## How the code maps onto this

The port was built **in stages**, all now complete. Stage 2 was the paper's VAE with an MLP
regressor; Stage 3a swapped the head for the RealNVP flow regressor (issue #35); Stage 3b added the
latent RealNVP flow and the Monte-Carlo latent loss (issue #36), which is what makes the two
families the two models the paper actually reports. Both share `PresetGenVAENetwork` (a
`regressor_architecture` knob) and `LightningVAERegressor`.

| Stage | Network | Model (family) | Trained by | = which paper piece |
|-------|---------|----------------|------------|---------------------|
| **2** | `PresetGenVAENetwork` (`mlp` head) | `PresetGenVAEMLPRegressor` | `LightningVAERegressor` | the paper's VAE + **MLP** regressor, with closed-form KL |
| **3a** | `PresetGenVAENetwork` (`flow` head, `realnvp.py`) | `PresetGenVAEFlowRegressor` | `LightningVAERegressor` | the paper's VAE + **flow** regressor, still closed-form KL |
| **3b** *(current)* | *(+ latent RealNVP flow)* | *(same two families)* | `LightningVAERegressor` (Monte-Carlo latent term) | latent flow z0 → zK + Monte-Carlo latent loss ⇒ the two models of Table 1 |

The two regressor variants are **separate registry entries**, so they stay separately trainable and
evaluable — mirroring the paper's MLP-vs-Flow comparison. There are exactly two, because the paper
reports exactly two.

- **`PresetGenVAEMLPRegressor` = the paper's "MLP" rows** — the full VAE (encoder + latent + latent
  flow + decoder + reconstruction + latent loss) with an MLP head (`3l1024`) off `zK`.
- **`PresetGenVAEFlowRegressor` = the paper's "Flow" rows** — the same VAE, but the head is a RealNVP
  flow (`realnvp_6l300`) used feed-forward. The nflows pieces the paper uses are ported as plain
  torch in `models/presetgen_vae/realnvp.py` (both flows run at predict time, so they cannot hide
  behind the training-only lazy imports).

(Numbering starts at 2 because an earlier Stage 1 — a no-VAE encoder→regressor baseline, to isolate
what the VAE machinery buys — was built and then removed as out of scope. The discriminative slot in
the benchmark is covered by `Sound2SynthSpectrogramRegressor`.)

## The two regression architectures: MLP vs flow

The paper's Table 1 compares two configurations of the regressor head. Everything else is
shared: the same mel-dB front-end, the same `speccnn8l1_bn` encoder and decoder, the same
**latent flow**, the same joint loss (reconstruction + beta·latent + controls), the same
optimizer recipe. Only the map `zK` → parameters changes. Our two registry entries
(`PresetGenVAEMLPRegressor` / `PresetGenVAEFlowRegressor`) mirror exactly this knob.

**MLP regression** (`mlp_3l1024`, `model/regression.py` `MLPRegression`): three
fully-connected layers of 1024 units, ReLU each, batch-norm + dropout on all but the two
deepest layers, then a plain linear to the parameter vector — the paper's "4-layers MLP with
1024 hidden units… BN and dropout appended to the two first MLP layers" (§3.4.2), counting the
output layer. A generic function approximator: any latent size works.

**Flow regression** (`flow_realnvp_6l300`, `model/regression.py` `FlowRegression`): a
RealNVP normalizing flow — six affine coupling layers, each conditioned by a 2-block
residual network with 300 hidden features. Used purely feed-forward here (the log-det
output is discarded). The trade the paper makes: a flow is *invertible*, so besides
predicting parameters from a latent it can also map a *known preset back into the latent
space*. The paper uses that inverse for its interactive preset-morphing interface
(Section 5.2); our benchmark never calls it. Invertibility costs a constraint: input and
output must have the same width, so `dim_z` must equal the learnable preset vector length
(the assert in `model/build.py:70`).

**Both families use `latent_dimension = ml_dimension`.** The flow head *requires* it. The MLP head
does not, but Table 1 gives the MLP model the same D as the flow model anyway (340 and 610, the
`NumCat++` vector lengths), and §3.4.2 says why: otherwise the MLP-vs-flow comparison is confounded
by latent size. So the base family pins it for both, and neither exposes a `latent_dimension` knob.

Shared vs different, at a glance:

| | MLP regression | Flow regression |
|---|---|---|
| VAE (front-end, encoder, decoder, latent, latent flow, loss) | identical | identical |
| Head | 3 FC layers × 1024, ReLU | 6 RealNVP couplings, 300 hidden features |
| Latent size | = parameter vector length (Table 1) | = parameter vector length (forced) |
| Invertible (preset → latent) | no | yes (unused in our benchmark) |
| Our family | `PresetGenVAEMLPRegressor` | `PresetGenVAEFlowRegressor` |

## Walkthrough: our code ↔ the paper's code

Piece-by-piece counterpart map. Ours under `models/presetgen_vae/`, paper under
`paper_repos/preset-gen-vae/`. Line numbers are as of this writing; names are the stable
reference.

The map is backed by **parity tests** (`tests/test_paper_parity.py`): each network
component is built from both codebases, the paper's randomly-initialized weights are
transplanted into ours, and the outputs on identical inputs are asserted numerically
equal — proving the two implementations compute the same function. The flows are checked on
their log-determinants too, in both train and eval mode. Encoder, decoder, beta warmup and the
latent log-densities always run; the flow / MLP-head / latent-loss / KL tests need the paper's
`nflows` dependency and skip unless it is installed (`pip install nflows --no-deps`, dev-only).

| Piece | Ours | Paper |
|---|---|---|
| audio → mel-dB spectrogram | `network.py` `_compute_mel_db_spectrogram` + `_build_mel_filterbank` | `utils/audio.py` `MelSpectrogram` (73–87), applied per item in `data/abstractbasedataset.py` (121–133) |
| dB min/max normalization endpoints | `network.py` `measure_corpus_mel_db_range` (D-MELNORM) | dataset-wide `spec_stats` min/max, `data/abstractbasedataset.py` (129–131, 311) |
| encoder CNN (enc1–enc8) | `network.py` `_build_spectrogram_cnn` | `model/encoder.py` `SpectrogramCNN`, `'speccnn8l1_bn'` branch (233–259) + `SpectrogramEncoder.features_mixer_cnn` (54–70) |
| conv / tconv building block | `network.py` `_conv2d_block` / `_tconv2d_block` | `model/layer.py` `Conv2D` / `TConv2D` |
| encoder MLP → mu, logvar | `network.py` `encoder_mlp` + `_encode` | `model/encoder.py` (85, 102–108) |
| latent-flow input batch-norm | `network.py` `encoder_mlp`'s `lat_in_regularization` | `model/encoder.py` (86–87), switched on by `latent_flow_input_regularization = 'bn'` |
| reparameterization (sample z0; eval uses mu) | `network.py` `_reparameterize` | `model/VAE.py` `FlowVAE.forward` (170–176) |
| **latent flow z0 → zK** | `network.py` `latent_flow` + `_apply_latent_flow` (a `realnvp.py` `RealNVP`) | `model/VAE.py` `FlowVAE` (69–181), built from nflows' `SimpleRealNVP` (118–125) |
| decoder MLP + 1×1 un-mixer + CNN (dec1–dec8) | `network.py` `decoder_mlp` + `_build_decoder_cnn` | `model/decoder.py` `SpectrogramDecoder` (57–92) + decoder `SpectrogramCNN` (199–220) |
| MLP regressor head | `network.py` `_build_regressor` | `model/regression.py` `MLPRegression` (61–102) |
| flow regressor head | `realnvp.py` `RealNVP` | `model/regression.py` `FlowRegression` (105–189) + `model/flows.py` `CustomRealNVP` (42–90) |
| flow internals (coupling, conditioner, flow BN) | `realnvp.py` `AffineCouplingLayer`, `ResidualNetwork`, `FlowBatchNorm` | the `nflows` package (`AffineCouplingTransform`, `nets.ResidualNet`, `transforms.normalization.BatchNorm`) — a dependency there, ported as plain torch here |
| `dim_z` == parameter vector length | `families.py` `BasePresetGenVAERegressor._build_architecture_hparams` | `model/build.py` assert (70) for the flow head; Table 1 / §3.4.2 for the MLP one |
| assembled network | `network.py` `PresetGenVAENetwork` | `model/VAE.py` `FlowVAE` + `model/extendedAE.py` `ExtendedAE`, wired in `model/build.py` `build_extended_ae_model` (55–80) |
| benchmark wrapper (fit/save/load/predict) | `families.py` (via `models/base_deep_model.py`) | none — `train.py` / `eval.py` drive the raw networks directly |
| training step + loss assembly | `lightning_module.py` `LightningVAERegressor._shared_step` | `train.py` epoch loop (222–248) |
| reconstruction loss (MSE) | `F.mse_loss` in `_shared_step` | `model/loss.py` `L2Loss` / `nn.MSELoss` via `train.py:106` |
| **Monte-Carlo latent loss** | `models/training/loss.py` `flow_latent_loss`, picked by `lightning_module.py` `_latent_loss` | `model/VAE.py` `FlowVAE.latent_loss` (183–193) |
| latent log-densities | `models/training/loss.py` `gaussian_log_probability`, `standard_gaussian_log_probability` | `utils/probability.py` (13–29) |
| closed-form KL term (only without a latent flow) | `models/training/loss.py` `gaussian_kl_divergence` | `model/loss.py` `GaussianDkl` (46–66) |
| controls (parameter) loss | `models/training/loss.py` `ParameterLoss` | `model/loss.py` `SynthParamsLoss` (73–186) |
| beta warmup (scales the latent term) | `lightning_module.py` `linear_warmup` | `LinearDynamicParam` schedule, `train.py:150, 227` |
| hyperparameters | `cluster/training_configs/presetgen_full_config.yaml` | `config.py` (16–118) |

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
  per-sample log-det is exposed (`forward_with_log_determinant`) because the latent flow
  needs it.
- **One `RealNVP` class, two flows.** The paper builds its two flows from *different* nflows
  classes: `CustomRealNVP` for the regressor head, plain `SimpleRealNVP` for the latent flow
  (`model/VAE.py:118`). They differ only in that `CustomRealNVP` suppresses dropout and
  between-layer batch-norm on the last two couplings. The paper gives its latent flow neither
  (`dropout_probability` defaults to 0; `batch_norm_between_layers=False`, since "True would
  prevent reversibility during train"), which makes that difference inert — the two build an
  identical stack. So one ported `RealNVP` serves both roles, configured differently.
  `tests/test_paper_parity.py` proves this against *both* nflows classes rather than assuming it.
- **Latent loss parity.** With a latent flow, `q(zK|x)` has no closed form, so `flow_latent_loss`
  ports `FlowVAE.latent_loss`: `-(log p(zK) - log q(z0) + log|det J|)`, averaged over the batch
  and optionally divided by `dim_z`. `gaussian_kl_divergence` (a line-for-line port of
  `GaussianDkl`) is kept for the no-latent-flow path only. Either term is multiplied by the same
  warmed-up beta (`train.py:227`); `normalize_latent_loss: true` + `beta: 0.2` matches the
  paper's `normalize_losses = True`, `beta = 0.2`, warmup 0.1 → 0.2 over 25 epochs.
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
- **Latent-flow dropout: the paper contradicts itself.** Its §3.3.1 says "Internal scale and
  translation coefficients of each RealNVP layer are computed using a 2-layer MLP (300 hidden
  units) with residual connection, Batch Normalization (BN) and 0.4 dropout probability" — for
  *both* flows. Its code passes `dropout_p=0.4` to the regression flow but **no** dropout to the
  latent flow (`model/VAE.py:118–123`, so nflows' default of 0.0). We follow the code, because
  that is what the parity tests can check and what the published runs actually did. Consequence:
  our latent flow has BN but no dropout; the regressor flow has both.
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
