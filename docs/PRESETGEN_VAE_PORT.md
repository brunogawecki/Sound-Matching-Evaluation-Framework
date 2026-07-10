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

## Caveats

- Reproduced numbers will **not** match the paper's tables, by design: **103** Dexed params (D1), not
  144; our own categorical scheme (close to `NumCat`, not `NumCat++`); a single train/test split, not
  5-fold; **dawdreamer** rendering, not RenderMan. We reproduce the *method*, not the *numbers*.
