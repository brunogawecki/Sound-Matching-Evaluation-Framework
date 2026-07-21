# The flow-matching port — networks, models, and modules

How the paper *"Audio Synthesizer Inversion in Symmetric Parameter Spaces with Approximately
Equivariant Flow Matching"* (Hayes, Saitis, Fazekas, ISMIR 2025,
`paper_repos/synth-permutations/`) maps onto the code in the `models/flow_matching/` package — one
paper, one package, one file per role: `flow_matching.py` (the framework-agnostic flow math),
`encoder.py` (the conditioning encoder), `vector_field.py` (the two learned fields), `network.py`
(the assembled network), `families.py` (the registered model wrappers), and `lightning_module.py`
(the training-only loss recipe). Design rationale (D1, D-METRIC-SR, D-REPR, D-SELFDESC, D-MELNORM,
D-FRAMEWORK, D-EVAL, D-REPRO) lives in `docs/DECISIONS.md`; this doc is just the map.

This is the benchmark's **conditional-generative** family — and the only one built around
*symmetry*. It is not a second VAE: preset-gen-vae is a VAE whose flow runs feed-forward as a
regressor head, whereas this family samples parameters by integrating a learned ODE.

## The thesis in one paragraph

Synth inversion is ill-posed because synthesizers have **symmetries**: permuting functionally
equivalent operators gives a different parameter vector but identical audio. A regressor trained on
such data averages over the equivalent solutions and lands between them — the paper's
"responsibility problem". A **conditional generative** model of `p(params | audio)` sidesteps this
by representing the whole solution set instead of one point. Building the synth's permutation
symmetry *into* the model does better still, which is what the Param2Tok field is for.

## Three words that mean different things

| Word | What it is | Examples |
|------|-----------|----------|
| **network** | The raw neural net — layers + a `forward`. Pure PyTorch. | `FlowMatchingNetwork`, `AudioSpectrogramTransformer` |
| **model / family** | The benchmark wrapper: `fit` / `save` / `load` / `predict`. What the pipeline scripts drive. | `FlowMatchingMLP`, `FlowMatchingParam2Tok` |
| **(Lightning) module** | A *training-time-only* wrapper defining the loss + training step. Discarded after `fit`. | `LightningFlowMatching` |

One **model** contains one **network** and, only while training, hands it to one **module**.

## Two models, one difference

The paper reports several models; we port the **two** it carries to its real-synth (Surge) task.
They share the corpus, the encoder, the loss, and the sampler, and differ **only** in the vector
field — which is exactly what makes the pair a controlled experiment:

| Family | Vector field | Permutation-equivariant? | Paper config |
|--------|-------------|--------------------------|--------------|
| `FlowMatchingMLP` | `ConditionalResidualMLPField` — a 9-block conditional residual MLP, `d_model` 768 | no | `configs/model/surge_flowmlp.yaml` |
| `FlowMatchingParam2Tok` | `EquivariantTransformerField` — Param2Tok + an 8-block DiT, `d_model` 512 | approximately | `configs/model/surge_flow.yaml` |

The MLP variant is **not** a baseline in the "weak floor" sense — it is the paper's own control, and
the MLP↔Param2Tok gap is the result the port exists to measure. Reporting Param2Tok without it says
nothing about symmetry.

### Deliberately not ported: the AST regression baseline

The "AST" row in the paper's Table 1 is a *separate discriminative* model (Bruford et al. DAFx24),
not part of the flow-matching method. The framework already covers discriminative approaches
(`sound2synth`, `inversynth2`), and the shared-corpus / shared-metric design already delivers the
generative-vs-discriminative comparison. Note the name collision: our `AudioSpectrogramTransformer`
is the flow's **conditioning encoder**, not that baseline.

## How flow matching works here

Rectified flow. A sample is a straight-line path from Gaussian noise `x0` to the parameter vector
`x1`, and the network learns the constant velocity along it:

```
x_t = (1 - t) * x0 + t * x1          target velocity = x1 - x0
```

Training regresses the field onto that velocity (plain MSE, uniform time weighting) plus the
field's auxiliary `penalty()`. Sampling integrates the learned ODE from fresh noise with 4th-order
Runge-Kutta, guided by classifier-free guidance (CFG): each step evaluates the field twice, once
conditioned on the audio and once on a learned dropout token, and blends them.

Two details worth knowing:

- **Minibatch optimal transport.** Before building the path, noise is paired to targets by a
  Hungarian assignment (`scipy.optimize.linear_sum_assignment` over `torch.cdist`). Pairing each
  target with its nearest noise sample straightens the paths and lowers the variance of the
  training signal.
- **CFG dropout.** At the configured rate a sample's conditioning is replaced by a learned token,
  which is what makes the unconditional branch — and therefore guided sampling — available at
  inference.

Everything above lives in `flow_matching.py` and is framework-agnostic, so the offline `predict`
path needs no Lightning (D-FRAMEWORK).

## Param2Tok — where the symmetry argument lives

`Param2TokProjection` maps the ML-side vector to a **set of tokens** and back:

1. each scalar parameter is embedded along its own learned direction (`in_projection`), giving
   `[batch, num_params, d_model]`;
2. an FFN lifts it;
3. a learned **assignment matrix** `A [num_tokens, num_params]` contracts the parameters into
   `num_tokens` (128) token slots;
4. the DiT stack processes the tokens **with no positional encoding**, so it is permutation-
   equivariant over them;
5. `A^T` and `out_projection` map back to a velocity.

The symmetry is *coaxed, not enforced*: an **L1 penalty on `A`** pushes the assignment to be sparse,
so each token slot ends up carrying a small group of parameters (ideally one operator's worth). That
is the "approximately" in approximately-equivariant, and it is why the paper's own name for the
field is `ApproxEquivTransformer`.

`tests/test_flow_matching.py` asserts the equivariance on `DiffusionTransformerBlock` directly
(permuting the tokens permutes the output identically) rather than inferring it from the field's
output, since that property is the load-bearing one.

## Sampling is what `predict` does

The important consequence of being a *sampler*, not a regressor: `BaseDeepModel.predict`'s single
forward pass is wrong for this family, so `BaseFlowMatchingModel` **overrides `predict`**. It draws
one sample by integrating the ODE (the paper's test-time protocol: **200 RK4 steps, CFG strength
2.0**), maps the result from flow space `[-1, 1]` back to the ML-side `[0, 1]`, and decodes it
through `ParameterSpace.ml_vector_to_synth_dict`.

The draw is **seeded per call** (`_predict_seed`, default 0), so a model's prediction for a given
clip is deterministic and the Evaluator's re-render is reproducible. Best-of-N sampling is a
deferred extension — cheap to add later, since the Evaluator already re-renders every prediction.

Cost note: at 200 steps × 2 field evaluations per step, `FlowMatchingParam2Tok` measured ~3.7×
slower per sample than `FlowMatchingMLP` on CPU (13.9 s vs 3.8 s in a smoke run). Budget eval time
accordingly.

## Training corpus: synthetic-uniform, deliberately unlike the other families

Every other deep family trains on the human preset corpus. This one should train on a
**synthetic-uniform** corpus (`ParameterSpace.sample_uniform` via `scripts/build_dataset.py
synthetic`), because the equivariance benefit assumes a **G-invariant parameter prior** — one that
respects the synth's permutation symmetry. Uniform sampling gives that for free; curated human
presets break it (the paper attributes a VAE+flow collapse to exactly that bias). Train Param2Tok on
human presets and you have discarded the reason it should win.

The *test* corpus is the shared benchmark one, same as every family — that is what keeps the results
table comparable. See `docs/DECISIONS.md` for the decision record.

**The corpus we can actually build is not exactly G-invariant.** `scripts/build_dataset.py
synthetic` always applies `synth.audible_sampling_ranges` (D-AUDIBLE), which for Dexed pins three
**OP1** parameters. Naming one operator makes the prior non-invariant under operator permutation —
a partial break of the very property this section relies on. It is 3 parameters of 103 and one
operator of six, so the prior stays *approximately* invariant, which is why it is accepted for now;
`FlowMatchingMLP` is the control that keeps the conclusion honest. Recorded in full, with the
rejected alternatives, under **D-FLOW-CORPUS** in `docs/DECISIONS.md`.

## How the code maps onto this

| Piece | File | Notes |
|---|---|---|
| rectified path, target velocity, OT pairing, CFG, RK4 | `flow_matching.py` | framework-agnostic; no Lightning import |
| conditioning encoder (mel patches → per-layer conditioning) | `encoder.py` | `AudioSpectrogramTransformer` + `PatchEmbed` + `PositionalEncoding` |
| non-equivariant field | `vector_field.py` | `ConditionalResidualMLPField` |
| Param2Tok + DiT field | `vector_field.py` | `Param2TokProjection`, `DiffusionTransformerBlock`, `EquivariantTransformerField` |
| shared CFG-dropout token + `penalty()` hook | `vector_field.py` | `ConditionedField` base |
| mel front-end + assembly + `sample` | `network.py` | `FlowMatchingNetwork`; `vector_field_architecture` switches `"mlp"` / `"param2tok"` |
| corpus mel-dB statistics | `network.py` | `measure_corpus_mel_db_statistics` (D-MELNORM) |
| benchmark wrappers | `families.py` | `BaseFlowMatchingModel` + the two families |
| training recipe | `lightning_module.py` | `LightningFlowMatching`, lazily imported |

Both families are registered in `models/registry.py`, so each is independently trainable
(`scripts/fit_model.py --model`) and evaluable (`scripts/evaluate.py --model`), and mirrored as
plain strings in `dashboard/script_specs.py::MODEL_CHOICES`.

## Walkthrough: our code ↔ the paper's code

Ours under `models/flow_matching/`, paper under `paper_repos/synth-permutations/`. Line numbers are
as of this writing; names are the stable reference.

**No weight-transplant parity tests.** The reference ships no trained checkpoints, and its task is
Surge, not Dexed. This port reproduces the *architecture and recipe*, checked by shape / round-trip
/ behavioral tests (`tests/test_flow_matching.py`), including RK4 against a closed-form ODE and the
permutation-equivariance assertion above.

| Piece | Ours | Paper |
|---|---|---|
| audio → mel-dB spectrogram | `network.py` `_compute_mel_db_spectrogram` (torch, in-graph) | `src/data/audio_datamodule.py::make_spectrogram` (11, librosa, offline) |
| mel-dB standardization | `network.py` `measure_corpus_mel_db_statistics` (scalar corpus mean/std) | per-bin `stats.npz`, `src/data/surge_datamodule.py` (52, 62, 125) |
| target rescale to `[-1, 1]` | `lightning_module.py` `targets * 2.0 - 1.0` | `surge_datamodule.py` (138) |
| patch embedding | `encoder.py` `PatchEmbed` | `components/transformer.py::PatchEmbed` (512) |
| conditioning encoder | `encoder.py` `AudioSpectrogramTransformer` | `components/transformer.py::AudioSpectrogramTransformer` (557) |
| sinusoidal time encoding | `vector_field.py` `SinusoidalEncoding` | `components/transformer.py::SinusoidalEncoding` (245) |
| residual-MLP block / field | `vector_field.py` `ConditionalResidualMLPBlock` / `ConditionalResidualMLPField` | `components/residual_mlp.py::ConditionalResidualMLPBlock` (62) / `ConditionalResidualMLP` (89) |
| Param2Tok projection | `vector_field.py` `Param2TokProjection` | `components/transformer.py::LearntProjection` (56) |
| Ada-LN DiT block | `vector_field.py` `DiffusionTransformerBlock` | `components/transformer.py::DiTransformerBlock` (155) + `AdaptiveLayerNorm` (142) |
| equivariant field | `vector_field.py` `EquivariantTransformerField` | `components/transformer.py::ApproxEquivTransformer` (351) |
| rectified path | `flow_matching.py` `rectified_path_sample` | `surge_flow_matching_module.py::_rectified_probability_path` (88) |
| target velocity | `flow_matching.py` `rectified_target_velocity` | `surge_flow_matching_module.py::_rectified_vector_field` (101) |
| OT pairing | `flow_matching.py` `optimal_transport_pairing` | `src/data/ot.py::_hungarian_match` (9) |
| CFG-guided velocity | `flow_matching.py` `_guided_velocity` | `surge_flow_matching_module.py::call_with_cfg` (10) |
| RK4 sampler | `flow_matching.py` `rk4_sample` | `surge_flow_matching_module.py::rk4_with_cfg` (23) |
| training step | `lightning_module.py` `training_step` | `surge_flow_matching_module.py::_train_step` (120) / `training_step` (154) |
| validation (samples, logs param MSE) | `lightning_module.py` `validation_step` | `surge_flow_matching_module.py::validation_step` (203) |
| single-sample predict | `families.py` `BaseFlowMatchingModel.predict` | `surge_flow_matching_module.py::predict_step` (242) |
| benchmark wrapper (fit/save/load) | `families.py` (via `models/base_deep_model.py`) | none — the repo drives the raw nets through Hydra |

## Deviations from the paper

Intentional, dictated by our framework contract:

- **Mono 22.05 kHz, not stereo 44.1 kHz.** Our render contract is mono at 22.05 kHz for 4.0 s
  (D3, D-METRIC-SR LOCKED), so the AST takes 1 input channel, and the mel analysis is specified in
  **milliseconds** (128 mels, 25 ms window, 10 ms hop) and recomputed in samples. This lands on
  `n_fft=551`, `hop_length=220`, and features of shape `(batch, 1, 128, 401)` — the reference
  hardcodes `spec_shape: [128, 401]`, so `patch_size 16` / `patch_stride 10` carry over unchanged.
  Same analysis, different sample rate.
- **Mel computed in-graph, not offline into HDF5.** Featurization lives inside the network's
  `forward` (D-REPR), using `torch.stft` to reproduce librosa's `power_to_db(ref=np.max)`. Matched
  spec, not bit-identical.
- **Scalar corpus mean/std, not the reference's per-bin `stats.npz`.** Following D-MELNORM, the
  statistics are measured over the train corpus at `fit` time and folded into the checkpoint, so
  `load` rebuilds the identical front-end offline with no side files. This is the deviation most
  worth revisiting if results disappoint — per-bin normalization is strictly more expressive.
- **OT pairing permutes only the noise.** The reference reorders both sides; permuting one side
  yields the same coupling and avoids reordering the audio batch.
- **Noise drawn in `training_step`, not in a collate function.** Same distribution, no custom
  collate needed.
- **`num_conditioning_outputs` derived from the field's layer count.** One conditioning token per
  field layer reproduces both reference configs exactly (9 for the MLP field, 8 for Param2Tok), but
  the reference leaves them as two independent knobs. Ours ties them.
- **`out_projection` is initialized as the transpose of `in_projection`, not weight-tied.** The two
  drift apart during training. This matches the reference, which clones rather than ties — worth
  stating because the paper's text reads as if it were tied.
- **`penalty()` returns the already-weighted L1** (× `projection_penalty`, 0.01), so the training
  step adds it unscaled. Same arithmetic as the reference, different split of the multiplication.
- **Unused reference knobs dropped.** The Surge configs use none of `pe_type` other than `none`,
  `adaln_mode` `zero`/`res`, `zero_init=True`, `outer_residual`, scalar time encoding, or
  `learn_pe`, so the port implements only the paths the paper actually runs. (`learn_projection:
  false` would in fact crash the reference — it reaches for `projection.proj`, which
  `LearntProjection` does not have.) Also not ported: `torch.compile`, gradient-norm logging, the
  `_weight_time` hook (returns ones — inlined as uniform weighting), and the `oversample` branch of
  `_basic_sample`.
- **Framework optimizer defaults.** AdamW 3e-4 rather than the paper's Adam 1e-4 + cosine. The
  paper's schedule is fully reachable through `TrainingConfig`:
  `{"optimizer": {"name": "adam", "learning_rate": 1e-4, "scheduler": "cosine"}}`. (The reference
  sets `warmup_steps: 0` for both Surge models, so its warmup scheduler is dead code there.)

## Caveats

- Reproduced numbers will **not** match the paper's tables, by design: the paper's task is **Surge
  XT**, ours is **Dexed** with **103** parameters (D1) and our own categorical scheme over
  `ParameterSpace.loss_slices`; **dawdreamer** rendering; a single train/test split; and a
  22.05 kHz mono contract. We reproduce the *method*, not the *numbers*.
- The equivariance claim depends on the **training corpus** being symmetry-respecting (above). A
  Param2Tok run on human presets losing to its own MLP control is not a broken result — it is
  evidence about the premise, and should be reported as such. But note the corpus we can build today
  carries the OP1 audibility confound (above): if Param2Tok fails to separate from the MLP control
  on the *synthetic* corpus, rule that out before concluding anything about the paper's premise.
- `predict` returns **one** sample. Generative families have a variance the regression families do
  not, and a single seeded draw does not measure it. Best-of-N or per-target sample statistics would,
  and are deferred.
