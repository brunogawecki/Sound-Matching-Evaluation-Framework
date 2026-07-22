# The SynthRL port — networks, models, and modules

How the paper *"SynthRL: Cross-domain Synthesizer Sound Matching via Reinforcement Learning"*
(Shin & Lee, IJCAI-25, `github.com/argaaw/SynthRL`) maps onto the code in the `models/synthrl/`
package — one paper, one package, one file per role: `representation.py` (the class-index view),
`network.py` (the transformer), `families.py` (the registered model wrappers),
`lightning_module.py` (the training-only loss recipes), and `reward.py` / `reward_buffer.py` (the
RL reward and its replay buffer). Design rationale (D1, D2, D-MELNORM, D-FRAMEWORK, D-SELFDESC,
D-EVAL, D-REPRO, D-METRIC-NORM, D-RL-RENDER) lives in `docs/DECISIONS.md`; this doc is just the map.

SynthRL fills the benchmark's **reinforcement-learning** family slot. It is a peer-paper approach
alongside the discriminative (Sound2Synth), generative (preset-gen-vae), neural-proxy
(InverSynth II) and flow-matching families, **not** a baseline.

The reference repo is not vendored under `paper_repos/` — it is AGPL-3.0 and we only need it as a
reading reference. Clone it separately when checking a detail against source.

## Three words that mean different things

Same split as the other ports:

| Word | What it is | Examples |
|------|-----------|----------|
| **network** | The raw neural net — layers + a `forward`. Pure PyTorch. | `SynthRLNetwork` |
| **model / family** | The benchmark wrapper: `fit` / `save` / `load` / `predict`. | `SynthRLp`, `SynthRLi` |
| **(Lightning) module** | A *training-time-only* wrapper defining the loss + training step. | `SynthRLParameterRegressor`, `SynthRLReinforceRegressor` |

## The paper stacks THREE stages — we port two

SynthRL trains one network in three stages, each named in the paper and each independently
reported (paper §4.2, Table 1):

| Stage | Paper name | What it is | Ported? |
|-------|-----------|-----------|---------|
| 1 | `SynthRL-p` | Parameter loss only, in-domain. 200 epochs. | ✅ `SynthRLp` |
| 2 | `SynthRL-i` | Adds the RL loss, ramping the parameter loss out. 200 epochs. | ✅ `SynthRLi` |
| 3 | `SynthRL-o` | RL-only fine-tune on **out-of-domain** sounds (a second synth). | ❌ deferred |

Stage 3 needs a second synthesizer (the paper uses Surge XT). The second-synth decision is open
(**D-FAMILIES**), so `SynthRL-o` is deferred rather than guessed at. Nothing in the port blocks
it: it is stage 2's recipe with the parameter loss switched off and a different corpus.

`SynthRLi` warm-starts from a `SynthRLp` checkpoint through `--init-from`, the generic
`BaseDeepModel._warm_start_network` hook.

## Everything is a classification problem

The single idea that shapes the whole port (paper §3.2–3.3). SynthRL does **not** regress
parameter values. Every parameter — numerical or categorical — becomes a classification head:

- a **categorical** parameter keeps its own option cardinality;
- a **numerical** parameter is discretized onto 25 equally spaced classes.

`SynthRLRepresentation` owns that view. It **wraps** the shared `ParameterSpace` and never
modifies it, so D2/D-KIND (continuous = float, categorical = one-hot) is untouched and the
class-index view stays private to this family.

The grid matters. Class `c` of a numerical parameter decodes to `low + c·(high−low)/(n−1)` — the
repo's `round(v·(n−1))` / `c/(n−1)` scheme — so **both bounds are reachable**. Equal-width bins
decoding to bin *centers* would put the extremes at 0.02 and 0.98, and a Dexed operator predicted
"off" would still render at output level 2.

Because the heads are ordinal for numerical parameters, the cross-entropy target is
**Gaussian-smoothed** over neighbouring classes (paper §3.3, after Chen et al. 2022): a prediction
one class off is penalised less than one twenty classes off. Categorical heads are unordered, so
they keep a hard one-hot. Smoothing width is σ = 0.5 class indices, matching the repo's
`GaussianKernelConv(sigma=0.02)` scaled by its 25 classes.

## The network

`SynthRLNetwork`, a transformer encoder-decoder (paper §3.2, Figure 2):

```
audio → mel-dB → strided conv reducer → +2D sinusoidal pos. enc. → transformer encoder → z
                                                                                          ↓
   learnable query per parameter → self-attn → cross-attn(z) → per-parameter class heads → scores
```

The decoder is DETR-style: `n` learnable queries, one **per synthesis parameter**, attend to each
other (capturing dependencies between parameters) and then cross-attend onto the encoder feature
map. Each query's output goes through its own projection head to that parameter's class scores.

Head outputs are squashed to `[0, 1]` (`tanh` then `0.5·(x+1)`, as the repo does). This is not
cosmetic: it caps the within-head score gap at 1.0, so after the softmax temperature the policy's
sharpness is bounded and it keeps exploring. The temperatures live with the consumers, not the
network — `LOSS_SOFTMAX_TEMPERATURE = 0.2` for the cross-entropy, `POLICY_SOFTMAX_TEMPERATURE =
0.1` for the RL policy, both in `lightning_module.py`. Squashing is monotonic, so argmax decoding
at predict time is unaffected.

## The RL stage

`SynthRLReinforceRegressor`. The paper frames sound matching as a **single-step** MDP (a
contextual bandit): state = the target sound, action = the estimated parameters, reward = audio
similarity between the target and the render of that action. REINFORCE, no value function.

### Reward (paper §3.4, Eqs. 2–5)

`R = 1 / clamp(w₁·Spec + w₂·SC + w₃·MFCC, 0.1, 5.0)` with `w = (0.27, 0.70, 0.03)`, bounding the
reward to `[0.2, 10.0]`. The three distances reuse the framework's own metric callables (`lsd`,
`spectral_convergence`, `mfcc_mae`), so the RL reward and the evaluation panel measure similarity
the same way — one source of truth.

### Reward-based PER (paper Algorithm 1)

Plain REINFORCE fails here: the action space is ~100 near-independent discrete parameters, so
high-reward actions are sampled too rarely to learn from. SynthRL keeps a small per-target buffer
of the best `m = 5` actions found so far and trains on those instead of only the freshest sample.
A new experience is kept if the buffer has room or beats its current minimum.

Two details that are easy to get wrong:

- **Pre-fill.** The repo runs `m` gradient-free render passes over the training set (the first
  greedy, the rest sampled) *before* the first update, so every target starts with a full buffer.
  Without it the buffer holds one entry per target for the first epochs and the objective
  degenerates to plain on-policy REINFORCE — exactly while the parameter loss is ramping out.
  `on_train_start` does this; `rl.prefill_epochs` controls it.
- **Objective scale.** The loss is `−mean(m · R · mean_log_π(action))`. The log-probability is
  **averaged** over parameters, not summed, and the `m` factor is the importance weight under the
  uniform proposal `μ = 1/m` (Eq. 6). Summing over ~100 heads instead would inflate the RL term by
  two orders of magnitude and make the curriculum ramp meaningless. The policy-ratio half of Eq.
  6's importance weight is dropped, as it is in the released code.

### The curriculum ramp

Stage 2 blends `loss = α·RL + (1−α)·parameter`, with α rising linearly `0 → 1` over
`rl.ramp_epochs` (100) and staying at 1 afterwards. This is the repo's `rl_coef` schedule over
epochs 199→299. The parameter loss carries the repo's 0.2 categorical factor so the two terms
balance the way the paper's do.

### Silent operators are dropped from the parameter loss

A Dexed operator at output level 0 is inaudible, so its other parameters carry no learnable signal
and would train the model on noise. The repo excludes them per sample
(`data/preset.py: get_useless_learned_params_indexes`). `gated_parameter_groups` derives the same
grouping from the **parameter names** (`OP<n> OUTPUT LEVEL` gates `OP<n> …`), so it honours
D-NAMING and is a silent no-op on any synth that does not use that convention.

## Training renders with the real Dexed

The RL stage renders every sampled patch inside the training loop through
`ParallelFreshProcessRenderBackend` — the fresh-process-per-render isolation of D-REPRO, widened
to a worker pool. So a **`SynthRLi` training run is not VST-free**, a deliberate deviation from
D-SELFDESC. It is training-only: `predict` decodes class scores through the representation with no
synth involved, so the eval path is identical for both families and unchanged from every other
family in the benchmark. **D-RL-RENDER** records the decision and the alternatives.

Practical consequences:

- Stage 2 is far slower per epoch than stage 1 (one fresh VST process per patch per step, plus
  `prefill_epochs` full passes up front). Budget wall-clock accordingly, and set
  `rl.num_render_workers` to the job's `--cpus-per-task`.
- The training environment needs **Dexed 0.9.8** — the version the D-SELFDESC cluster spike pinned.
  Parameter-name parity between that build and the one that rendered the corpus is still unverified,
  and it matters here: the backend sets patches by name, so a renamed parameter would silently
  change what the reward scores.

**Future optimization, not built:** reusing one in-process synth with a state reset between patches
would be roughly two orders of magnitude faster per render. It is rejected for now because it
reintroduces the context leakage D-REPRO excludes, which would leave the reward and the reported
metric measuring different audio. Revisit only if stage 2 proves render-bound, and only with the
leakage measured.

## How the code maps onto this

| Paper / repo | Here |
|---|---|
| §3.3 25-class discretization, `data/preset.py` one-hot | `representation.py: SynthRLRepresentation` |
| §3.3 Gaussian label smoothing, `utils/probability.py` | `representation.py: smoothing_matrices` |
| §3.2 CNN + transformer enc/dec, `model/network.py` | `network.py: SynthRLNetwork` |
| §3.3 parameter loss, `model/loss.py: ParameterLoss` | `lightning_module.py: SmoothedClassCrossEntropy` |
| §3.4 Eq. 5 reward, `model/loss.py: calculate_rewards` | `reward.py: sound_matching_reward` |
| Algorithm 1 buffer, `utils/buffer.py: Replaybuffer` | `reward_buffer.py: RewardPrioritizedReplayBuffer` |
| `train.py` (stage 1) | `lightning_module.py: SynthRLParameterRegressor` |
| `finetune.py` (stages 2–3) | `lightning_module.py: SynthRLReinforceRegressor` |
| `config/stage1.yaml` | `cluster/training_configs/synthrl_p_config.yaml` |
| `config/stage2.yaml` | `cluster/training_configs/synthrl_i_config.yaml` |
| `config/stage3.yaml` | — (deferred, see D-FAMILIES) |

## Deviations from the paper

Deliberate, and each with a reason:

| # | Deviation | Why |
|---|---|---|
| 1 | **103-param Dexed subset**, not the paper's 144 | D1 LOCKED. The dropped parameters are non-identifiable under the D3 single-note render contract. |
| 2 | **Which parameters are discretized** follows the framework's own continuous/categorical split (16 categorical, 87 continuous), not the repo's `all<=34` rule | D2/D-KIND LOCKED. The outcome is the same in practice: every Dexed parameter has cardinality ≤ 34, so the repo's rule also makes them all categorical. |
| 3 | **Reward compares raw audio**; the repo peak-normalizes both the target and the render first | D-METRIC-NORM. Consequence: level error is part of our reward, so `SynthRL-i` optimizes a slightly different objective than the paper's level-invariant one. Worth stating in any results discussion. |
| 4 | **Mel front-end** is the preset-gen-vae one (257 mels, corpus-measured dB range) rather than the repo's 128 mels + `clip(log(x+1e-5)/12, −1, 1)` | D-MELNORM + cross-family consistency: every deep family here shares one front-end, so architecture comparisons are not confounded by input preprocessing. |
| 5 | **No LR warmup, no `CosineAnnealingWarmRestarts`** (the repo uses a 10-epoch linear warmup from 0.2× plus warm restarts every 50 epochs) | `models/training/` supports plain cosine only. Same limitation accepted for Sound2Synth and InverSynth II. |
| 6 | **Network is smaller**: `d_model` 256 and 4+4 transformer layers vs the repo's 512 and 6+6; 4 conv reducer layers vs 5 | Family defaults, chosen for cluster budget. Settable per-run through the family constructor if a fuller replication is wanted. |
| 7 | **Positional encoding added once** before the encoder, no separate decoder query positional encoding, no 0.3 projection dropout | DETR-style detail. The queries are learnable per position, so a separate query positional encoding adds little. |
| 8 | **Single seeded 10% validation split**, not the repo's 5-fold CV with a 20% test holdout | Framework convention across all families. |
| 9 | **Eq. 6's policy-ratio importance weight is dropped** | The released code drops it too (`importance_sampling=False`); we match the code, not the equation. |
| 10 | **`SynthRL-o` not ported** | Needs a second synth; blocked on D-FAMILIES. |

## Caveats

- **Checkpoint selection in stage 2 is on `val_reward`** (logged as `val_loss = −val_reward` so the
  min-mode monitor picks the best epoch). That reward comes from a single greedy render per
  validation sample, so it is noisy. The repo does not early-stop and neither does
  `synthrl_i_config.yaml`.
- **Stage 2 runs at `32-true` precision**, not the cluster's usual `bf16-mixed`. The policy
  log-probability is a sum over ~100 categorical heads carrying the REINFORCE gradient; bf16's
  mantissa is not enough to keep it accurate.
- **The reward buffer is keyed on the target's ML-vector bytes.** Fine for a fixed corpus (same
  patch → same key), but it means the buffer grows with the training-set size: roughly
  `targets × m × parameters × 8` bytes.
