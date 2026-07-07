# Running training on the PUT GPU cluster

This directory packages the training path to run on the Poznań University of Technology
SLURM cluster (`slurm.cs.put.poznan.pl`, `hgx` partition — A100-80GB GPUs). Training reads
a self-describing corpus and never touches a VST (D-SELFDESC), so the cluster needs only the
VST-free dependency split in `../requirements-cluster.txt`, not the full stack.

The flow is: **build the corpus locally → push it up → submit an sbatch job → pull the
checkpoint down → score it with the local Evaluator.** Only the training step runs on the
cluster; everything else stays on the laptop.

## Files

| File | Where it runs | What it does |
|---|---|---|
| `../requirements-cluster.txt` | cluster | The complete VST-free dependency set (pinned). |
| `cluster.env.example` | both | Template for the gitignored `cluster.env` (paths, SSH target, account). |
| `train.sbatch` | cluster | SLURM job: activates the conda env and runs `scripts/fit_model.py`. |
| `training_configs/smoke_config.yaml` | cluster | Reduced-scale config for the acceptance run (2 epochs). |
| `training_configs/full_config.yaml` | cluster | Full training run (30 epochs, Sound2Synth paper hyperparameters). |
| `push_corpus.sh` | laptop | `rsync` a corpus dir up to the cluster. |
| `pull_checkpoint.sh` | laptop | `rsync` the trained checkpoint (+ logs) back down. |

## One-time cluster setup

Access itself has to be requested first: email `obliczenia@cs.put.poznan.pl` (name, PUT
student number, supervisor, project name, whether GPU access is needed) — the reply carries
your LDAP login and your SLURM billing account name (e.g. `ai`), which is the `SLURM_ACCOUNT`
value used below and at submission time.

Three shells are involved: **your laptop**, **the login node** (reached via SSH), and an
**interactive compute-node shell** started from inside that same SSH session with `srun`.
Installing software must happen in that interactive shell, not on the login node directly —
the login node is only for submitting/monitoring jobs, and the interactive session is what
actually schedules you onto a real node with a GPU so `pip`/`conda` see the same environment
training will run in.

1. **(On your laptop) SSH in** (LDAP credentials):

   ```bash
   ssh your_login@slurm.cs.put.poznan.pl
   ```

   Your prompt changes (e.g. `you@svradmin:~$`) — that's the signal you're now on the login
   node. Everything through step 6 runs from this SSH session.

2. **(On the login node) Clone the repo** (it is public, so no auth needed). Every run is then
   traceable to a commit:

   ```bash
   git clone https://github.com/brunogawecki/Sound-Matching-Evaluation-Framework.git
   cd Sound-Matching-Evaluation-Framework
   ```

3. **(On the login node) Start an interactive session** on an `hgx` GPU node, so the install
   below lands on a node that actually has the GPU driver visible:

   ```bash
   srun -p hgx -w hgx1 --gres=gpu:1 --pty /bin/bash -l
   ```

   This queues like any other job — it starts as soon as a GPU is free. Your prompt changes
   again (e.g. `you@hgx1:~$`); steps 4–5 run here. `/home` (and this checkout) is shared
   across every node, so anything installed here is visible from the login node and from
   batch jobs too.

4. **(On the interactive session) Install Miniconda** into your home dir (no root):

   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
   ```

5. **(On the interactive session) Create the env and install the VST-free deps** — run from
   inside the repo checkout from step 2, so `requirements-cluster.txt` is right there:

   ```bash
   source "$HOME/miniconda3/etc/profile.d/conda.sh"
   conda create -y -n smef python=3.11
   conda activate smef
   pip install -r requirements-cluster.txt
   ```

   If `conda create` fails with `CondaToSNonInteractiveError` (Anaconda now gates the default
   channels behind a Terms of Service acceptance), accept them once and retry:

   ```bash
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
   ```

   Conda only provides a user-space Python 3.11; `pip` installs everything from
   `requirements-cluster.txt`. On Linux `pip install torch==...` fetches the CUDA wheel
   automatically. Sanity check while still on the GPU node: `python3 -c "import torch;
   print(torch.cuda.is_available())"` should print `True`. Then `exit` to leave the
   interactive session and return to the login node.

6. **(On the login node) Configure `cluster.env`**, still inside the checkout:

   ```bash
   cp cluster/cluster.env.example cluster/cluster.env
   # edit cluster/cluster.env: set SLURM_ACCOUNT (from the activation email), CONDA_BASE,
   # REMOTE_REPO_DIR, REMOTE_CORPUS_DIR, CONDA_ENV=smef, etc.
   ```

## Per-run workflow

Every run alternates machines: laptop (transfer) → cluster (submit/monitor) → laptop
(transfer). Fill in `LOCAL_CORPUS_DIR` / `CLUSTER_SSH` / `REMOTE_CORPUS_DIR` etc. in a
`cluster.env` on your laptop too (copied from `cluster.env.example` the same way as step 6
above) — the transfer scripts read it from wherever you run them.

1. **(On your laptop) Push the corpus.** Dry-run first if unsure:

   ```bash
   cluster/push_corpus.sh -n     # preview
   cluster/push_corpus.sh        # transfer (~10 GB for the preset-gen-vae corpus)
   ```

2. **(On the login node) Sync the code** to the commit you want to train from:

   ```bash
   cd "$REMOTE_REPO_DIR" && git pull
   ```

3. **(On the login node) Submit the job** from the repo root. The billing account is a
   submission-time flag, so pass it explicitly (it is also stored in `cluster.env`):

   ```bash
   sbatch -A "$SLURM_ACCOUNT" cluster/train.sbatch
   ```

   `sbatch` and `srun` are both run from the login node — training itself runs unattended on
   whichever `hgx` node SLURM schedules it to; you don't need an interactive session for this
   (that's only for installing software, per the setup above).

4. **(On the login node) Monitor:**

   ```bash
   squeue -u "$USER"            # queue / running state
   tail -f slurm-<JOBID>.out    # live stdout (tqdm progress, per-epoch loss)
   sacct -j <JOBID>             # final state (COMPLETED / FAILED / TIMEOUT)
   scancel <JOBID>              # cancel it if needed
   ```

   The job requests one GPU (`--gres=gpu:1`) with a 12-hour wall-clock limit
   (`--time=12:00:00`), sized for the full run and still under the 24 h cap. Lower
   `--time` (e.g. back to `03:00:00`) if submitting a quick smoke-test run instead.

5. **(On your laptop) Pull the checkpoint** back down:

   ```bash
   cluster/pull_checkpoint.sh
   ```

6. **(On your laptop) Score it locally** with the Evaluator (`scripts/evaluate.py`), which
   needs only the base (VST) environment, not conda and not the cluster — the checkpoint is
   self-describing (D-SELFDESC).

## The full run

`train.sbatch` + `training_configs/smoke_config.yaml` is the reduced-scale acceptance slice;
`training_configs/full_config.yaml` is the real run (30 epochs, `val_fraction: 0.1`,
hyperparameters carried over from the Sound2Synth paper's own `train.py`/`sound2synth.py`
defaults where `models/training/` supports them — see `docs/PAPER_CANDIDATES.md`). Point
`TRAINING_CONFIG=cluster/training_configs/full_config.yaml` in `cluster.env` to use it;
`train.sbatch`'s `--time=12:00:00` is already sized for it. Nothing else changes.

## Experiment tracking (wandb)

Off by default. A `CSVLogger` always writes `lightning_logs/`. To also stream metrics to
Weights & Biases (deep families only; `MeanParameterBaseline` ignores the training config):

1. Add a `logger` block to the training config (`project`/`entity`/`run_name` are
   optional overrides; the project defaults to `Sound-Matching-Evaluation-Framework`).
   `run_name` is left unset by default, so wandb auto-names each run with a unique
   memorable label; runs are identified by the `architecture`/`dataset` config columns
   (below) rather than the name. Set `run_name` explicitly to override:
   ```yaml
   logger:
     wandb: true
   ```
2. Set `WANDB_API_KEY` in `cluster.env` (from <https://wandb.ai/authorize>; exported by
   `train.sbatch`). `WANDB_MODE` defaults to `online`; `WANDB_ENTITY` is optional (blank
   uses your account's default entity).

Each run also logs `architecture` (model class) and `dataset` (corpus dir name) as top-level
config columns, so the wandb Runs table can Group by / filter on either one directly rather than
parsing the run name.

If the compute node has no outbound internet, set `WANDB_MODE=offline` in `cluster.env`; the
run writes to `lightning_logs/wandb/offline-run-*` and you sync it afterward from a networked
machine with `wandb sync lightning_logs/wandb/offline-run-*`.

## Notes

- **Job time limit.** The cluster sends SIGTERM at the wall-clock limit, then SIGKILL 60
  minutes later if the job hasn't exited. The harness attaches
  `SLURMEnvironment(auto_requeue=True)` when SLURM is detected
  (`models/training/trainer_factory.py`), so a timed-out job checkpoints and requeues — treat
  this as a safety net, not a tested path. Keep `--time` comfortably above the run's expected
  length; the cluster-wide cap is 24h per job.
- **No `.env` needed.** `config.py` has defaults for every value and asserts nothing, so the
  training path runs on the cluster with no `.env` file.
- **Corpus lives in `/home`.** Read in place over the shared Lustre filesystem; no node-local
  `/raid` staging (premature at this corpus size).
