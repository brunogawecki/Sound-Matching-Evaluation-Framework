#!/usr/bin/env bash
# Score every trained family on the out-of-domain NSynth corpora (D-OOD).
#
# No training: this reuses the in-domain checkpoints unchanged and only runs the Evaluator,
# which re-renders each prediction fresh-process (D-REPRO), so it needs both VSTs locally.
#
# Ordering is deliberate (D-EVAL-DEVICE): cheap families first, so an interrupted sweep still
# leaves a usable partial table; IS2 on mps, where its finetuning loop is 7.4x faster; and the
# flow-matching pair on cpu, where mps is 23% *slower* because the 400 sequential batch-1
# transformer passes are kernel-launch-latency bound.
#
# Resumable: a cell whose eval_summary.json already exists is skipped, so re-running after an
# interruption picks up where it stopped. Delete a results folder to force a re-score.
#
#   caffeinate -ims ./scripts/run_ood_sweep.sh 2>&1 | tee ood_sweep.log
set -uo pipefail
cd "$(dirname "$0")/.."

run_cell() {  # synth_corpus model checkpoint device
    local corpus="$1" model="$2" checkpoint="$3" device="$4"
    if [ -f "results/${corpus}/${model}/eval_summary.json" ]; then
        echo "== skip ${corpus}/${model} (already scored)"
        return 0
    fi
    if [ ! -f "$checkpoint" ]; then
        echo "!! skip ${corpus}/${model}: checkpoint missing ($checkpoint)"
        return 0
    fi
    echo "== ${corpus}/${model} on ${device}  [$(date +%H:%M:%S)]"
    python scripts/evaluate.py --model "$model" --checkpoint "$checkpoint" \
        --corpus "dataset/${corpus}" --device "$device" \
        || echo "!! FAILED ${corpus}/${model} -- continuing"
}

started=$(date +%s)

# ---- Dexed ----
run_cell nsynth_c4_dexed Sound2SynthSpectrogramRegressor checkpoints/1073236/spectrogram_cnn.pt cpu
run_cell nsynth_c4_dexed PresetGenVAEMLPRegressor        checkpoints/1073699/presetgen_vae_mlp.pt cpu
run_cell nsynth_c4_dexed PresetGenVAEFlowRegressor       checkpoints/1073700/presetgen_vae_flow.pt cpu
run_cell nsynth_c4_dexed IS                              checkpoints/992632/inversynth_is.pt cpu
run_cell nsynth_c4_dexed IS2xITF                         checkpoints/993074/inversynth_is2xitf.pt cpu
run_cell nsynth_c4_dexed SynthRLp                        checkpoints/996865/synthrl_p.pt cpu
run_cell nsynth_c4_dexed SynthRLi                        checkpoints/1007123/synthrl_i.pt cpu
run_cell nsynth_c4_dexed IS2                             checkpoints/993088/inversynth_is2.pt mps
run_cell nsynth_c4_dexed FlowMatchingMLP                 checkpoints/994884/flow_matching_mlp.pt cpu
run_cell nsynth_c4_dexed FlowMatchingParam2Tok           checkpoints/994886/flow_matching_param2tok.pt cpu

# ---- Diva (no SynthRLi: never trained on Diva, D-RL-RENDER) ----
run_cell nsynth_c4_diva Sound2SynthSpectrogramRegressor checkpoints/1073242/spectrogram_cnn.pt cpu
run_cell nsynth_c4_diva PresetGenVAEMLPRegressor        checkpoints/1073776/presetgen_vae_mlp.pt cpu
run_cell nsynth_c4_diva PresetGenVAEFlowRegressor       checkpoints/1073698/presetgen_vae_flow.pt cpu
run_cell nsynth_c4_diva IS                              checkpoints/1073817/inversynth_is.pt cpu
run_cell nsynth_c4_diva IS2xITF                         checkpoints/1073818/inversynth_is2xitf.pt cpu
run_cell nsynth_c4_diva SynthRLp                        checkpoints/1073820/synthrl_p.pt cpu
run_cell nsynth_c4_diva IS2                             checkpoints/1073819/inversynth_is2.pt mps
run_cell nsynth_c4_diva FlowMatchingMLP                 checkpoints/1073243/flow_matching_mlp.pt cpu
run_cell nsynth_c4_diva FlowMatchingParam2Tok           checkpoints/1073244/flow_matching_param2tok.pt cpu

echo
echo "sweep finished in $((($(date +%s) - started) / 60)) min"
echo "scored cells:"
find results/nsynth_c4_dexed results/nsynth_c4_diva -name eval_summary.json | sort
