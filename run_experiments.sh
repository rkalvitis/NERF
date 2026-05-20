#!/bin/bash
# ================================================================
# NeRF — Table 1 reproduction: LLFF scenes × 5 seeds
# 8 scenes × 5 seeds = 40 sequential runs
# Estimated total: ~20 days on one RTX 4080
#
# HOW TO SET THE GPU
#   Change DEVICE below to the index of the GPU you have booked
#   (check with nvidia-smi on rhea before launching).
#   The variable is passed as CUDA_VISIBLE_DEVICES so the
#   container sees exactly one GPU regardless of what else is
#   running on the node.
# ================================================================

DEVICE=1    # <-- set to your booked GPU index (0, 1, 2, ...)

# ----------------------------------------------------------------
# Scene and seed lists — do not change these for Table 1
# ----------------------------------------------------------------
SEEDS="0 1 2 3 4"
LLFF_SCENES="fern flower fortress horns leaves orchids room trex"

# ----------------------------------------------------------------
# Resume from a specific seed/scene after a crash.
# Set RESUME_SEED and RESUME_SCENE to the first run that did NOT
# finish. Everything before this point is skipped unconditionally.
# Set RESUME_SEED="" to disable and run from the beginning.
# ----------------------------------------------------------------
RESUME_SEED=2
RESUME_SCENE=orchids

# ----------------------------------------------------------------
# Container-internal paths (set by --bind flags in the launch cmd)
# ----------------------------------------------------------------
WORKSPACE=/workspace
DATA=/data/nerf_llff_data
EXPERIMENTS=/output/experiments
RUN_LOGS=/output/run_logs

mkdir -p "$EXPERIMENTS" "$RUN_LOGS"

# ----------------------------------------------------------------
# Main loop: outer = seeds, inner = scenes.
# Completing one full seed gives a complete Table 1 comparison;
# additional seeds build variance estimates.
# ----------------------------------------------------------------
SKIPPING=false
if [ -n "$RESUME_SEED" ] && [ -n "$RESUME_SCENE" ]; then
    SKIPPING=true
fi

for SEED in $SEEDS; do
    for SCENE in $LLFF_SCENES; do

        # Fast-skip everything before the resume point.
        if [ "$SKIPPING" = true ]; then
            if [ "$SEED" -eq "$RESUME_SEED" ] && [ "$SCENE" = "$RESUME_SCENE" ]; then
                SKIPPING=false
            else
                echo "  Skipping $SCENE seed$SEED (before resume point)"
                continue
            fi
        fi

        RUN_NAME="${SCENE}_seed${SEED}"
        LOG_FILE="${RUN_LOGS}/${RUN_NAME}.log"

        echo "========================================================"
        echo "  Run : $RUN_NAME"
        echo "  Time: $(date)"
        echo "========================================================"

        # Resume support: skip runs that already finished.
        # A completed 200k run writes testset_200000/ at the very end.
        if [ -d "${EXPERIMENTS}/${RUN_NAME}/testset_200000" ]; then
            echo "  testset_200000 found — already complete, skipping."
            echo ""
            continue
        fi

        # PYTHONHASHSEED: seeds Python's built-in hash randomisation.
        # Must be set before Python starts, hence the env-var prefix.
        # TF_DETERMINISTIC_OPS=1 is set globally in the container's
        # %environment block, so cuDNN ops are deterministic here.
        PYTHONHASHSEED=$SEED \
        CUDA_VISIBLE_DEVICES=$DEVICE \
        TF_DETERMINISTIC_OPS=0 \
        python "$WORKSPACE/run_nerf.py" \
            --config "$WORKSPACE/paper_configs/llff_config.txt" \
            --expname  "$RUN_NAME" \
            --datadir  "$DATA/$SCENE" \
            --basedir  "$EXPERIMENTS" \
            --random_seed "$SEED" \
            2>&1 | tee "$LOG_FILE"

        echo ""
        echo "  Finished: $RUN_NAME  $(date)"
        echo ""
    done
done

echo "========================================================"
echo "All 40 runs complete."
echo "Metrics: run compute_metrics.py to get SSIM and LPIPS."
echo "========================================================"
