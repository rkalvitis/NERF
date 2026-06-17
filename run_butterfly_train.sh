#!/bin/bash
# ================================================================
# NeRF on the FineView butterfly dataset — 5 species, seed 1
#
# Input : COLMAP scenes produced by the 3DGS fineview_pipeline
#         (bound at /colmap_data — see RUN_BUTTERFLY.md)
# Output: converted NeRF scenes in /data/nerf_butterfly/<species>
#         trained runs in /output/experiments/<species>_seed1
#
# Per species: convert (once, ~5 min) + train 200k iters (~11-12 h)
# Total: ~2.5 days on one RTX 4080
#
# HOW TO SET THE GPU
#   Change DEVICE below to the index of the GPU you have booked
#   (check with nvidia-smi on rhea before launching).
# ================================================================

DEVICE=0    # <-- set to your booked GPU index (0, 1, 2, ...)

SEED=1
SPECIES_LIST="
009-Neophasia_Menapia-001
072-Colias_Eurytheme-002
110-Nymphalis_l_album-001
184-Speyeria_Hydaspe-001
195-Lycaena_Arota-002
"

# ----------------------------------------------------------------
# Container-internal paths (set by --bind flags in the launch cmd)
# ----------------------------------------------------------------
WORKSPACE=/workspace
COLMAP_DATA=/colmap_data          # 3DGS fineview COLMAP scenes (read-only ok)
DATA=/data/nerf_butterfly         # converted poses_bounds.npy + images
EXPERIMENTS=/output/experiments
RUN_LOGS=/output/run_logs

mkdir -p "$DATA" "$EXPERIMENTS" "$RUN_LOGS"

for SPECIES in $SPECIES_LIST; do

    RUN_NAME="${SPECIES}_seed${SEED}"
    LOG_FILE="${RUN_LOGS}/${RUN_NAME}.log"

    echo "========================================================"
    echo "  Run : $RUN_NAME"
    echo "  Time: $(date)"
    echo "========================================================"

    # ------------------------------------------------------------
    # Phase 1 — convert COLMAP -> poses_bounds.npy (once per species)
    # ------------------------------------------------------------
    if [ ! -f "$DATA/$SPECIES/poses_bounds.npy" ]; then
        echo "  Converting $SPECIES to NeRF format..."
        python "$WORKSPACE/fineview2nerf.py" \
            --scene_dir "$COLMAP_DATA/$SPECIES" \
            --out_dir   "$DATA/$SPECIES" \
            --width 2179 \
            2>&1 | tee "${RUN_LOGS}/${SPECIES}_convert.log"

        if [ ! -f "$DATA/$SPECIES/poses_bounds.npy" ]; then
            echo "  ERROR: conversion failed for $SPECIES — skipping."
            continue
        fi
    fi

    # ------------------------------------------------------------
    # Phase 2 — train. Skip runs that already finished (a completed
    # 200k run writes testset_200000/ at the very end).
    # ------------------------------------------------------------
    if [ -d "${EXPERIMENTS}/${RUN_NAME}/testset_200000" ]; then
        echo "  testset_200000 found — already complete, skipping."
        echo ""
        continue
    fi

    PYTHONHASHSEED=$SEED \
    CUDA_VISIBLE_DEVICES=$DEVICE \
    TF_DETERMINISTIC_OPS=0 \
    python "$WORKSPACE/run_nerf.py" \
        --config "$WORKSPACE/paper_configs/butterfly_config.txt" \
        --expname  "$RUN_NAME" \
        --datadir  "$DATA/$SPECIES" \
        --basedir  "$EXPERIMENTS" \
        --random_seed "$SEED" \
        2>&1 | tee "$LOG_FILE"

    echo ""
    echo "  Finished: $RUN_NAME  $(date)"
    echo ""
done

echo "========================================================"
echo "All butterfly runs complete."
echo "Metrics: run compute_metrics.py to get SSIM and LPIPS."
echo "========================================================"
