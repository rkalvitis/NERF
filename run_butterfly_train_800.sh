#!/bin/bash
# ================================================================
# Control experiment: the four remaining butterfly species at the
# proven 800 px width (009 already trained fine at 800 px).
#
# Purpose: 110 collapsed when trained on full-resolution (2179 px)
# data; this reruns the others on 800 px data with the config
# untouched, to separate "scene is hard" from "resolution broke it".
#
# Differences from run_butterfly_train.sh:
#   * converts into /data/nerf_butterfly_800 (width-keyed dir) — never
#     touches the full-res data, and a changed width can never
#     silently reuse a stale conversion
#   * verifies the converted images really are 800 px before training
#   * refuses to resume an experiment trained on different data
#     (protects against launching with the old /output bind)
#
# Expnames stay <species>_seed1 so compute_metrics.py works as-is;
# isolation comes from binding a FRESH host dir to /output:
#
#   export COLMAP_DATA=/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data/fineview_full_resolution_run
#   #  ^ the fineview_full_resolution_run SUBDIR — the top-level
#   #    3dgs_data/<species> dirs are a stale export with 0 points!
#   export OUTPUT_DIR=/media/white/nanodrones/roberts.kalvitis/nerf/butterfly_output_800
#   mkdir -p "$OUTPUT_DIR"
#   singularity exec --nv --cleanenv --contain \
#       --bind "$CODE_DIR:/workspace" \
#       --bind "$COLMAP_DATA:/colmap_data" \
#       --bind "$DATA_DIR:/data" \
#       --bind "$OUTPUT_DIR:/output" \
#       ~/containers/nerf.sif \
#       bash /workspace/run_butterfly_train_800.sh
#
# (CODE_DIR / COLMAP_DATA / DATA_DIR exactly as in RUN_BUTTERFLY.md
# Phase 4 — only OUTPUT_DIR differs.)
# ================================================================

DEVICE=2    # <-- set to your booked GPU index (0, 1, 2, ...)

SEED=1
WIDTH=800
SPECIES_LIST="
072-Colias_Eurytheme-002
110-Nymphalis_l_album-001
184-Speyeria_Hydaspe-001
195-Lycaena_Arota-002
"

# ----------------------------------------------------------------
# Container-internal paths (set by --bind flags in the launch cmd)
# ----------------------------------------------------------------
WORKSPACE=/workspace
COLMAP_DATA=/colmap_data                  # 3DGS fineview COLMAP scenes
DATA_800=/data/nerf_butterfly_${WIDTH}    # host: $DATA_DIR/nerf_butterfly_800
EXPERIMENTS=/output/experiments
RUN_LOGS=/output/run_logs

mkdir -p "$DATA_800" "$EXPERIMENTS" "$RUN_LOGS"

for SPECIES in $SPECIES_LIST; do

    RUN_NAME="${SPECIES}_seed${SEED}"
    LOG_FILE="${RUN_LOGS}/${RUN_NAME}.log"

    echo "========================================================"
    echo "  Run : $RUN_NAME  (width $WIDTH)"
    echo "  Time: $(date)"
    echo "========================================================"

    # ------------------------------------------------------------
    # Guard — if this experiment dir already exists, it must have
    # been trained on the 800px data. Otherwise the old full-res
    # run was bound to /output and resuming would load its
    # collapsed checkpoints.
    # ------------------------------------------------------------
    ARGS_TXT="${EXPERIMENTS}/${RUN_NAME}/args.txt"
    if [ -f "$ARGS_TXT" ] && ! grep -q "nerf_butterfly_${WIDTH}" "$ARGS_TXT"; then
        echo "  ERROR: ${EXPERIMENTS}/${RUN_NAME} exists but was trained on"
        echo "         $(grep '^datadir' "$ARGS_TXT")"
        echo "         You are probably binding the OLD output dir."
        echo "         Bind a fresh one (butterfly_output_800) to /output."
        exit 1
    fi

    # ------------------------------------------------------------
    # Phase 1 — convert COLMAP -> poses_bounds.npy at $WIDTH px
    # ------------------------------------------------------------
    if [ ! -f "$DATA_800/$SPECIES/poses_bounds.npy" ]; then
        echo "  Converting $SPECIES at width $WIDTH..."
        python "$WORKSPACE/fineview2nerf.py" \
            --scene_dir "$COLMAP_DATA/$SPECIES" \
            --out_dir   "$DATA_800/$SPECIES" \
            --width $WIDTH \
            2>&1 | tee "${RUN_LOGS}/${SPECIES}_convert.log"

        if [ ! -f "$DATA_800/$SPECIES/poses_bounds.npy" ]; then
            echo "  ERROR: conversion failed for $SPECIES — skipping."
            continue
        fi
    fi

    # ------------------------------------------------------------
    # Sanity check — 320 poses, images actually $WIDTH px wide,
    # real point-cloud bounds (not the empty-cloud heuristic).
    # ------------------------------------------------------------
    python - "$DATA_800/$SPECIES" $WIDTH <<'EOF'
import os, sys
import numpy as np
from PIL import Image

scene, width = sys.argv[1], int(sys.argv[2])
pb = np.load(os.path.join(scene, 'poses_bounds.npy'))
imgdir = os.path.join(scene, 'images')
files = sorted(os.listdir(imgdir))
w, h = Image.open(os.path.join(imgdir, files[0])).size
near, far = pb[:, -2].min(), pb[:, -1].max()
ratio = (pb[:, -1] / pb[:, -2]).max()
print('  check: %d poses, %d images, %dx%d px, near %.2f far %.2f '
      '(worst far/near %.1fx)'
      % (pb.shape[0], len(files), w, h, near, far, ratio))
assert pb.shape[0] == len(files), 'pose count != image count'
assert w == width, 'images are %d px wide, expected %d' % (w, width)
# Real point-cloud bounds on this telephoto rig give far/near ~1.5x;
# the empty-point-cloud heuristic gives exactly 20x and trains to fog.
assert ratio < 5, ('far/near ratio %.1fx — heuristic bounds from an '
                   'empty point cloud?' % ratio)
EOF
    if [ $? -ne 0 ]; then
        echo "  ERROR: sanity check failed for $SPECIES — skipping."
        continue
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
        --datadir  "$DATA_800/$SPECIES" \
        --basedir  "$EXPERIMENTS" \
        --random_seed "$SEED" \
        2>&1 | tee "$LOG_FILE"

    echo ""
    echo "  Finished: $RUN_NAME  $(date)"
    echo ""
done

echo "========================================================"
echo "All 800px control runs complete."
echo "Metrics (note the _800 data dir and output bind):"
echo "  compute_metrics.py --datadir /data/nerf_butterfly_800 --factor 1 --seeds 1 ..."
echo "========================================================"
