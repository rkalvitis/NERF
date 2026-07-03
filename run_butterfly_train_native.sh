#!/bin/bash
# ================================================================
# Max-resolution arm of the butterfly experiment: all 5 species at
# NATIVE resolution (fineview2nerf.py --width 0 picks, per species,
# the largest canvas that upscales no camera — e.g. 2494 px for 110,
# 1317 px for 195).
#
# Config (butterfly_config.txt) is deliberately UNTOUCHED — this is
# the "change only the resolution" experiment. Two knobs in it are
# denominated in pixels, so at native res they mean less supervision
# per pixel than at 800 px: precrop_iters (500) and the total ray
# budget (N_rand x N_iters). If a species collapses here but trains
# fine in the 800 px arm, scale precrop_iters by (width/800)^2 first.
#
# Launch exactly like run_butterfly_train_800.sh but with its own
# fresh output dir:
#
#   export CODE_DIR=/home/robertsk/NERF
#   export COLMAP_DATA=/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data/fineview_full_resolution_run
#   #  ^ the fineview_full_resolution_run SUBDIR — the top-level
#   #    3dgs_data/<species> dirs are a stale export with 0 points!
#   export DATA_DIR=/media/white/nanodrones/roberts.kalvitis/nerf/nerf_data
#   export OUTPUT_DIR=/media/white/nanodrones/roberts.kalvitis/nerf/butterfly_output_native
#   mkdir -p "$OUTPUT_DIR"
#   singularity exec --nv --cleanenv --contain \
#       --bind "$CODE_DIR:/workspace" \
#       --bind "$COLMAP_DATA:/colmap_data" \
#       --bind "$DATA_DIR:/data" \
#       --bind "$OUTPUT_DIR:/output" \
#       ~/containers/nerf.sif \
#       bash /workspace/run_butterfly_train_native.sh
#
# Expnames stay <species>_seed1 so compute_metrics.py works as-is
# (run it with --datadir /data/nerf_butterfly_native --factor 1 and
# this output dir bound to /output).
# ================================================================

DEVICE=2    # <-- set to your booked GPU index (0, 1, 2, ...)

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
COLMAP_DATA=/colmap_data                  # fineview_full_resolution_run scenes
DATA_NATIVE=/data/nerf_butterfly_native   # host: $DATA_DIR/nerf_butterfly_native
EXPERIMENTS=/output/experiments
RUN_LOGS=/output/run_logs

mkdir -p "$DATA_NATIVE" "$EXPERIMENTS" "$RUN_LOGS"

for SPECIES in $SPECIES_LIST; do

    RUN_NAME="${SPECIES}_seed${SEED}"
    LOG_FILE="${RUN_LOGS}/${RUN_NAME}.log"

    echo "========================================================"
    echo "  Run : $RUN_NAME  (native resolution)"
    echo "  Time: $(date)"
    echo "========================================================"

    # ------------------------------------------------------------
    # Guard — an existing experiment dir must belong to the native
    # data; otherwise a wrong /output bind would resume checkpoints
    # from a different arm of the experiment.
    # ------------------------------------------------------------
    ARGS_TXT="${EXPERIMENTS}/${RUN_NAME}/args.txt"
    if [ -f "$ARGS_TXT" ] && ! grep -q "nerf_butterfly_native" "$ARGS_TXT"; then
        echo "  ERROR: ${EXPERIMENTS}/${RUN_NAME} exists but was trained on"
        echo "         $(grep '^datadir' "$ARGS_TXT")"
        echo "         You are probably binding the wrong output dir."
        echo "         Bind a fresh one (butterfly_output_native) to /output."
        exit 1
    fi

    # ------------------------------------------------------------
    # Phase 1 — convert COLMAP -> poses_bounds.npy at native width
    # ------------------------------------------------------------
    if [ ! -f "$DATA_NATIVE/$SPECIES/poses_bounds.npy" ]; then
        echo "  Converting $SPECIES at native width..."
        python "$WORKSPACE/fineview2nerf.py" \
            --scene_dir "$COLMAP_DATA/$SPECIES" \
            --out_dir   "$DATA_NATIVE/$SPECIES" \
            --width 0 \
            2>&1 | tee "${RUN_LOGS}/${SPECIES}_convert.log"

        if [ ! -f "$DATA_NATIVE/$SPECIES/poses_bounds.npy" ]; then
            echo "  ERROR: conversion failed for $SPECIES — skipping."
            continue
        fi
    fi

    # ------------------------------------------------------------
    # Sanity check — 320 poses, plausibly-native image width,
    # real point-cloud bounds (not the empty-cloud heuristic).
    # ------------------------------------------------------------
    python - "$DATA_NATIVE/$SPECIES" <<'EOF'
import os, sys
import numpy as np
from PIL import Image

scene = sys.argv[1]
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
# native widths for this export are 1317..2494 px; anything smaller
# means a stale low-res conversion is being reused
assert w >= 1200, 'images only %d px wide — stale conversion?' % w
# Real point-cloud bounds on this telephoto rig give far/near ~1.1x;
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
        --datadir  "$DATA_NATIVE/$SPECIES" \
        --basedir  "$EXPERIMENTS" \
        --random_seed "$SEED" \
        2>&1 | tee "$LOG_FILE"

    echo ""
    echo "  Finished: $RUN_NAME  $(date)"
    echo ""
done

echo "========================================================"
echo "All native-resolution runs complete."
echo "Metrics (native data dir + this output bind):"
echo "  compute_metrics.py --datadir /data/nerf_butterfly_native --factor 1 --seeds 1 ..."
echo "========================================================"
