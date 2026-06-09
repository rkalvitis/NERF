# NeRF on the FineView butterfly dataset (rhea)

Trains NeRF on the **same 5 FineView butterfly species** used for the 3DGS
FineView runs — the exact COLMAP scenes produced by the 3DGS
`fineview_pipeline` (calibrated poses + masked white-background images) are
converted to NeRF's input format and trained for 200k iterations each.

**Branch:** `nerf-butterfly`
**Scale:** 5 species × 1 seed (seed 1, same as the 3DGS FineView runs)
**Estimated time:** ~5 min conversion + ~11–12 h training per species → **~2.5 days total**

> Note on naming: NeRF's data loader is called "llff" (`dataset_type = llff`)
> because it reads the `poses_bounds.npy` file format. We use that loader for
> the butterfly data — this has nothing to do with the LLFF fern/flower/…
> scenes from the Table 1 reproduction.

---

## Prerequisites (already done if the 3DGS FineView run happened)

| What | Where |
|---|---|
| COLMAP butterfly scenes (input) | `/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data/<species>/` — must contain `sparse/0/{cameras.txt, images.txt, points3D.bin}` and `images/camera1..8/` |
| NeRF Singularity container | `~/containers/nerf.sif` (built from `nerf.def` — see REPRODUCTION.md Phase 4) |
| Torch cache (LPIPS weights) | `/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache` |

New paths created by this run:

| What | rhea path |
|---|---|
| Converted NeRF scenes | `/media/white/nanodrones/roberts.kalvitis/nerf/nerf_data/nerf_butterfly/<species>/` |
| Training output | `/media/white/nanodrones/roberts.kalvitis/nerf/butterfly_output/` |

---

## Files on this branch

| File | Purpose |
|---|---|
| `fineview2nerf.py` | Converts a COLMAP scene → flat `images/` + `poses_bounds.npy`. Unifies the 8 physical cameras into one virtual camera (common focal, common 800-px canvas, white padding) because NeRF assumes a single H/W/focal |
| `paper_configs/butterfly_config.txt` | 360° object-scene config: `no_ndc`, `spherify`, `lindisp`, `factor 1`, 200k iters |
| `run_butterfly_train.sh` | Converts (if needed) + trains all 5 species sequentially; set `DEVICE=` at the top |
| `compute_metrics.py` | Now accepts `--datadir` / `--factor` so it works for butterfly scenes too |

---

## Phase 1 — Push the branch from your Mac

```bash
cd /Users/robertskalvitis/Documents/repos/NERF

git add fineview2nerf.py \
        paper_configs/butterfly_config.txt \
        run_butterfly_train.sh \
        compute_metrics.py \
        RUN_BUTTERFLY.md
# (CLAUDE.md is gitignored in this repo — it stays local for Claude Code sessions)
git commit -m "Add FineView butterfly pipeline: converter, config, run script"
git push origin nerf-butterfly
```

### Optional — local smoke test of the converter (Mac)

A FineView COLMAP scene exists locally in the 3DGS repo:

```bash
python3.10 fineview2nerf.py \
    --scene_dir /Users/robertskalvitis/Documents/repos/3DGS/fineview_colmap/009-Neophasia_Menapia-001 \
    --out_dir /tmp/nerf_butterfly_test/009 --width 800
```

Expected: `320 poses`, all images 800×572. The local scene has an empty
point cloud so it warns about heuristic bounds — on rhea the triangulated
`points3D.bin` gives real near/far bounds.

---

## Phase 2 — Set up rhea

```bash
ssh robertsk@rhea.idsia.ch

git -C ~/NERF pull
git -C ~/NERF checkout nerf-butterfly

mkdir -p /media/white/nanodrones/roberts.kalvitis/nerf/nerf_data
mkdir -p /media/white/nanodrones/roberts.kalvitis/nerf/butterfly_output
```

Verify the COLMAP scenes are in place:

```bash
ls /media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data/
# Expected: 009-Neophasia_Menapia-001/ 072-Colias_Eurytheme-002/ ...
ls /media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data/009-Neophasia_Menapia-001/sparse/0/
# Expected: cameras.txt images.txt points3D.bin
```

---

## Phase 3 — Book a GPU

1. Check **MSTeams** (`rhea-users` channel) for a free GPU.
2. Confirm it is idle: `nvidia-smi -l 1` (0 MiB usage, 0% util).
3. Post `"Using GPU X"` on MSTeams.
4. Edit `~/NERF/run_butterfly_train.sh` and set `DEVICE=X` at the top.

---

## Phase 4 — Launch

Run inside screen — the job takes ~2.5 days:

```bash
screen -S butterfly -U
```

Then (single launch command):

```bash
export CODE_DIR=/home/robertsk/NERF
export COLMAP_DATA=/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data
export DATA_DIR=/media/white/nanodrones/roberts.kalvitis/nerf/nerf_data
export OUTPUT_DIR=/media/white/nanodrones/roberts.kalvitis/nerf/butterfly_output

singularity exec --nv --cleanenv --contain \
    --bind "$CODE_DIR:/workspace" \
    --bind "$COLMAP_DATA:/colmap_data" \
    --bind "$DATA_DIR:/data" \
    --bind "$OUTPUT_DIR:/output" \
    ~/containers/nerf.sif \
    bash /workspace/run_butterfly_train.sh
```

Detach: **Ctrl+A, D** — reattach: `screen -r butterfly -U`

**Restart safety:** re-running the same command resumes automatically —
conversion is skipped if `poses_bounds.npy` exists, training is skipped if
`testset_200000/` exists.

**PTX JIT note:** on the very first run TF silently JIT-compiles kernels for
sm_89 (3–5 min, no output). Do not kill the process if it looks frozen.

---

## Phase 5 — Monitor

```bash
# Completed species (of 5)
ls /media/white/nanodrones/roberts.kalvitis/nerf/butterfly_output/experiments/*/testset_200000 -d 2>/dev/null | wc -l

# Live training log
tail -f /media/white/nanodrones/roberts.kalvitis/nerf/butterfly_output/run_logs/009-Neophasia_Menapia-001_seed1.log

# GPU
nvidia-smi
```

PSNR is printed every 100 iterations:
`<run_name>  iter  psnr  loss  global_step`

**Sanity-check the first run early.** After ~10–20k iterations PSNR should
be well above ~20 dB (mostly white background makes PSNR high). If it
plateaus near 12–15 dB or the renders in `tboard_val_imgs/` are empty/fog,
stop and investigate the converted scene before burning days of GPU time —
this conversion path is new on this branch.

---

## Phase 6 — Collect results

PSNR for one run from its log:

```bash
grep "009-Neophasia_Menapia-001_seed1" \
    /media/white/nanodrones/roberts.kalvitis/nerf/butterfly_output/run_logs/009-Neophasia_Menapia-001_seed1.log | tail -1
```

SSIM/LPIPS from the saved test renders (note `--datadir` and `--factor 1`,
plus seed 1 and the species names as scenes):

```bash
singularity exec --nv --cleanenv --contain \
    --bind /media/white/nanodrones/roberts.kalvitis/nerf/butterfly_output:/output \
    --bind /media/white/nanodrones/roberts.kalvitis/nerf/nerf_data:/data \
    --bind /home/robertsk/NERF:/workspace \
    --bind /media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache:/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache \
    ~/containers/nerf.sif \
    python /workspace/compute_metrics.py \
        --datadir /data/nerf_butterfly --factor 1 \
        --seeds 1 \
        --scenes 009-Neophasia_Menapia-001 072-Colias_Eurytheme-002 \
                 110-Nymphalis_l_album-001 184-Speyeria_Hydaspe-001 \
                 195-Lycaena_Arota-002 \
        --out /output/butterfly_results.csv
```

The test split (every 8th image, `llffhold = 8`) matches the 3DGS `--eval`
protocol, so these metrics are directly comparable to the 3DGS FineView
`results.json` numbers.

Free the GPU when done: confirm with `nvidia-smi`, post "GPU X is free" on MSTeams.

---

## How the conversion works (fineview2nerf.py)

NeRF's loader requires a single camera (one H, W, focal) and equally-sized
images; FineView scenes have 8 calibrated cameras with slightly different
focal lengths (~10200–10440 px) and crop sizes. The converter:

1. Picks a common focal so the median camera lands at 800 px wide.
2. Rescales each camera's images by `F_common / fx_camera` — after this,
   every image shares the same focal length.
3. Center-crops / white-pads to one common canvas (principal points are
   centered in the FineView export; the background is pure white, so the
   padding adds only correct background pixels).
4. Converts COLMAP world-to-camera poses to LLFF-convention camera-to-world
   3×5 matrices and writes `poses_bounds.npy`; near/far come from the depth
   percentiles of the triangulated point cloud per view.
5. Flattens `images/camera<N>/<M>.png` → `images/camera<N>_<M>.png`
   (pose order matches the loader's sorted-filename order) and adds an
   `images_1` symlink so `factor = 1` skips the ImageMagick minify path.

## Training settings (butterfly_config.txt)

| Setting | Value | Reason |
|---|---|---|
| `no_ndc` | True | NDC is for forward-facing scenes; this is a 360° object capture |
| `spherify` | True | Recenters/normalizes inward-facing poses, spherical render path |
| `lindisp` | True | Sample in inverse depth — standard for 360° scenes |
| `factor` | 1 | Images already resized to 800 px by the converter |
| `llffhold` | 8 | Every 8th image → test (40 test views), same as 3DGS `--eval` |
| `N_rand` | 1024 | RTX 4080 16 GB memory limit (see REPRODUCTION.md) |
| `N_iters` | 200000 | Same budget as the LLFF Table 1 reproduction runs |
| `raw_noise_std` | 1.0 | Density regularizer, suppresses floaters |
| seed | 1 | Matches the 3DGS FineView runs |

## Quick reference

| Item | Detail |
|---|---|
| Branch | `nerf-butterfly` |
| Container | `~/containers/nerf.sif` (same as LLFF reproduction) |
| Species | 009-Neophasia_Menapia-001, 072-Colias_Eurytheme-002, 110-Nymphalis_l_album-001, 184-Speyeria_Hydaspe-001, 195-Lycaena_Arota-002 |
| Input | 320 views/species (8 cameras × 40 positions), white background |
| Converted size | 800 × ~570 px, single shared focal (~3890 px) |
| If SSH drops | `screen -r butterfly -U` |
| Results | `butterfly_output/run_logs/*.log` (PSNR) + `compute_metrics.py` (SSIM/LPIPS) |
