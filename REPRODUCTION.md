# Reproducing NeRF Table 1 on rhea

Replicates **Mildenhall et al. 2020 (NeRF) Table 1** — PSNR/SSIM/LPIPS across 8 LLFF forward-facing scenes, each run 5 times with seeds 0–4. Runs sequentially on a single RTX 4080 16 GB via Singularity on the rhea server.

**Scale:** 8 scenes × 5 seeds = 40 runs  
**Estimated time:** ~12 h per run → ~20 days total

---

## Files in this directory

| File | Purpose |
|---|---|
| `run_nerf.py` | Training script — `--random_seed` and `--N_iters` added vs. original |
| `paper_configs/llff_config.txt` | LLFF hyperparameters for Table 1 — `N_iters = 200000` added |
| `nerf.def` | Singularity definition file — nvidia-tensorflow 1.15 on CUDA 11.8 |
| `run_experiments.sh` | Runs all 8 scenes × 5 seeds; set `DEVICE=` at the top |
| `compute_metrics.py` | Computes SSIM + LPIPS from saved test renders (**still to create**) |

---

## Seed situation — read before running

**`--random_seed` exists but defaults to `None` (unseeded) in the original code.**

The `run_nerf.py` seeds two sources when `--random_seed N` is passed:

```python
np.random.seed(args.random_seed)               # ray shuffle, image selection
tf.compat.v1.set_random_seed(args.random_seed) # TF graph-level RNG
```

`run_experiments.sh` additionally sets `PYTHONHASHSEED=$SEED` before each Python call, which seeds Python's built-in hash randomisation (dict/set ordering).

**What is still non-deterministic even with seeds fixed:**

- `tf.random.uniform` (stratified ray perturbation) and `tf.random.normal` (density noise) are TF1 eager ops. Their op-level seeds are partially controlled by the TF global seed but not guaranteed to be fully reproducible across restarts.
- GPU atomics in cuDNN reductions are non-deterministic by default. The container sets `TF_DETERMINISTIC_OPS=1` to switch cuDNN to deterministic algorithms. This helps but does not fully eliminate all GPU-level variance.

**Practical impact:** expect ~0.1–0.3 dB PSNR variance run-to-run. The original paper reports **single runs** with no seeds mentioned. The 5-seed setup here is for statistical confidence, not because the paper did it.

---

## Changes made to the original code

Only minimal additions; all original defaults are preserved:

| File | Change |
|---|---|
| `run_nerf.py` | Added `--N_iters` argument (default `1000000` = original hardcoded value) |
| `run_nerf.py` | Line 746: `N_iters = args.N_iters` instead of hardcoded `1000000` |
| `paper_configs/llff_config.txt` | Added `N_iters = 200000` (matches the comment already in that file) |

Everything else in `run_nerf.py` (all hyperparameter defaults, training loop, logging frequency, checkpoint frequency) is unchanged from the original.

---

## Critical GPU compatibility issue

`environment.yml` pins `cudatoolkit=10.0` and `tensorflow-gpu==1.15`. **CUDA 10.0 does not support the RTX 4080 (sm_89).** The standard TF 1.15 wheel will fail to run GPU kernels on rhea.

**Solution:** `nerf.def` uses NVIDIA's maintained fork (`nvidia-tensorflow==1.15.5+nv22.12`) compiled for CUDA 11.x, on the same `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04` base as the 3DGS container.

**PTX JIT note:** TF 1.15 has no pre-compiled kernels for sm_89. On the very first run, TF silently JIT-compiles PTX → native sm_89 code (3–5 minutes, no output). Subsequent runs use the cached binary. Do not kill the process if it appears frozen at startup.

---

## Phase 1 — Connect to rhea

Connect via SUPSI VPN or campus ethernet:

```bash
ssh your_username@rhea.idsia.ch
```

**First time only** — redirect Singularity's temp files away from `/tmp`:

```bash
mkdir -p ~/.singularity/tmp
echo 'export SINGULARITY_TMPDIR=${HOME}/.singularity/tmp' >> ~/.bashrc
source ~/.bashrc
```

---

## Phase 2 — Copy code to rhea

```bash
git clone https://github.com/rkalvitis/NERF.git ~/NERF
```

For subsequent updates (push from local first, then on rhea):

```bash
git -C ~/NERF pull
```

---

## Phase 3 — Download datasets on rhea

Store data on fast RAID storage, not in home:

```bash
mkdir -p /media/white/nanodrones/roberts.kalvitis/nerf/nerf_data
mkdir -p /media/white/nanodrones/roberts.kalvitis/nerf/nerf_output
cd /media/white/nanodrones/roberts.kalvitis/nerf/nerf_data
```

Install `gdown` for Google Drive downloads:

```bash
pip install gdown
```

**LLFF real scenes (~3 GB) — Table 1:**

The data is hosted in a shared Google Drive folder. Download the LLFF zip directly by file ID:

```bash
gdown "https://drive.google.com/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g" -O nerf_llff_data.zip
unzip nerf_llff_data.zip
rm nerf_llff_data.zip
```

If `gdown` hits a quota error ("Too many users have viewed or downloaded this file"), wait 24 hours and retry, or use a browser via X11 forwarding (`ssh -X`).

Expected layout:

```
nerf_data/
└── nerf_llff_data/
    ├── fern/
    ├── flower/
    ├── fortress/
    ├── horns/
    ├── leaves/
    ├── orchids/
    ├── room/
    └── trex/
```

Each scene folder contains raw images and `poses_bounds.npy` (pre-computed COLMAP poses). No COLMAP run required.

**Blender synthetic scenes (~4 GB) — Table 2, if needed:**

```bash
gdown "https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG" -O nerf_synthetic.zip
unzip nerf_synthetic.zip
rm nerf_synthetic.zip
```

**DeepVoxels (~6 GB) — Table 3, if needed:**

```bash
gdown "https://drive.google.com/uc?id=1lUvJWB6oFtT8EQ_NzBrXnmi25BufxRfl" -O deepvoxels.zip
unzip deepvoxels.zip
rm deepvoxels.zip
```

---

## Phase 4 — Build the Singularity container on rhea

The container must be built on rhea (not your local machine) so its environment matches rhea's CUDA driver version.

```bash
mkdir -p ~/containers
singularity build --fakeroot ~/containers/nerf.sif ~/NERF/nerf.def
```

Build takes **15–20 minutes**. If it fails on the `nvidia-pyindex` step due to network issues, retry — the NVIDIA package index occasionally times out.

**Verify the build:**

```bash
CUDA_VISIBLE_DEVICES=1 singularity exec --nv ~/containers/nerf.sif /opt/conda/envs/nerf/bin/python -c "
import tensorflow as tf
print('TF:', tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print('GPUs visible:', gpus)
import numpy; print('numpy:', numpy.__version__)
import imageio; print('imageio: OK')
import skimage; print('scikit-image: OK')
import lpips; print('lpips: OK')
"
```

Expected output:

```
TF: 1.15.5
GPUs visible: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
numpy: 1.x.x
imageio: OK
scikit-image: OK
lpips: OK
```

If `GPUs visible: []`, the `--nv` flag is not working — check that you are on rhea (not a login node without GPU access) and that `nvidia-smi` shows a device.

---

## The definition file explained (`nerf.def`)

```singularity
Bootstrap: docker
From: nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
```

Same CUDA 11.8 devel base as the 3DGS container. Provides NVCC and CUDA headers needed by nvidia-tensorflow's bundled extensions.

```singularity
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

TF1 allocates the entire GPU memory on startup by default. This flag makes it grow on demand, preventing OOM if another process briefly holds memory.

```singularity
export TF_DETERMINISTIC_OPS=1
export TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS=1
```

Switches cuDNN to deterministic algorithms. Makes GPU ops reproducible at the cost of ~10–20% slower training. Essential for a seed-controlled reproduction study. Without this, cuDNN reductions use non-deterministic atomics even if all other seeds are fixed.

```singularity
pip install nvidia-pyindex
pip install "nvidia-tensorflow==1.15.5+nv22.12"
```

`nvidia-pyindex` registers NVIDIA's private PyPI index. `nvidia-tensorflow` is NVIDIA's CUDA-11.x-compatible fork of TF 1.15 — the only path to running TF1 on the RTX 4080 without recompiling TensorFlow from source.

```singularity
pip install scikit-image lpips
```

Not in the original `environment.yml`. Required by `compute_metrics.py` to compute SSIM (`skimage.metrics.structural_similarity`) and LPIPS (`lpips.LPIPS(net='alex')`). The paper uses AlexNet LPIPS, not VGG.

---

## Phase 5 — Book the GPU and launch

**1. Check MSTeams** (`rhea-users` channel) for a free GPU.

**2. Confirm it is idle:**

```bash
nvidia-smi -l 1
# Free GPU = 0 MiB memory usage, 0% GPU-Util. Ctrl+C to stop.
```

**3. Claim it on MSTeams:** post `"Using GPU X"`.

**4. Set the GPU in the run script:**

Open `~/NERF/run_experiments.sh` and change the `DEVICE` line at the top to match the index of the GPU you booked:

```bash
DEVICE=1    # <-- your booked GPU
```

**5. Open a screen session** — 40 runs take ~20 days; this keeps them alive if your SSH connection drops:

```bash
screen -S nerf -U
```

Reattach at any time with `screen -r nerf -U`.

**6. Launch inside the screen session:**

```bash
export CODE_DIR=/home/robertsk/NERF
export DATA_DIR=/media/white/nanodrones/roberts.kalvitis/nerf/nerf_data
export OUTPUT_DIR=/media/white/nanodrones/roberts.kalvitis/nerf/nerf_output

mkdir -p "$OUTPUT_DIR"

singularity exec --nv --cleanenv --contain \
    --bind "$CODE_DIR:/workspace" \
    --bind "$DATA_DIR:/data" \
    --bind "$OUTPUT_DIR:/output" \
    ~/containers/nerf.sif \
    bash /workspace/run_experiments.sh
```

| Flag | What it does |
|---|---|
| `--nv` | Exposes NVIDIA drivers to the container. Required for GPU access. |
| `--cleanenv` | Does not import host environment variables. Prevents version conflicts. |
| `--contain` | Does not auto-mount your home directory. You control what the container sees. |
| `--bind src:dest` | Mounts a host path at a path inside the container. |

**7. Detach from screen:**

Press `Ctrl+A`, then `D`.

**Restart support:** if the job is interrupted, re-running the exact same command resumes automatically. `run_experiments.sh` checks for the presence of `testset_200000/` before each run and skips any run that already completed.

---

## Phase 6 — Monitor progress

```bash
# How many of the 40 runs have completed
ls /media/white/nanodrones/roberts.kalvitis/nerf/nerf_output/experiments/*/testset_200000 2>/dev/null | wc -l

# Live log for whichever run is currently executing
tail -f /media/white/nanodrones/roberts.kalvitis/nerf/nerf_output/run_logs/fern_seed0.log

# GPU utilisation
nvidia-smi
```

PSNR is printed every 100 iterations. Example output mid-training:

```
fern_seed0 100 15.23 0.00492 100
fern_seed0 1000 20.14 0.00197 1000
...
fern_seed0 200000 26.84 0.00032 200000
```

Format: `expname  iter  psnr  loss  global_step`. Paper reports ~26 dB for fern at factor=4.

---

## Phase 7 — Collect results

**PSNR** is already logged in each run's `.log` file. To extract the final value:

```bash
grep "fern_seed0" /media/white/nanodrones/roberts.kalvitis/nerf/nerf_output/run_logs/fern_seed0.log | tail -1
```

**SSIM and LPIPS** require running `compute_metrics.py` against the saved test renders. Test images are written to `experiments/<run_name>/testset_200000/` every `i_testset` iterations (default 50 000). The final set at 200 000 iterations is used for reporting.

You still need to create `compute_metrics.py`. It should:
1. Find all `experiments/*/testset_200000/*.png` files
2. Load the matching ground-truth images from `nerf_llff_data/<scene>/images_4/` (factor=4 downsampled)
3. Compute PSNR, SSIM (`skimage.metrics.structural_similarity`, multichannel), LPIPS (`lpips.LPIPS(net='alex')`)
4. Write a `results.json` per run and a summary CSV

Run it inside the container after all experiments finish:

```bash
singularity exec --nv --cleanenv --contain \
    --bind /media/white/nanodrones/roberts.kalvitis/nerf/nerf_output:/output \
    --bind /media/white/nanodrones/roberts.kalvitis/nerf/nerf_data:/data \
    --bind /home/robertsk/NERF:/workspace \
    ~/containers/nerf.sif \
    python /workspace/compute_metrics.py

# Free the GPU when done
nvidia-smi  # confirm no processes running
# Post on MSTeams: "GPU X is free"
```

---

## Metrics — what the paper evaluates

| Metric | Higher = better | Computed by | Note |
|---|---|---|---|
| **PSNR** | yes | `run_nerf.py` inline | Logged every 100 iters; final value at iter 200 000 |
| **SSIM** | yes | `compute_metrics.py` | Not in original code; computed post-training from saved PNGs |
| **LPIPS** | no | `compute_metrics.py` | AlexNet backbone (`net='alex'`) — same as the paper |

---

## Quick reference

| Item | Detail |
|---|---|
| GPU | 1× RTX 4080 16 GB — set `DEVICE=` at top of `run_experiments.sh` |
| Container | Python 3.8, nvidia-tensorflow 1.15.5+nv22.12, CUDA 11.8 |
| Scenes | 8× LLFF: `fern flower fortress horns leaves orchids room trex` |
| Seeds | 5 (0, 1, 2, 3, 4) — outer loop; all 8 scenes complete per seed |
| Total runs | 40, sequential |
| Iters/run | 200 000 (set via `N_iters = 200000` in `paper_configs/llff_config.txt`) |
| Estimated wall time | ~20 days |
| Seed scope | `np.random` + TF global seed + `PYTHONHASHSEED`; GPU cuDNN deterministic via `TF_DETERMINISTIC_OPS=1` |
| Expected PSNR variance | ~0.1–0.3 dB across seeds (residual TF1 eager non-determinism) |
| If SSH drops | `screen -r nerf -U` |
| Restart safety | Script skips runs with `testset_200000/` already present |
| Output per run | `experiments/<scene>_seed<N>/testset_200000/*.png` + `args.txt` |
| PSNR in logs | `run_logs/<scene>_seed<N>.log` — grep for the expname at iter 200000 |
