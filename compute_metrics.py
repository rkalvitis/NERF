"""
Compute PSNR / SSIM / LPIPS for all NeRF LLFF runs.

Paths are hardcoded from REPRODUCTION.md.
Run inside the Singularity container:

    singularity exec --nv --cleanenv --contain \
        --bind /media/white/nanodrones/roberts.kalvitis/nerf/nerf_output:/output \
        --bind /media/white/nanodrones/roberts.kalvitis/nerf/nerf_data:/data \
        --bind /home/robertsk/NERF:/workspace \
        --bind /media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache:/torch_cache \
        ~/containers/nerf.sif \
        python /workspace/compute_metrics.py [--seeds 0 1] [--out results.csv]

On first run the AlexNet weights (~232 MB) are downloaded and saved to
  /media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache/hub/checkpoints/
alongside the existing vgg16-397923af.pth.  Subsequent runs reuse them offline.
"""

import argparse
import os
import re
import sys
import csv
from pathlib import Path
from collections import defaultdict

# ── point torch at the shared checkpoint cache BEFORE importing torch/lpips ──
TORCH_CACHE = "/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache"
os.environ["TORCH_HOME"] = TORCH_CACHE

import numpy as np
import imageio

# ── hardcoded paths (from REPRODUCTION.md) ────────────────────────────────────
EXPDIR   = "/output/experiments"
DATADIR  = "/data/nerf_llff_data"
LOGDIR   = "/output/run_logs"

SCENES   = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
FACTOR   = 4
LLFFHOLD = 8

# ── scikit-image SSIM ─────────────────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as _ssim_fn
    def compute_ssim(pred, gt):
        return float(_ssim_fn(pred, gt, channel_axis=-1, data_range=1.0))
except ImportError:
    try:
        from skimage.measure import compare_ssim as _ssim_fn
        def compute_ssim(pred, gt):
            return float(_ssim_fn(pred, gt, multichannel=True, data_range=1.0))
    except ImportError:
        print("WARNING: scikit-image not found — SSIM will be NaN", file=sys.stderr)
        def compute_ssim(pred, gt):
            return float("nan")

# ── LPIPS (AlexNet = paper; weights cached at TORCH_CACHE on first run) ───────
try:
    import torch
    import lpips as lpips_lib

    _lpips_fn = lpips_lib.LPIPS(net="alex")   # downloads alexnet to TORCH_CACHE once
    _lpips_fn.eval()
    LPIPS_NET = "alex"
    print(f"LPIPS: net='alex'  (weights at {TORCH_CACHE}/hub/checkpoints/)")

    def compute_lpips(pred, gt):
        def to_t(img):
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            return t * 2.0 - 1.0          # [0,1] → [-1,1]
        with torch.no_grad():
            return float(_lpips_fn(to_t(pred), to_t(gt)).item())

except Exception as e:
    print(f"WARNING: LPIPS unavailable ({e}) — column will be NaN", file=sys.stderr)
    LPIPS_NET = "na"
    def compute_lpips(pred, gt):
        return float("nan")

# ── metric helpers ────────────────────────────────────────────────────────────

def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    return float("inf") if mse == 0 else float(-10.0 * np.log10(mse))


def load_gt_images(scene):
    imgdir = os.path.join(DATADIR, scene, f"images_{FACTOR}")
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"}
    paths = sorted(p for p in Path(imgdir).iterdir() if p.suffix in exts)
    if not paths:
        raise FileNotFoundError(f"No images in {imgdir}")
    imgs = np.stack([imageio.imread(str(p)) for p in paths], axis=0).astype(np.float32) / 255.0
    if imgs.shape[-1] == 4:
        imgs = imgs[..., :3]
    return imgs[np.arange(len(imgs))[::LLFFHOLD]]


def find_latest_testset(run_dir):
    candidates = sorted(
        d for d in Path(run_dir).iterdir()
        if d.is_dir() and d.name.startswith("testset_")
    )
    return candidates[-1] if candidates else None


def load_renders(testset_dir):
    paths = sorted(Path(testset_dir).glob("*.png"))
    if not paths:
        return None
    imgs = np.stack([imageio.imread(str(p)) for p in paths], axis=0).astype(np.float32) / 255.0
    if imgs.shape[-1] == 4:
        imgs = imgs[..., :3]
    return imgs


def psnr_from_log(scene, seed):
    """Return the last PSNR value printed in the run log, or None."""
    log_path = os.path.join(LOGDIR, f"{scene}_seed{seed}.log")
    if not os.path.isfile(log_path):
        return None
    pattern = re.compile(
        rf"^{re.escape(scene)}_seed{seed}\s+(\d+)\s+([\d.]+)\s+[\d.eE+\-]+\s+\d+"
    )
    last = None
    with open(log_path) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                last = float(m.group(2))
    return last

# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds",  nargs="+", type=int, default=[0, 1])
    p.add_argument("--scenes", nargs="+", default=SCENES)
    p.add_argument("--out",    default="results.csv")
    p.add_argument("--datadir", default=DATADIR,
                   help="GT scene root (e.g. /data/nerf_butterfly)")
    p.add_argument("--factor", type=int, default=FACTOR,
                   help="GT image downsample factor (1 for butterfly scenes)")
    return p.parse_args()


def main():
    global DATADIR, FACTOR
    args = parse_args()
    DATADIR = args.datadir
    FACTOR = args.factor
    rows = []

    hdr = f"{'scene':<12} {'seed':>4} {'iter':>8}  {'PSNR(img)':>9}  {'PSNR(log)':>9}  {'SSIM':>7}  {'LPIPS':>7}"
    print(f"\n{hdr}")
    print("─" * len(hdr))

    for scene in args.scenes:
        try:
            gt_imgs = load_gt_images(scene)
        except Exception as e:
            print(f"  [SKIP] {scene}: {e}", file=sys.stderr)
            continue

        for seed in args.seeds:
            run_name = f"{scene}_seed{seed}"
            run_dir  = os.path.join(EXPDIR, run_name)
            psnr_log = psnr_from_log(scene, seed)
            plog_s   = f"{psnr_log:.3f}" if psnr_log is not None else "—"

            if not os.path.isdir(run_dir):
                print(f"{scene:<12} {seed:>4} {'—':>8}  {'—':>9}  {plog_s:>9}  {'—':>7}  {'—':>7}  (no exp dir)")
                continue

            testset_dir = find_latest_testset(run_dir)
            if testset_dir is None:
                print(f"{scene:<12} {seed:>4} {'—':>8}  {'—':>9}  {plog_s:>9}  {'—':>7}  {'—':>7}  (no testset yet)")
                continue

            iter_num = int(testset_dir.name.split("_")[-1])
            renders  = load_renders(testset_dir)
            if renders is None:
                print(f"  [SKIP] {run_name}: no PNGs in {testset_dir.name}", file=sys.stderr)
                continue

            n = min(len(renders), len(gt_imgs))
            if len(renders) != len(gt_imgs):
                print(f"  [WARN] {run_name}: {len(renders)} renders vs {len(gt_imgs)} GT — using first {n}",
                      file=sys.stderr)

            psnrs, ssims, lpipss = [], [], []
            for pred, gt in zip(renders[:n], gt_imgs[:n]):
                if pred.shape != gt.shape:
                    from skimage.transform import resize as sk_resize
                    pred = sk_resize(pred, gt.shape, anti_aliasing=True).astype(np.float32)
                psnrs.append(compute_psnr(pred, gt))
                ssims.append(compute_ssim(pred, gt))
                lpipss.append(compute_lpips(pred, gt))

            mp = np.mean(psnrs)
            ms = np.mean(ssims)
            ml = np.mean(lpipss)

            print(f"{scene:<12} {seed:>4} {iter_num:>8}  {mp:>9.3f}  {plog_s:>9}  {ms:>7.4f}  {ml:>7.4f}")
            rows.append([scene, seed, iter_num, mp, psnr_log, ms, ml])

    # ── per-scene mean across seeds ───────────────────────────────────────────
    if rows:
        print(f"\n── Mean across seeds {args.seeds} ──")
        print(f"\n{'scene':<12}  {'PSNR':>9}  {'SSIM':>7}  {'LPIPS':>7}")
        print("─" * 42)

        by_scene = defaultdict(list)
        for r in rows:
            by_scene[r[0]].append(r)

        scene_means = []
        for scene in args.scenes:
            if scene not in by_scene:
                continue
            rs = by_scene[scene]
            mp = np.nanmean([r[3] for r in rs])
            ms = np.nanmean([r[5] for r in rs])
            ml = np.nanmean([r[6] for r in rs])
            scene_means.append((mp, ms, ml))
            print(f"{scene:<12}  {mp:>9.3f}  {ms:>7.4f}  {ml:>7.4f}")

        if scene_means:
            print("─" * 42)
            print(f"{'OVERALL':<12}  "
                  f"{np.nanmean([m[0] for m in scene_means]):>9.3f}  "
                  f"{np.nanmean([m[1] for m in scene_means]):>7.4f}  "
                  f"{np.nanmean([m[2] for m in scene_means]):>7.4f}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    if rows and args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "seed", "iter", "psnr_img", "psnr_log", "ssim", f"lpips_{LPIPS_NET}"])
            w.writerows(rows)
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
