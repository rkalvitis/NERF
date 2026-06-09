#!/usr/bin/env python3
"""Convert a FineView butterfly COLMAP scene (produced by the 3DGS
fineview_pipeline) into the on-disk format NeRF's LLFF *loader* reads:
a flat images/ directory + poses_bounds.npy.

NOTE: "llff" below refers only to the file format / data loader — the data
is the FineView butterfly dataset, not the LLFF forward-facing scenes.

Why this exists
---------------
run_nerf.py's LLFF loader (load_llff.py) assumes ONE camera — a single
(H, W, focal) shared by every image, stacked into one numpy array.
FineView scenes have 8 physical cameras with slightly different focal
lengths and crop sizes (e.g. 2148x1588 .. 1928x1414, fx ~10200..10440).

This script unifies them:

  1. Choose a target focal F so the median camera lands at --width px wide.
  2. Rescale each camera's images by F / fx_cam so all images share focal F.
  3. Center-crop / white-pad every image to one common canvas. Principal
     points are centered in the FineView export and the background is pure
     white, so padding only adds correct background pixels.
  4. Convert COLMAP world-to-camera poses to LLFF's 3x5 camera-to-world
     convention ([down, right, backwards] columns + hwf column).
  5. Compute per-image near/far bounds from the triangulated 3D points
     (points3D.bin or points3D.txt). Falls back to camera-distance
     heuristics if the point cloud is empty (local smoke tests only).

Also creates an `images_1` symlink so configs can use `factor = 1`
without triggering load_llff's ImageMagick minification path.

Usage:
  python fineview2nerf.py \
      --scene_dir /data/009-Neophasia_Menapia-001 \
      --out_dir   /nerf_data/009-Neophasia_Menapia-001 \
      --width 800

Requires: numpy, Pillow.
"""

import argparse
import os
import struct

import numpy as np
from PIL import Image


def read_cameras_txt(path):
    """Returns {camera_id: (W, H, fx, fy, cx, cy)}."""
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tok = line.split()
            cam_id, model = int(tok[0]), tok[1]
            W, H = int(tok[2]), int(tok[3])
            params = [float(x) for x in tok[4:]]
            if model == 'PINHOLE':
                fx, fy, cx, cy = params
            elif model == 'SIMPLE_PINHOLE':
                fx, cx, cy = params
                fy = fx
            else:
                raise ValueError(f'Unsupported camera model {model} '
                                 f'(images must be undistorted)')
            cams[cam_id] = (W, H, fx, fy, cx, cy)
    return cams


def read_images_txt(path):
    """Returns [(name, qvec(4,), tvec(3,), camera_id)], unsorted.

    images.txt alternates image lines and POINTS2D lines; the FineView
    export writes empty POINTS2D lines. Image lines are identified by a
    non-numeric final token (the file name).
    """
    images = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tok = line.split()
            try:
                float(tok[-1])
                continue  # POINTS2D line — all numeric
            except ValueError:
                pass
            qvec = np.array([float(x) for x in tok[1:5]])
            tvec = np.array([float(x) for x in tok[5:8]])
            cam_id = int(tok[8])
            name = tok[9]
            images.append((name, qvec, tvec, cam_id))
    return images


def read_points3d(sparse_dir):
    """Returns (N,3) xyz array from points3D.bin (preferred) or .txt."""
    bin_path = os.path.join(sparse_dir, 'points3D.bin')
    txt_path = os.path.join(sparse_dir, 'points3D.txt')
    if os.path.exists(bin_path):
        pts = []
        with open(bin_path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            for _ in range(n):
                data = struct.unpack('<QdddBBBd', f.read(43))
                pts.append(data[1:4])
                track_len = struct.unpack('<Q', f.read(8))[0]
                f.seek(8 * track_len, 1)
        return np.array(pts).reshape(-1, 3)
    if os.path.exists(txt_path):
        pts = []
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                tok = line.split()
                pts.append([float(tok[1]), float(tok[2]), float(tok[3])])
        return np.array(pts).reshape(-1, 3)
    return np.zeros((0, 3))


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True,
                    help='COLMAP scene with sparse/0/ and images/')
    ap.add_argument('--out_dir', required=True,
                    help='Output dir for images/ + poses_bounds.npy')
    ap.add_argument('--width', type=int, default=800,
                    help='Target image width in px (default 800)')
    args = ap.parse_args()

    sparse = os.path.join(args.scene_dir, 'sparse', '0')
    cams = read_cameras_txt(os.path.join(sparse, 'cameras.txt'))
    images = read_images_txt(os.path.join(sparse, 'images.txt'))
    pts = read_points3d(sparse)
    print(f'{len(cams)} cameras, {len(images)} images, {len(pts)} 3D points')

    # Common focal: median camera, scaled to the target width.
    F_target = float(np.median([fx * args.width / W
                                for (W, H, fx, fy, cx, cy) in cams.values()]))
    scales = {cid: F_target / c[2] for cid, c in cams.items()}
    new_sizes = {cid: (int(round(c[0] * scales[cid])),
                       int(round(c[1] * scales[cid])))
                 for cid, c in cams.items()}
    W_t = args.width
    H_t = int(round(np.median([s[1] for s in new_sizes.values()])))
    print(f'Common camera: {W_t}x{H_t}, focal {F_target:.2f}')

    # Order must match sorted(os.listdir(images/)) of the flattened names,
    # because load_llff pairs pose rows with sorted image files.
    images.sort(key=lambda im: im[0].replace('/', '_'))

    out_imgdir = os.path.join(args.out_dir, 'images')
    os.makedirs(out_imgdir, exist_ok=True)

    rows = []
    for name, qvec, tvec, cam_id in images:
        # --- image: rescale to common focal, center crop/pad to canvas ---
        img = Image.open(os.path.join(args.scene_dir, 'images', name))
        img = img.convert('RGB').resize(new_sizes[cam_id], Image.LANCZOS)
        w, h = img.size
        left, top = max(0, (w - W_t) // 2), max(0, (h - H_t) // 2)
        img = img.crop((left, top, left + min(w, W_t), top + min(h, H_t)))
        canvas = Image.new('RGB', (W_t, H_t), (255, 255, 255))
        canvas.paste(img, ((W_t - img.width) // 2, (H_t - img.height) // 2))
        canvas.save(os.path.join(out_imgdir, name.replace('/', '_')))

        # --- pose: COLMAP w2c -> c2w -> LLFF [down, right, back] ---
        R = qvec2rotmat(qvec)                      # world-to-camera
        c2w = np.concatenate(
            [R.T, (-R.T @ tvec).reshape(3, 1)], axis=1)
        llff = np.concatenate(
            [c2w[:, 1:2], c2w[:, 0:1], -c2w[:, 2:3], c2w[:, 3:4],
             np.array([[H_t], [W_t], [F_target]])], axis=1)

        # --- bounds: depth range of the 3D points in this view ---
        if len(pts) > 0:
            z = (pts @ R.T + tvec)[:, 2]
            z = z[z > 0]
            near, far = np.percentile(z, 0.1), np.percentile(z, 99.9)
        else:
            d = float(np.linalg.norm(-R.T @ tvec))
            near, far = 0.1 * d, 2.0 * d

        rows.append(np.concatenate([llff.ravel(), [near, far]]))

    if len(pts) == 0:
        print('WARNING: empty point cloud — near/far bounds are a '
              'camera-distance heuristic. Use the triangulated '
              'points3D.bin for real runs.')

    poses_bounds = np.stack(rows)
    np.save(os.path.join(args.out_dir, 'poses_bounds.npy'), poses_bounds)

    # factor = 1 support without the ImageMagick minify path
    link = os.path.join(args.out_dir, 'images_1')
    if not os.path.exists(link):
        os.symlink('images', link)

    print(f'Wrote {poses_bounds.shape[0]} poses -> '
          f'{args.out_dir}/poses_bounds.npy')
    print(f'Bounds: near {poses_bounds[:, -2].min():.2f}  '
          f'far {poses_bounds[:, -1].max():.2f}')


if __name__ == '__main__':
    main()
