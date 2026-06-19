"""Extract a triangle mesh from a trained NeRF density field via marching cubes.

NeRF is volumetric, so this samples the learned density (sigma) on a dense 3D
grid and runs marching cubes to produce a surface mesh you can open in any 3D
viewer (MeshLab, Blender, ...) and inspect from all angles.

Only the density field is needed, so near/far bounds are irrelevant here -- the
grid lives in the trained scene's coordinate frame. For the butterfly scenes
that frame is the spherified one: cameras are normalized to radius ~1 looking
inward, so the object sits near the origin well inside a unit cube. The default
--extent 1.0 (grid spanning [-1, 1] per axis) covers it; shrink it to raise
effective resolution once you see where the object lives.

Examples
--------
# one species
python extract_butterfly_mesh.py \
    --basedir /output/experiments \
    --expname 009-Neophasia_Menapia-001_seed1

# all five, several thresholds each
python extract_butterfly_mesh.py \
    --basedir /output/experiments \
    --expname 009-Neophasia_Menapia-001_seed1 072-Colias_Eurytheme-002_seed1 \
              110-Nymphalis_l_album-001_seed1 184-Speyeria_Hydaspe-001_seed1 \
              195-Lycaena_Arota-002_seed1 \
    --threshold 20 50 \
    --N 384
"""
import os
import argparse

import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

import run_nerf  # noqa: E402


def query_density(net_fn, network_fn, N, extent, chunk):
    """Evaluate sigma on an (N+1)^3 grid spanning [-extent, extent] per axis."""
    t = np.linspace(-extent, extent, N + 1)
    grid = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    sh = grid.shape
    flat = grid.reshape([-1, 3])

    raw = []
    for i in range(0, flat.shape[0], chunk):
        pts = flat[i:i + chunk, None, :]
        out = net_fn(pts, viewdirs=np.zeros_like(flat[i:i + chunk]),
                     network_fn=network_fn)
        raw.append(out.numpy())
    raw = np.concatenate(raw, 0).reshape(list(sh[:-1]) + [-1])
    sigma = np.maximum(raw[..., -1], 0.)
    return sigma


def extract_one(expname, args):
    config = os.path.join(args.basedir, expname, 'config.txt')
    if not os.path.isfile(config):
        print('  [skip] no config.txt at {}'.format(config))
        return

    parser = run_nerf.config_parser()
    nerf_args = parser.parse_args(
        '--config {} --basedir {} --expname {}'.format(config, args.basedir, expname))

    # create_nerf auto-reloads the latest checkpoint from basedir/expname.
    _, render_kwargs_test, start, _, _ = run_nerf.create_nerf(nerf_args)
    if start == 0:
        print('  [warn] no checkpoint reloaded for {} (start=0)'.format(expname))
    else:
        print('  loaded checkpoint at iter {}'.format(start))

    net_fn = render_kwargs_test['network_query_fn']
    network_fn = render_kwargs_test['network_fine'] or render_kwargs_test['network_fn']

    sigma = query_density(net_fn, network_fn, args.N, args.extent, args.chunk)
    print('  sigma grid {}  min/median/max = {:.2f}/{:.2f}/{:.2f}'.format(
        sigma.shape, sigma.min(), np.median(sigma), sigma.max()))

    import mcubes
    import trimesh

    out_dir = os.path.join(args.basedir, expname)
    for threshold in args.threshold:
        occ = float(np.mean(sigma > threshold))
        print('  threshold {:g}: fraction occupied {:.4f}'.format(threshold, occ))
        if occ == 0.0:
            print('    [skip] nothing above threshold -- lower it')
            continue
        verts, tris = mcubes.marching_cubes(sigma, threshold)
        # grid index -> world coords in [-extent, extent]
        verts = verts / args.N * (2 * args.extent) - args.extent
        mesh = trimesh.Trimesh(verts, tris)
        out_path = os.path.join(
            out_dir, '{}_mesh_t{:g}_N{}.{}'.format(expname, threshold, args.N, args.format))
        mesh.export(out_path)
        print('    wrote {}  ({} verts, {} faces)'.format(
            out_path, len(verts), len(tris)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--basedir', required=True,
                   help='experiments dir holding <expname>/config.txt + weights')
    p.add_argument('--expname', required=True, nargs='+',
                   help='one or more trained run names')
    p.add_argument('--N', type=int, default=256, help='grid resolution per axis')
    p.add_argument('--extent', type=float, default=1.0,
                   help='grid half-width; spans [-extent, extent] per axis')
    p.add_argument('--threshold', type=float, nargs='+', default=[50.],
                   help='one or more sigma iso-levels to extract')
    p.add_argument('--chunk', type=int, default=1024 * 64,
                   help='query points per forward pass (lower if OOM)')
    p.add_argument('--format', default='ply', choices=['ply', 'obj', 'stl', 'glb'],
                   help='output mesh format')
    args = p.parse_args()

    for expname in args.expname:
        print('=' * 56)
        print('  {}'.format(expname))
        print('=' * 56)
        extract_one(expname, args)


if __name__ == '__main__':
    main()
