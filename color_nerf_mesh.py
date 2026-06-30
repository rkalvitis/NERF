"""Bake per-vertex color onto a NeRF marching-cubes mesh.

extract_butterfly_mesh.py produces geometry only (sigma -> surface). The scene's
color lives in NeRF's view-dependent radiance head, so this script loads an
existing mesh, queries the trained color network at every vertex, and writes a
new PLY with vertex colors that ParaView/MeshLab/Blender show directly.

The mesh vertices are already in the network's coordinate frame (the extraction
maps grid index -> [-extent, extent], which is exactly the network input space),
so we query them as-is. Color is view-dependent; we use the radial view
direction (camera outside, looking toward the object center) as a fixed choice.

Example
-------
python color_nerf_mesh.py \
    --basedir /output/experiments \
    --expname 009-Neophasia_Menapia-001_seed1 \
    --mesh /output/experiments/009-Neophasia_Menapia-001_seed1/009-Neophasia_Menapia-001_seed1_mesh_t50_N512.ply \
    --out  /output/experiments/009-Neophasia_Menapia-001_seed1/009_mesh_t50_N512_colored.ply
"""
import os
import argparse

import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

import run_nerf  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--basedir', required=True)
    p.add_argument('--expname', required=True)
    p.add_argument('--mesh', required=True, help='input .ply (geometry only)')
    p.add_argument('--out', required=True, help='output colored .ply')
    p.add_argument('--center', type=float, nargs=3, default=[0., 0., 0.],
                   help='scene center the cameras look at (default origin)')
    p.add_argument('--offset', type=float, default=0.005,
                   help='push query points this far inside the surface (toward '
                        'center) where density is higher -> truer, less washed '
                        'color. 0 = sample exactly on the iso-surface')
    p.add_argument('--chunk', type=int, default=1024 * 64)
    args = p.parse_args()

    import trimesh
    mesh = trimesh.load(args.mesh, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    print('loaded {} ({} verts, {} faces)'.format(args.mesh, len(verts), len(mesh.faces)))

    config = os.path.join(args.basedir, args.expname, 'config.txt')
    parser = run_nerf.config_parser()
    nerf_args = parser.parse_args(
        '--config {} --basedir {} --expname {}'.format(config, args.basedir, args.expname))

    _, render_kwargs_test, start, _, _ = run_nerf.create_nerf(nerf_args)
    print('loaded checkpoint at iter {}'.format(start))
    net_fn = render_kwargs_test['network_query_fn']
    network_fn = render_kwargs_test['network_fine'] or render_kwargs_test['network_fn']

    # fixed view dir: ray from an outside camera toward the center => points inward
    center = np.asarray(args.center, dtype=np.float32)
    dirs = center[None] - verts
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)

    # sample color slightly inside the surface (toward center) where density is
    # higher -> avoids the washed/background-white color on the fuzzy iso-surface
    query_pts = verts + args.offset * dirs

    rgb = []
    for i in range(0, len(query_pts), args.chunk):
        pts = query_pts[i:i + args.chunk, None, :]
        vd = dirs[i:i + args.chunk]
        raw = net_fn(pts, viewdirs=vd, network_fn=network_fn).numpy()  # (M,1,4)
        rgb.append(1.0 / (1.0 + np.exp(-raw[:, 0, :3])))               # sigmoid -> [0,1]
    rgb = np.concatenate(rgb, 0)
    colors = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    print('vertex color mean RGB = {}'.format(colors.mean(0)))

    mesh.visual.vertex_colors = np.concatenate(
        [colors, np.full((len(colors), 1), 255, np.uint8)], axis=-1)
    mesh.export(args.out)
    print('wrote {}'.format(args.out))


if __name__ == '__main__':
    main()
