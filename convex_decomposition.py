"""
Near-Convex Decomposition of the Stanford Bunny
================================================

This script loads the Stanford bunny mesh, splits it into near-convex regions
using CoACD (Approximate Convex Decomposition), then colors and saves each
region with a distinct color to a PNG file using matplotlib (no display
server required).

Prerequisites
-------------
Install the required packages::

    pip install trimesh coacd matplotlib

Usage
-----
Run from the repository root::

    python convex_decomposition.py

The script will save ``decomposition.png`` in the current directory,
showing the bunny decomposed into near-convex parts with distinct colors.

CoACD Parameters (tuneable at top of script)
--------------------------------------------
- ``THRESHOLD``:   Concavity threshold in (0, 1).  Smaller values produce more
  (and smaller) parts; larger values allow more concavity per part.
  Default ``0.2`` gives a reasonable ~5–30 parts for the bunny.
- ``RESOLUTION``:  Voxelization resolution used internally by CoACD.  Higher
  values give a more accurate decomposition at the cost of longer compute time.
  Default ``2000`` (CoACD's built-in default).
- ``MAX_HULLS``:   Upper bound on the number of convex hulls produced
  (-1 means unlimited).
- ``OUTPUT_PNG``:  Path of the output PNG file.
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display server needed

import numpy as np
import trimesh
import coacd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── tuneable parameters ──────────────────────────────────────────────────────
MESH_PATH  = "meshes/bunny.obj"
THRESHOLD  = 0.2     # CoACD concavity threshold (0 < threshold < 1)
RESOLUTION = 2000    # CoACD voxelization resolution (higher = more accurate)
MAX_HULLS  = -1      # max number of convex parts (-1 = unlimited)
OUTPUT_PNG = "decomposition.png"
# ─────────────────────────────────────────────────────────────────────────────


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load an OBJ file and return a single watertight Trimesh."""
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a single mesh from {path!r}")
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def decompose(mesh: trimesh.Trimesh, threshold: float, resolution: int, max_hulls: int):
    """Run CoACD near-convex decomposition and return a list of trimesh parts."""
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        resolution=resolution,
        max_convex_hull=max_hulls,
    )
    print(f"Decomposed into {len(parts)} near-convex parts")
    meshes = []
    for verts, faces in parts:
        meshes.append(trimesh.Trimesh(vertices=verts, faces=faces))
    return meshes


def render_to_png(parts, colours, threshold: float, resolution: int, output_path: str):
    """Render the decomposed parts with distinct colors and save as PNG."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # The bunny mesh uses Y-up; swap Y and Z so the bunny stands upright in
    # matplotlib's Z-up coordinate system.
    for part, colour in zip(parts, colours):
        r, g, b = colour[0] / 255.0, colour[1] / 255.0, colour[2] / 255.0
        tri_verts = part.vertices[part.faces][:, :, [0, 2, 1]]  # swap Y↔Z
        poly = Poly3DCollection(tri_verts, shade=True,
                                facecolors=[(r, g, b)] * len(tri_verts))
        poly.set_edgecolor('none')
        ax.add_collection3d(poly)

    all_verts = np.vstack([p.vertices for p in parts])[:, [0, 2, 1]]
    ax.set_xlim(all_verts[:, 0].min(), all_verts[:, 0].max())
    ax.set_ylim(all_verts[:, 1].min(), all_verts[:, 1].max())
    ax.set_zlim(all_verts[:, 2].min(), all_verts[:, 2].max())
    ax.set_box_aspect(
        [np.ptp(all_verts[:, i]) for i in range(3)]
    )
    ax.view_init(elev=15, azim=-60)  # side view with slight elevation

    ax.set_title(
        f"Stanford Bunny – {len(parts)} near-convex parts\n"
        f"(CoACD, threshold={threshold}, resolution={resolution})",
        fontsize=11,
    )
    ax.set_axis_off()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_path}")


def assign_colors(n: int):
    """Return *n* visually distinct RGBA colours (values in 0–255)."""
    cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
    colours = []
    for i in range(n):
        r, g, b, a = cmap(i / max(n - 1, 1))
        colours.append((int(r * 255), int(g * 255), int(b * 255), 255))
    return colours


def main():
    # 1. Load mesh
    mesh = load_mesh(MESH_PATH)

    # 2. Near-convex decomposition
    parts = decompose(mesh, threshold=THRESHOLD, resolution=RESOLUTION, max_hulls=MAX_HULLS)

    # 3. Render with distinct colors and save to PNG
    colours = assign_colors(len(parts))
    render_to_png(parts, colours, THRESHOLD, RESOLUTION, OUTPUT_PNG)


if __name__ == "__main__":
    main()
