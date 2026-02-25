"""
Near-Convex Decomposition of the Stanford Bunny
================================================

This script loads the Stanford bunny mesh, splits it into near-convex regions
using CoACD (Approximate Convex Decomposition), then colors and saves each
region with a distinct color to a PNG file using PyVista.

Prerequisites
-------------
Install the required packages::

    pip install trimesh coacd pyvista matplotlib

Usage
-----
Run from the repository root::

    python bunny_convex_decomposition.py

The script will save ``bunny_decomposition.png`` in the current directory,
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

import numpy as np
import trimesh
import coacd
import pyvista as pv
import matplotlib.pyplot as plt

# ── tuneable parameters ──────────────────────────────────────────────────────
MESH_PATH  = "meshes/bunny.obj"
THRESHOLD  = 0.2     # CoACD concavity threshold (0 < threshold < 1)
RESOLUTION = 2000    # CoACD voxelization resolution (higher = more accurate)
MAX_HULLS  = -1      # max number of convex parts (-1 = unlimited)
OUTPUT_PNG = "bunny_decomposition.png"
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


def build_pyvista_polydata(tm: trimesh.Trimesh) -> pv.PolyData:
    """Convert a trimesh.Trimesh to a pv.PolyData surface mesh."""
    faces_vtk = np.hstack(
        [np.full((len(tm.faces), 1), 3, dtype=np.int_), tm.faces]
    )
    return pv.PolyData(np.asarray(tm.vertices, dtype=float), faces_vtk)


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

    # 3. Build coloured PyVista actors and save to PNG (off-screen)
    colours = assign_colors(len(parts))
    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
    plotter.set_background("white")

    for part, colour in zip(parts, colours):
        poly = build_pyvista_polydata(part)
        hex_colour = "#{:02x}{:02x}{:02x}".format(*colour[:3])
        plotter.add_mesh(
            poly,
            color=hex_colour,
            opacity=1.0,
            smooth_shading=True,
            show_edges=False,
        )

    plotter.add_title(
        f"Stanford Bunny – {len(parts)} near-convex parts "
        f"(CoACD, threshold={THRESHOLD}, resolution={RESOLUTION})",
        font_size=10,
        color="black",
    )
    plotter.camera_position = "xy"
    plotter.screenshot(OUTPUT_PNG)
    plotter.close()
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
