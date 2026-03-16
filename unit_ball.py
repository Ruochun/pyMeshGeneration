import numpy as np
import tetgen
import pyvista as pv

# ----------------------------------------
# Configuration
# ----------------------------------------

RADIUS = 1.0
CENTER = (0.0, 0.0, 0.0)
THETA_RESOLUTION = 32  # subdivisions around the equator
PHI_RESOLUTION = 32    # subdivisions from pole to pole
ELEM_ORDER = 1         # 1 for linear tet4, 2 for quadratic tet10
OUTPUT_FILE = "unit_ball.vtu"

# ----------------------------------------
# 1. Create a unit sphere surface mesh (boundary for TetGen)
# ----------------------------------------

sphere = pv.Sphere(
    radius=RADIUS,
    center=CENTER,
    theta_resolution=THETA_RESOLUTION,
    phi_resolution=PHI_RESOLUTION,
)

# Extract points and triangular faces expected by TetGen
surf_points = np.array(sphere.points)
# pyvista PolyData stores faces as [n, i0, i1, ..., n, i0, ...]; extract triangles
surf_faces = sphere.faces.reshape(-1, 4)[:, 1:]  # drop the leading '3' count

# ----------------------------------------
# 2. Run TetGen to fill the ball with tetrahedra
# ----------------------------------------

tg = tetgen.TetGen(surf_points, surf_faces)

# mindihedral=20, minratio=1.5 give a good-quality mesh
mesh_res = tg.tetrahedralize(order=ELEM_ORDER, mindihedral=20, minratio=1.5)

nodes = mesh_res[0]
elements = mesh_res[1]

# ----------------------------------------
# 3. Convert to PyVista UnstructuredGrid
# ----------------------------------------

# For quadratic elements (order=2) reorder mid-edge nodes to match VTK convention.
# TetGen edge ordering: (0,1), (1,2), (2,0), (3,0), (3,1), (3,2)
# VTK edge ordering:    (0,1), (1,2), (2,0), (0,3), (1,3), (2,3)
if ELEM_ORDER == 2:
    reordered_elements = np.empty_like(elements)
    reordered_elements[:, 0:4] = elements[:, 0:4]  # corner nodes unchanged
    reordered_elements[:, 4] = elements[:, 6]  # VTK edge (0,1) <- TetGen node index 6
    reordered_elements[:, 5] = elements[:, 7]  # VTK edge (1,2) <- TetGen node index 7
    reordered_elements[:, 6] = elements[:, 9]  # VTK edge (2,0) <- TetGen node index 9
    reordered_elements[:, 7] = elements[:, 5]  # VTK edge (0,3) <- TetGen node index 5
    reordered_elements[:, 8] = elements[:, 8]  # VTK edge (1,3) <- TetGen node index 8
    reordered_elements[:, 9] = elements[:, 4]  # VTK edge (2,3) <- TetGen node index 4
    elements = reordered_elements

# VTK cell array format: [n_pts, i0, i1, ..., n_pts, i0, ...]
cells = np.hstack([
    np.full((elements.shape[0], 1), elements.shape[1]),
    elements
]).astype(np.int64)

elem_type = 10 if ELEM_ORDER == 1 else 24  # VTK_TETRA or VTK_QUADRATIC_TETRA
grid = pv.UnstructuredGrid(cells,
                           np.full(elements.shape[0], elem_type),
                           nodes)

# ----------------------------------------
# 4. Save to VTU
# ----------------------------------------

surface = grid.extract_geometry()
surface.save('sphere_highres.obj')

grid.save(OUTPUT_FILE, binary=False)

print(f"Saved {OUTPUT_FILE}")
print(f"  Number of points: {grid.n_points}")
print(f"  Number of cells:  {grid.n_cells}")
