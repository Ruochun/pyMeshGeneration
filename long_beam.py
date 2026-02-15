import numpy as np
import tetgen
import pyvista as pv

# ----------------------------------------
# 1. Define beam geometry (box)
# ----------------------------------------

L = 10.0   # length
W = 1.0    # width
H = 1.0    # height

# 8 corner points of a rectangular beam
points = np.array([
    [0, 0, 0],
    [L, 0, 0],
    [L, W, 0],
    [0, W, 0],
    [0, 0, H],
    [L, 0, H],
    [L, W, H],
    [0, W, H],
])

# Define surface triangles (2 per face × 6 faces = 12 triangles)
faces = np.array([
    [0,1,2], [0,2,3],  # bottom
    [4,5,6], [4,6,7],  # top
    [0,1,5], [0,5,4],  # side
    [1,2,6], [1,6,5],
    [2,3,7], [2,7,6],
    [3,0,4], [3,4,7],
])

# ----------------------------------------
# 2. Run TetGen
# ----------------------------------------

tg = tetgen.TetGen(points, faces)

# Options:
#   pq1.2 -> quality mesh
#   a0.05 -> max tetra volume
#   A     -> assign region attributes
elem_order = 2  # 1 for linear, 2 for quadratic
mesh_res = tg.tetrahedralize(order=elem_order, mindihedral=20, minratio=1.5)
# print(f"mesh_res is {mesh_res}")

# nodes, elements
nodes = mesh_res[0]
elements = mesh_res[1]

# ----------------------------------------
# 3. Convert to PyVista grid
# ----------------------------------------

# For quadratic elements (order=2), reorder mid-edge nodes to match VTK convention
# TetGen and VTK use different node ordering for tet10 elements (type 24)
# TetGen edge ordering: (0,1), (1,2), (2,0), (3,0), (3,1), (3,2)
# VTK edge ordering:    (0,1), (1,2), (2,0), (0,3), (1,3), (2,3)
# Remapping: keep corners [0,1,2,3], reorder mid-edges from [4,5,6,7,8,9] to [4,5,6,7,8,9]
# which means taking nodes at positions [0,1,2,3,6,7,9,5,8,4]
if elem_order == 2:
    # Reorder from TetGen to VTK convention for tet10
    reordered_elements = np.empty_like(elements)
    reordered_elements[:, 0:4] = elements[:, 0:4]  # Keep corner nodes
    reordered_elements[:, 4] = elements[:, 6]  # VTK edge (0,1) <- TetGen position 6
    reordered_elements[:, 5] = elements[:, 7]  # VTK edge (1,2) <- TetGen position 7
    reordered_elements[:, 6] = elements[:, 9]  # VTK edge (2,0) <- TetGen position 9
    reordered_elements[:, 7] = elements[:, 5]  # VTK edge (0,3) <- TetGen position 5
    reordered_elements[:, 8] = elements[:, 8]  # VTK edge (1,3) <- TetGen position 8
    reordered_elements[:, 9] = elements[:, 4]  # VTK edge (2,3) <- TetGen position 4
    elements = reordered_elements

# VTK expects special cell format
cells = np.hstack([
    np.full((elements.shape[0], 1), elements.shape[1]),
    elements
]).astype(np.int64)

elem_type = 10 if elem_order == 1 else 24  # VTK_TETRA or VTK_QUADRATIC_TETRA
grid = pv.UnstructuredGrid(cells, 
                           np.full(elements.shape[0], elem_type),
                           nodes)

# ----------------------------------------
# 4. Save to VTU
# ----------------------------------------

grid.save("beam.vtu", binary=False)

print("Saved beam.vtu")
