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
mesh_res = tg.tetrahedralize(order=2, mindihedral=20, minratio=1.5)
# print(f"mesh_res is {mesh_res}")

# nodes, elements
nodes = mesh_res[0]
elements = mesh_res[1]

# ----------------------------------------
# 3. Convert to PyVista grid
# ----------------------------------------

# VTK expects special cell format
cells = np.hstack([
    np.full((elements.shape[0], 1), elements.shape[1]),
    elements
]).astype(np.int64)

grid = pv.UnstructuredGrid(cells, 
                           np.full(elements.shape[0], 10),  # 10 = VTK_TETRA
                           nodes)

# ----------------------------------------
# 4. Save to VTU
# ----------------------------------------

grid.save("beam.vtu", binary=False)

print("Saved beam.vtu")
