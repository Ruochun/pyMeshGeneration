import numpy as np
import tetgen
import pyvista as pv

# Create a simple single tet for testing
points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])

# Single tet face
faces = np.array([
    [0, 2, 1],  # bottom
    [0, 1, 3],  # front 
    [1, 2, 3],  # right
    [2, 0, 3],  # left
])

tg = tetgen.TetGen(points, faces)
mesh_res = tg.tetrahedralize(order=2, mindihedral=20, minratio=1.5)

nodes = mesh_res[0]
elements = mesh_res[1]

elem = elements[0]
print("=== Element Connectivity Analysis ===")
print(f"Element from TetGen: {elem}")
print()

# Get corner nodes in local numbering
local_corners = elem[0:4]
print(f"Corner nodes (local indices 0-3): {local_corners}")
print("Corner positions:")
for i, gid in enumerate(local_corners):
    print(f"  Local corner {i} = global node {gid}: {nodes[gid]}")
print()

# Get mid-edge nodes
mid_edge_nodes = elem[4:10]
print(f"Mid-edge nodes (local indices 4-9): {mid_edge_nodes}")
print()

# For each mid-edge node, find which edge it belongs to
print("Determining edge assignment for each mid-edge node:")
for i, gid in enumerate(mid_edge_nodes):
    local_idx = i + 4
    mid_pos = nodes[gid]
    print(f"\nLocal index {local_idx} (global node {gid}): {mid_pos}")
    
    # Check all possible edges between corners
    for c1 in range(4):
        for c2 in range(c1+1, 4):
            corner1_gid = local_corners[c1]
            corner2_gid = local_corners[c2]
            corner1_pos = nodes[corner1_gid]
            corner2_pos = nodes[corner2_gid]
            expected_mid = (corner1_pos + corner2_pos) / 2
            
            # Check if this mid-edge node is at the expected midpoint
            if np.allclose(mid_pos, expected_mid, atol=1e-6):
                print(f"  -> Edge between local corners {c1} and {c2} (global {corner1_gid}-{corner2_gid})")

# Now show VTK's expected ordering
print("\n\n=== VTK Type 24 Expected Ordering ===")
print("VTK expects edges in this order:")
vtk_edge_pairs = [(0,1), (1,2), (2,0), (0,3), (1,3), (2,3)]
for i, (c1, c2) in enumerate(vtk_edge_pairs):
    local_idx = i + 4
    print(f"  Local index {local_idx}: edge ({c1}, {c2})")
