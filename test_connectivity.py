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

print("Generated nodes:")
print(nodes)
print(f"\nNode shape: {nodes.shape}")
print(f"\nGenerated {elements.shape[0]} element(s)")
print(f"Each element has {elements.shape[1]} nodes")
print("\nFirst element connectivity from TetGen:")
print(elements[0])

# Show which nodes are corners vs mid-edge
print("\n=== Node Analysis for First Element ===")
elem = elements[0]
print(f"Corner nodes (0-3): {elem[0:4]}")
print(f"Mid-edge nodes (4-9): {elem[4:10]}")

# Calculate expected mid-edge positions
print("\n=== Expected Mid-Edge Positions ===")
corners = nodes[elem[0:4]]
print("Corner positions:")
for i, c in enumerate(corners):
    print(f"  Corner {i}: {c}")

print("\nMid-edge nodes in TetGen output:")
for i in range(4, 10):
    node_idx = elem[i]
    print(f"  Node {i} (global {node_idx}): {nodes[node_idx]}")
    
print("\nExpected mid-edge positions (VTK ordering):")
vtk_edges = [(0,1), (1,2), (2,0), (0,3), (1,3), (2,3)]
for i, (n1, n2) in enumerate(vtk_edges):
    midpoint = (corners[n1] + corners[n2]) / 2
    print(f"  Edge {i+4} ({n1}-{n2}): {midpoint}")

print("\nTetGen mid-edge ordering (based on documentation):")
tetgen_edges = [(0,1), (1,2), (2,0), (3,0), (3,1), (3,2)]
for i, (n1, n2) in enumerate(tetgen_edges):
    midpoint = (corners[n1] + corners[n2]) / 2
    print(f"  Edge {i+4} ({n1}-{n2}): {midpoint}")
