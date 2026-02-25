import sys
import numpy as np
import pyvista as pv
from collections import deque

# ----------------------------------------
# Configuration
# ----------------------------------------

MESH_PATH = "meshes/bunny.obj"
ANGLE_THRESHOLD_DEG = 30.0  # split into a new region when bend > this angle


# ----------------------------------------
# 1. Load mesh
# ----------------------------------------

mesh = pv.read(MESH_PATH)
mesh = mesh.triangulate()  # ensure all faces are triangles

n_faces = mesh.n_faces_strict

# Extract triangle indices from the faces array
# PyVista stores faces as [n, i0, i1, i2, n, i0, i1, i2, ...]
faces_flat = mesh.faces
triangles = faces_flat.reshape(-1, 4)[:, 1:]  # shape (n_faces, 3)

points = np.array(mesh.points)  # shape (n_points, 3)


# ----------------------------------------
# 2. Compute face normals
# ----------------------------------------

v0 = points[triangles[:, 0]]
v1 = points[triangles[:, 1]]
v2 = points[triangles[:, 2]]

edge1 = v1 - v0
edge2 = v2 - v0
normals = np.cross(edge1, edge2)
norms = np.linalg.norm(normals, axis=1, keepdims=True)
norms = np.where(norms == 0, 1.0, norms)
normals = normals / norms  # unit normals, shape (n_faces, 3)


# ----------------------------------------
# 3. Build face adjacency via shared edges
# ----------------------------------------

# Map each directed edge (i, j) -> face index
edge_to_face: dict[tuple[int, int], int] = {}
face_neighbors: list[list[int]] = [[] for _ in range(n_faces)]

for f_idx in range(n_faces):
    tri = triangles[f_idx]
    for k in range(3):
        i = int(tri[k])
        j = int(tri[(k + 1) % 3])
        # The twin half-edge of (i->j) is (j->i)
        twin = (j, i)
        if twin in edge_to_face:
            nb = edge_to_face[twin]
            face_neighbors[f_idx].append(nb)
            face_neighbors[nb].append(f_idx)
        else:
            edge_to_face[(i, j)] = f_idx


# ----------------------------------------
# 4. Region-growing BFS segmentation
# ----------------------------------------

cos_threshold = np.cos(np.radians(ANGLE_THRESHOLD_DEG))

region_ids = np.full(n_faces, -1, dtype=int)
region_count = 0

for seed in range(n_faces):
    if region_ids[seed] != -1:
        continue
    queue = deque([seed])
    region_ids[seed] = region_count
    while queue:
        f = queue.popleft()
        for nb in face_neighbors[f]:
            if region_ids[nb] != -1:
                continue
            dot = np.dot(normals[f], normals[nb])
            # Same region if the angle between normals is <= threshold
            if dot >= cos_threshold:
                region_ids[nb] = region_count
                queue.append(nb)
    region_count += 1

print(f"Segmentation complete: {region_count} regions found "
      f"(angle threshold = {ANGLE_THRESHOLD_DEG}°)")


# ----------------------------------------
# 5. Color faces by region and display
# ----------------------------------------

# Build a per-face RGB color array so every region gets its own distinct hue.
# The largest (smooth body) region gets a neutral light-gray so that the
# smaller sharp-feature regions pop with vivid distinct colors.
from matplotlib import colormaps

region_sizes = np.bincount(region_ids, minlength=region_count)
rank_by_size = np.argsort(-region_sizes)   # rank_by_size[0] = largest region

rng = np.random.default_rng(seed=7)
# Choose from a high-contrast qualitative palette for the non-dominant regions
cmap_fn = colormaps["hsv"]
small_count = region_count - 1
region_colors = np.zeros((region_count, 3))

# Largest region → neutral light gray
region_colors[rank_by_size[0]] = [0.78, 0.78, 0.78]

# All other regions → evenly-spaced, shuffled hues
hue_positions = rng.permutation(small_count)
for rank, reg_id in enumerate(rank_by_size[1:]):
    region_colors[reg_id] = cmap_fn(hue_positions[rank] / small_count)[:3]

# Assign per-face RGB (uint8) directly so the colormap is truly categorical
face_colors_u8 = (region_colors[region_ids] * 255).astype(np.uint8)
mesh.cell_data["RGB"] = face_colors_u8

# Store the original region IDs too (useful for downstream processing)
mesh.cell_data["region"] = region_ids

plotter = pv.Plotter(off_screen=True, window_size=(1024, 768))
plotter.add_mesh(mesh, scalars="RGB", rgb=True, show_edges=False)

# Overlay sharp boundary edges (dihedral angle > threshold) in black
# to clearly visualise where the region splits occur
sharp_edge_pts: list[float] = []
for (i, j), f in edge_to_face.items():
    twin = (j, i)
    if twin in edge_to_face:
        nb = edge_to_face[twin]
        dot = float(np.dot(normals[f], normals[nb]))
        if dot < cos_threshold:  # sharp edge — region boundary
            sharp_edge_pts.extend([points[i].tolist(), points[j].tolist()])

if sharp_edge_pts:
    edge_pts = np.array(sharp_edge_pts)
    n_segs = len(edge_pts) // 2
    lines = np.column_stack([
        np.full(n_segs, 2),
        np.arange(0, 2 * n_segs, 2),
        np.arange(1, 2 * n_segs, 2),
    ]).flatten()
    edge_mesh = pv.PolyData(edge_pts, lines=lines)
    plotter.add_mesh(edge_mesh, color="black", line_width=1.5,
                     render_lines_as_tubes=False)

plotter.add_text(
    f"Regions: {region_count}  (threshold: {ANGLE_THRESHOLD_DEG}°)",
    font_size=12,
)

# Orient camera to show the bunny upright (sitting pose, viewed from front-right)
plotter.camera_position = [
    (0.25, 0.18, 0.35),   # camera location
    (0.0,  0.12, 0.02),   # focal point (centre of bunny)
    (0.0,  1.0,  0.0),    # view-up vector
]
plotter.show(screenshot="region_split.png")

print("Screenshot saved to region_split.png")
