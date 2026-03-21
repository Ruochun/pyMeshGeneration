import numpy as np
import tetgen
import pyvista as pv

# ----------------------------------------
# Configuration (Schäfer & Turek 3D benchmark dimensions)
# ----------------------------------------

# Box domain: flow travels in the +x direction
Lx = 2.2    # length (streamwise)
Ly = 0.41   # height
Lz = 0.41   # depth

# Cylinder obstacle: axis aligned with z, centered at (0.2, 0.2)
CYL_CENTER_X  = 0.2    # streamwise position
CYL_CENTER_Y  = 0.2    # cross-stream position
CYL_RADIUS    = 0.05   # cylinder radius
CYL_RESOLUTION = 24    # number of triangular segments around the circumference

# Maximum tetrahedron volume (controls mesh density; smaller = finer)
MAX_ELEMENT_VOLUME = 0.001

OUTPUT_FILE = "flow_past_cylinder.vtu"

# Small inset so the cylinder caps do not lie exactly on the box faces,
# which would create a coplanar intersection that TetGen cannot handle.
_CAP_INSET = 1e-3

# ----------------------------------------
# 1. Create outer box surface
# ----------------------------------------

box_tri = pv.Box(bounds=(0.0, Lx, 0.0, Ly, 0.0, Lz)).triangulate()

# ----------------------------------------
# 2. Create closed cylinder surface (the obstacle)
# ----------------------------------------

cyl_tri = pv.Cylinder(
    center=(CYL_CENTER_X, CYL_CENTER_Y, Lz / 2.0),
    direction=(0.0, 0.0, 1.0),
    radius=CYL_RADIUS,
    height=Lz - 2.0 * _CAP_INSET,  # slightly inset to avoid coplanar caps
    resolution=CYL_RESOLUTION,
    capping=True,
).triangulate()

# Flip cylinder face orientation so its normals point inward (into the
# cylinder), consistent with TetGen's convention for inner boundaries.
cyl_tri = cyl_tri.flip_faces()

# ----------------------------------------
# 3. Build combined surface mesh
#    (avoid pyvista's automatic point merging by concatenating manually)
# ----------------------------------------

combined_pts = np.vstack([box_tri.points, cyl_tri.points]).astype(np.float64)

box_faces = box_tri.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
cyl_faces = cyl_tri.faces.reshape(-1, 4)[:, 1:].astype(np.int32) + box_tri.n_points
all_faces = np.vstack([box_faces, cyl_faces])

# ----------------------------------------
# 4. Run TetGen
# ----------------------------------------

tg = tetgen.TetGen(combined_pts, all_faces)

# Mark a point strictly inside the cylinder so TetGen excludes that region.
# load_hole is only accessible via the internal _tetgen attribute because the
# public TetGen Python wrapper does not yet expose a holes argument.
tg._tetgen.load_hole([CYL_CENTER_X, CYL_CENTER_Y, Lz / 2.0])

mesh_res = tg.tetrahedralize(
    order=1,             # linear T4 tetrahedra
    mindihedral=10,
    minratio=1.5,
    fixedvolume=True,
    maxvolume=MAX_ELEMENT_VOLUME,
)

nodes    = mesh_res[0]
elements = mesh_res[1]

# ----------------------------------------
# 5. Convert to PyVista grid and save
# ----------------------------------------

# VTK cell array format: [n_pts_per_cell, i0, i1, i2, i3, ...]
cells = np.hstack([
    np.full((elements.shape[0], 1), 4),
    elements,
]).astype(np.int64)

grid = pv.UnstructuredGrid(
    cells,
    np.full(elements.shape[0], 10),  # VTK_TETRA = 10
    nodes,
)

grid.save(OUTPUT_FILE, binary=False)

print(f"Saved {OUTPUT_FILE}")
print(f"  Number of points:   {grid.n_points}")
print(f"  Number of elements: {grid.n_cells}")
