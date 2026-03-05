import pyvista as pv

# ----------------------------------------
# Configuration
# ----------------------------------------

RADIUS = 1.0
CENTER = (0.0, 0.0, 0.0)
THETA_RESOLUTION = 32  # subdivisions around the equator
PHI_RESOLUTION = 32    # subdivisions from pole to pole
OUTPUT_FILE = "unit_ball.vtu"

# ----------------------------------------
# 1. Create a unit sphere surface mesh
# ----------------------------------------

sphere = pv.Sphere(
    radius=RADIUS,
    center=CENTER,
    theta_resolution=THETA_RESOLUTION,
    phi_resolution=PHI_RESOLUTION,
)

# ----------------------------------------
# 2. Convert to UnstructuredGrid and save to VTU
# ----------------------------------------

# PolyData must be cast to UnstructuredGrid to write a .vtu file
grid = sphere.cast_to_unstructured_grid()

grid.save(OUTPUT_FILE, binary=False)

print(f"Saved {OUTPUT_FILE}")
print(f"  Number of points: {grid.n_points}")
print(f"  Number of cells:  {grid.n_cells}")
