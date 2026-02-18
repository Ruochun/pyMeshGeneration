#!/usr/bin/env python3
import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import meshio
import matplotlib.pyplot as plt


VTK_EXTS = (".vtu", ".vtk", ".vtp", ".vtr", ".vts", ".vti")


def _natural_key(s: str):
    # Sort like: file2 < file10
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_pvd(pvd_path: str):
    """
    Parse a ParaView .pvd file. Returns (times, filepaths).
    """
    base = os.path.dirname(os.path.abspath(pvd_path))
    tree = ET.parse(pvd_path)
    root = tree.getroot()

    times = []
    files = []
    # Typical structure: <VTKFile><Collection><DataSet timestep="..." file="..."/>
    for ds in root.iter("DataSet"):
        t = ds.attrib.get("timestep", None)
        f = ds.attrib.get("file", None)
        if f is None:
            continue
        times.append(float(t) if t is not None else np.nan)
        files.append(os.path.join(base, f))

    if not files:
        raise RuntimeError(f"No <DataSet ... file='...'> entries found in {pvd_path}")

    return np.array(times, dtype=float), files


def read_mesh_max_displacement(mesh_path: str, var_name="displacement"):
    """
    Read one mesh file and return max(|displacement|) over points/cells.
    Works if displacement is scalar or vector.
    """
    mesh = meshio.read(mesh_path)

    # Try point_data first, then cell_data
    arr = None
    location = None

    if var_name in (mesh.point_data or {}):
        arr = mesh.point_data[var_name]
        location = "point_data"
    elif var_name in (mesh.cell_data or {}):
        # cell_data[var] is a list over cell blocks; concatenate
        blocks = mesh.cell_data[var_name]
        arr = np.concatenate([np.asarray(b) for b in blocks], axis=0)
        location = "cell_data"

    if arr is None:
        available_p = list((mesh.point_data or {}).keys())
        available_c = list((mesh.cell_data or {}).keys())
        raise KeyError(
            f"'{var_name}' not found in {mesh_path}.\n"
            f"Available point_data: {available_p}\n"
            f"Available cell_data:  {available_c}"
        )

    arr = np.asarray(arr)

    # If vector/tensor, use magnitude per entity; if scalar, use abs
    if arr.ndim == 1:
        mags = np.abs(arr)
    else:
        # treat last axis as components
        mags = np.linalg.norm(arr, axis=-1)

    return float(np.max(mags)), location


def collect_files_from_dir(path: str):
    files = []
    for ext in VTK_EXTS:
        files.extend(glob.glob(os.path.join(path, f"*{ext}")))
    files = sorted(set(files), key=_natural_key)
    if not files:
        raise RuntimeError(f"No VTK files found in directory: {path}")
    return files


def main():
    ap = argparse.ArgumentParser(
        description="Plot time series of max(|displacement|) from VTK/VTU time series."
    )
    ap.add_argument("input", help="A .pvd file OR a directory containing .vtu/.vtk/... files")
    ap.add_argument("--var", default="displacement", help="Array name (default: displacement)")
    ap.add_argument("--out", default="", help="Save figure to this path instead of showing")
    ap.add_argument("--title", default="", help="Plot title")
    ap.add_argument("--assume-dt", type=float, default=1.0,
                    help="If input is a directory, use t = index * dt (default dt=1.0)")
    args = ap.parse_args()

    inp = args.input
    var = args.var

    if os.path.isfile(inp) and inp.lower().endswith(".pvd"):
        times, files = read_pvd(inp)
        # If some timesteps are missing, fallback to index
        if np.any(~np.isfinite(times)):
            times = np.arange(len(files), dtype=float)
    elif os.path.isdir(inp):
        files = collect_files_from_dir(inp)
        times = np.arange(len(files), dtype=float) * float(args.assume_dt)
    else:
        raise RuntimeError("Input must be a .pvd file or a directory of VTK files.")

    max_vals = []
    locations = []

    for f in files:
        m, loc = read_mesh_max_displacement(f, var_name=var)
        max_vals.append(m)
        locations.append(loc)

    max_vals = np.array(max_vals, dtype=float)

    # Plot
    plt.figure()
    plt.plot(times, max_vals, marker="o")
    plt.xlabel("Time")
    plt.ylabel(f"max(|{var}|)")
    ttl = args.title if args.title else f"Max |{var}| vs Time"
    plt.title(ttl)
    plt.grid(True)

    # Small annotation: where the data came from
    unique_locs = sorted(set(locations))
    plt.figtext(
        0.01, 0.01,
        f"Source: {', '.join(unique_locs)}",
        ha="left", va="bottom", fontsize=9
    )

    if args.out:
        plt.tight_layout()
        plt.savefig(args.out, dpi=200)
        print(f"Saved plot to: {args.out}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

