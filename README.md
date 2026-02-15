# pyMeshGeneration

A Python toolkit for generating 3D tetrahedral meshes for engineering applications using TetGen and PyVista.

## Overview

pyMeshGeneration provides Python scripts to create high-quality tetrahedral meshes from geometric definitions. These meshes can be used in finite element analysis (FEA), computational fluid dynamics (CFD), and other numerical simulations.

## Features

- Generate tetrahedral meshes from simple geometric primitives
- Control mesh quality with parameters like dihedral angle and element size
- Export meshes to VTU format for use in ParaView and other visualization tools
- Built on industry-standard libraries: TetGen for meshing and PyVista for 3D data handling

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Ruochun/pyMeshGeneration.git
cd pyMeshGeneration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** The `tetgen` package requires NumPy 1.x for compatibility.

## Usage

### Example: Generate a Beam Mesh

The `long_beam.py` script demonstrates how to create a tetrahedral mesh of a rectangular beam:

```python
python long_beam.py
```

This script:
1. Defines a rectangular beam geometry (10 × 1 × 1 units)
2. Creates surface triangulation
3. Generates high-quality tetrahedral mesh using TetGen
4. Exports the mesh to `beam.vtu`

You can modify the beam dimensions and mesh parameters in the script:
- `L`, `W`, `H`: Length, width, and height of the beam
- `mindihedral`: Minimum dihedral angle (default: 20°)
- `minratio`: Minimum radius-edge ratio (default: 1.5)

## Requirements

- Python 3.6+
- numpy 1.x
- tetgen
- pyvista

See `requirements.txt` for specific version constraints.

## Output Format

Meshes are exported in VTU format (VTK Unstructured Grid), which can be visualized using:
- [ParaView](https://www.paraview.org/)
- [VisIt](https://visit.llnl.gov/)
- PyVista's built-in plotting functions

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Author

Ruochun Zhang
