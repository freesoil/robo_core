# Mesh to Spheres Conversion Library

A comprehensive library for converting 3D meshes (STL files) to collections of spheres using octree-based adaptive subdivision. Designed for robotics applications, especially for collision detection and depth map rendering.

## Directory Structure

```
mesh_spheres/
├── mesh_spheres/              # Main package
│   ├── __init__.py           # Package exports
│   ├── converter.py          # Core conversion classes
│   ├── algorithms.py         # Conversion algorithms
│   ├── octree_adaptive.py   # Octree-based adaptive subdivision
│   ├── xml_parser.py         # URDF/MuJoCo XML parsing
│   ├── visualization/        # Visualization utilities
│   │   ├── __init__.py
│   │   ├── visualizer_open3d.py    # Interactive Open3D visualizer
│   │   ├── mujoco_visualizer.py     # MuJoCo XML generation
│   │   └── visualization.py         # Matplotlib visualizations
│   └── depth/                # Depth map rendering
│       ├── __init__.py
│       ├── depth_comparison.py      # Depth map synthesis & comparison
│       └── render_depth_from_pose.py # MuJoCo depth rendering utilities
├── scripts/                  # Executable scripts
│   ├── xml_to_spheres.py     # Batch convert XML meshes to spheres
│   ├── run_xml_parser.py    # Parse XML and show mesh summary
│   ├── demo_complete_workflow.sh    # Complete workflow demo
│   └── run_sphere_depth_comparison.sh # Depth map comparison
├── examples/                 # Example scripts
│   ├── example_render_depth.py      # Depth map rendering example
│   ├── examples.py                  # Various usage examples
│   └── convert_so100_robot.py       # SO-ARM100 robot conversion
├── tests/                    # Test scripts
│   └── test_demo.py          # Test suite
├── docs/                     # Documentation
│   ├── README.md             # This file (moved from root)
│   ├── COMPLETE_GUIDE.md     # Comprehensive user guide
│   ├── DEPTH_RENDERING_FIX.md # Depth rendering documentation
│   └── OCTREE_EXPLANATION.md # Algorithm explanation
├── data/                     # Data files
│   ├── mujoco/               # MuJoCo XML files
│   └── sphere_models/        # Generated sphere collections
└── requirements.txt          # Python dependencies
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from mesh_spheres import MeshToSpheresConverter

# Create converter with default config
converter = MeshToSpheresConverter()

# Convert a mesh
collection = converter.convert('path/to/mesh.stl')

# Save the result
collection.save('output_spheres.npz')

# Load later
from mesh_spheres import SphereCollection
collection = SphereCollection.load('output_spheres.npz')
```

### XML-Based Workflow

```bash
# 1. Parse XML to see available meshes
python scripts/run_xml_parser.py path/to/robot.xml

# 2. Convert all meshes to spheres
python scripts/xml_to_spheres.py path/to/robot.xml \
    --output-dir data/sphere_models \
    --target-coverage 0.95 \
    --target-precision 0.75 \
    --min-radius 0.01

# 3. Visualize
python -m mesh_spheres.visualization.visualizer_open3d \
    path/to/robot.xml \
    --sphere-dir data/sphere_models

# 4. Generate MuJoCo visualization
python -m mesh_spheres.visualization.mujoco_visualizer \
    path/to/robot.xml \
    --sphere-dir data/sphere_models \
    --output data/mujoco/visualization.xml
```

### Complete Workflow Demo

```bash
./scripts/demo_complete_workflow.sh
```

This runs the complete workflow: parsing, conversion, visualization, and MuJoCo XML generation.

## Key Features

- **Octree-based adaptive subdivision**: Automatically subdivides space until coverage and precision criteria are met
- **XML workflow**: Automatically extracts and converts all meshes from URDF/MuJoCo XML files
- **Interactive visualization**: Open3D-based visualizer with keyboard controls
- **MuJoCo integration**: Generate MuJoCo XML files with sphere representations
- **Depth map rendering**: Synthesize depth maps from sphere collections
- **Quality metrics**: Coverage ratio, precision (mesh occupancy), RMS error

## Documentation

- **User Guide**: `docs/USER_GUIDE.md` - Complete guide to using the library, scripts, examples, and testing
- **Technical Guide**: `docs/TECHNICAL_GUIDE.md` - Deep dive into algorithm details, implementation, and technical aspects

## API Reference

### Core Classes

- `MeshToSpheresConverter`: Main conversion class
- `SphereCollection`: Collection of spheres with metadata
- `Sphere`: Single sphere representation

### Visualization

- `MeshSphereVisualizer`: Interactive Open3D visualizer
- `create_mujoco_visualization_xml()`: Generate MuJoCo XML
- `visualize_sphere_collection()`: Matplotlib visualization

### Depth Rendering

- `synthesize_depth_from_spheres()`: Analytical depth synthesis
- `render_depth_from_mujoco_group()`: MuJoCo depth rendering
- `compare_depth_maps()`: Compare different depth maps
- `visualize_depth_maps_interactive()`: Interactive depth comparison

## Examples

See the `examples/` directory for:
- Basic conversion examples
- Collision detection
- 2D projection
- Robot arm conversion
- Depth map rendering

## Tests

Run the test suite:
```bash
python tests/test_demo.py
```

## License

See individual file headers for license information.

