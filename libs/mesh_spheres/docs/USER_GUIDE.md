# Mesh to Spheres - User Guide

**Complete guide to using the mesh-to-spheres conversion library**

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Usage](#usage)
5. [Scripts and Tools](#scripts-and-tools)
6. [Examples](#examples)
7. [Configuration](#configuration)
8. [Testing](#testing)
9. [MuJoCo Visualization](#mujoco-visualization)
10. [Depth Map Rendering](#depth-map-rendering)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The **Mesh to Spheres** library converts 3D STL mesh files into collections of spheres for efficient collision detection, depth map rendering, and 3D→2D projection operations. It uses an octree-based adaptive algorithm that automatically refines sphere placement until quality criteria (coverage and precision) are met.

### Key Features

- **Criteria-Driven Conversion**: Automatically refines until coverage and precision targets are met
- **Variable Sphere Sizes**: Adapts sphere sizes based on local geometry
- **XML Workflow**: Batch convert all meshes from URDF/MuJoCo XML files
- **Interactive Visualization**: Built-in Open3D visualizer for inspection
- **MuJoCo Integration**: Generate MuJoCo XML files with sphere representations
- **Depth Map Rendering**: Synthesize depth maps from sphere collections
- **Quality Metrics**: Coverage ratio, precision (mesh occupancy), RMS error

### Use Cases

- **Collision Detection**: Fast point-in-sphere and sphere-sphere collision tests
- **Depth Map Synthesis**: Generate depth maps from sphere approximations
- **3D→2D Projection**: Project spheres to image planes for rendering
- **Robot Visualization**: Visualize robot geometry in MuJoCo with sphere approximations
- **Motion Planning**: Use sphere collections for efficient collision checking

---

## Installation

### Prerequisites

- Python 3.8+
- NumPy
- Trimesh
- Open3D (for visualization)
- MuJoCo (for MuJoCo visualization and depth rendering)

### Install Dependencies

```bash
cd /path/to/mesh_spheres
pip install -r requirements.txt
```

### Verify Installation

```python
from mesh_spheres import MeshToSpheresConverter
print("Installation successful!")
```

---

## Quick Start

### Basic Conversion

```python
from mesh_spheres import MeshToSpheresConverter

# Use default configuration
converter = MeshToSpheresConverter()
collection = converter.convert('mesh.stl')

print(f"Generated {len(collection)} spheres")
print(f"Coverage: {collection.metadata['coverage_ratio']:.1%}")
print(f"Precision: {collection.metadata['precision_ratio']:.1%}")

# Save results
collection.save('mesh_spheres.npz')
```

### Command-Line Usage

```bash
# Convert all meshes from XML file
python scripts/xml_to_spheres.py robot.urdf \
    --target-coverage 0.95 \
    --target-precision 0.70 \
    --min-radius 0.003 \
    --output-dir sphere_models
```

---

## Usage

### Python API

#### Basic Conversion

```python
from mesh_spheres import MeshToSpheresConverter

# Default configuration (recommended starting point)
converter = MeshToSpheresConverter()
collection = converter.convert('mesh.stl')

# Custom configuration
config = {
    'coverage_threshold': 0.95,   # 95% surface coverage
    'precision_threshold': 0.70,  # 70% mesh occupancy
    'min_radius_ratio': 0.015,   # 1.5% of mesh size
    'max_radius_ratio': 0.25,    # 25% of mesh size
}
converter = MeshToSpheresConverter(config)
collection = converter.convert('mesh.stl')
```

#### Access Results

```python
# Basic information
print(f"Number of spheres: {len(collection)}")
print(f"Total volume: {collection.total_volume()}")

# Quality metrics
metadata = collection.metadata
print(f"Coverage: {metadata['coverage_ratio']:.1%}")
print(f"Precision: {metadata['precision_ratio']:.1%}")
print(f"Mean error: {metadata['mean_error']:.4f}m")
print(f"RMS error: {metadata['rms_error']:.4f}m")

# Individual spheres
for sphere in collection.spheres:
    print(f"Center: {sphere.center}, Radius: {sphere.radius}")
```

#### Collision Detection

```python
import numpy as np
from mesh_spheres import SphereCollection, Sphere

# Load spheres
collection = SphereCollection.load('mesh_spheres.npz')

# Point collision
point = np.array([0.1, 0.2, 0.3])
is_collision = collection.contains_point(point)
distance = collection.distance_to_point(point)

# Sphere-sphere collision
test_sphere = Sphere(center=np.array([0, 0, 0]), radius=0.05)
collides = any(test_sphere.intersects_sphere(s) for s in collection.spheres)
```

#### Save and Load

```python
# Save
collection.save('output/mesh_spheres.npz')

# Load
from mesh_spheres import SphereCollection
loaded = SphereCollection.load('output/mesh_spheres.npz')
```

#### 3D → 2D Projection

```python
import numpy as np
from mesh_spheres import SphereCollection

# Load spheres
collection = SphereCollection.load('mesh_spheres.npz')

# Define camera matrix (3x4 projection matrix)
camera_matrix = np.array([
    [800, 0, 320, 0],
    [0, 800, 240, 0],
    [0, 0, 1, 0]
])

# Project to 2D
projections = collection.project_to_2d(camera_matrix, (640, 480))

# Draw circles
for center_2d, radius_2d in projections:
    # cv2.circle(image, tuple(center_2d.astype(int)), int(radius_2d), color, -1)
    pass
```

### Command-Line Interface

#### Basic Conversion

```bash
# Default settings
python scripts/xml_to_spheres.py robot.urdf

# Specify quality targets
python scripts/xml_to_spheres.py robot.urdf \
    --target-coverage 0.95 \
    --target-precision 0.70

# Control sphere sizes
python scripts/xml_to_spheres.py robot.urdf \
    --min-radius 0.003 \
    --max-radius 0.10
```

#### Quality Control

```bash
# High quality (tight fit, complete coverage)
python scripts/xml_to_spheres.py robot.urdf \
    --target-coverage 0.98 \
    --target-precision 0.85 \
    --min-radius 0.002

# Fast (lower quality, fewer spheres)
python scripts/xml_to_spheres.py robot.urdf \
    --target-coverage 0.85 \
    --target-precision 0.60 \
    --min-radius 0.005

# Balanced (recommended)
python scripts/xml_to_spheres.py robot.urdf \
    --target-coverage 0.95 \
    --target-precision 0.70 \
    --min-radius 0.003
```

#### With Visualization

```bash
# Convert and visualize
python scripts/xml_to_spheres.py robot.urdf --visualize

# Visualize existing results
python -m mesh_spheres.visualization.visualizer_open3d \
    robot.urdf \
    --sphere-dir sphere_models
```

### XML Workflow (Batch Processing)

```bash
# Step 1: Parse XML to see available meshes
python scripts/run_xml_parser.py robot.urdf

# Step 2: Convert all meshes
python scripts/xml_to_spheres.py robot.urdf \
    --output-dir robot_spheres \
    --target-coverage 0.95 \
    --target-precision 0.70

# Step 3: Visualize results
python -m mesh_spheres.visualization.visualizer_open3d \
    robot.urdf \
    --sphere-dir robot_spheres
```

---

## Scripts and Tools

### Available Scripts

#### 1. `xml_to_spheres.py`

Batch convert all meshes from an XML file (URDF/MuJoCo) to sphere collections.

**Location**: `scripts/xml_to_spheres.py`

**Usage**:
```bash
python scripts/xml_to_spheres.py <xml_file> [OPTIONS]

Options:
  --min-radius FLOAT         Minimum sphere radius in meters
  --max-radius FLOAT         Maximum sphere radius in meters
  --min-radius-ratio FLOAT   Min radius as fraction of mesh size (default: 0.012)
  --max-radius-ratio FLOAT   Max radius as fraction of mesh size (default: 0.30)
  --target-coverage FLOAT    Target coverage ratio 0-1 (default: 0.95)
  --target-precision FLOAT   Target precision ratio 0-1 (default: 0.70)
  --output-dir PATH          Output directory (default: sphere_models)
  --visualize                Launch visualizer after conversion
```

**Example**:
```bash
python scripts/xml_to_spheres.py robot.urdf \
    --target-coverage 0.95 \
    --target-precision 0.70 \
    --min-radius 0.003 \
    --output-dir robot_spheres
```

#### 2. `run_xml_parser.py`

Parse XML file and display summary of found meshes.

**Location**: `scripts/run_xml_parser.py`

**Usage**:
```bash
python scripts/run_xml_parser.py <xml_file>
```

**Output**: Lists all STL meshes found in the XML, their paths, and associated bodies.

#### 3. `demo_complete_workflow.sh`

Complete end-to-end workflow demonstration.

**Location**: `scripts/demo_complete_workflow.sh`

**Usage**:
```bash
./scripts/demo_complete_workflow.sh
```

**What it does**:
1. Parses XML file
2. Converts all meshes to spheres
3. Launches interactive visualizer
4. Generates MuJoCo visualization XML
5. Launches MuJoCo viewer

**Requirements**: MuJoCo must be installed.

#### 4. `run_sphere_depth_comparison.sh`

Run depth map comparison example.

**Location**: `scripts/run_sphere_depth_comparison.sh`

**Usage**:
```bash
./scripts/run_sphere_depth_comparison.sh
```

**What it does**: Renders depth maps from mesh, MuJoCo spheres, and analytically synthesized spheres, then compares them.

### Interactive Visualizer

**Location**: `mesh_spheres/visualization/visualizer_open3d.py`

**Usage**:
```bash
python -m mesh_spheres.visualization.visualizer_open3d \
    <xml_file> \
    --sphere-dir <sphere_directory>
```

**Keyboard Controls**:
- **TAB or T**: Switch between Individual mode and Complete mode
- **LEFT/RIGHT**: Navigate meshes (Individual mode)
- **M**: Toggle mesh visibility
- **S**: Toggle spheres visibility
- **B or 3**: Show both mesh and spheres
- **1**: Show mesh only
- **2**: Show spheres only
- **R**: Reset view
- **H**: Show help
- **Q or ESC**: Quit

---

## Examples

### Example 1: Convert Single Mesh

```python
from mesh_spheres import MeshToSpheresConverter

converter = MeshToSpheresConverter()
collection = converter.convert('link.stl')

print(f"Generated {len(collection)} spheres")
print(f"Coverage: {collection.metadata['coverage_ratio']:.1%}")
print(f"Precision: {collection.metadata['precision_ratio']:.1%}")

# Save results
collection.save('link_spheres.npz')
```

### Example 2: Batch Convert Robot

```python
from mesh_spheres import MeshToSpheresConverter
from pathlib import Path

robot_meshes = Path('isaac_assets/SO_ARM100/meshes')
converter = MeshToSpheresConverter()

for mesh_file in robot_meshes.glob('*.stl'):
    link_name = mesh_file.stem
    collection = converter.convert(str(mesh_file))
    
    print(f"{link_name}: {len(collection)} spheres, "
          f"{collection.metadata['coverage_ratio']:.1%} coverage")
    
    collection.save(f'sphere_models/{link_name}_spheres.npz')
```

### Example 3: Collision Detection

```python
from mesh_spheres import SphereCollection
import numpy as np

# Load spheres
spheres = SphereCollection.load('link_spheres.npz')

# Check collision with point
point = np.array([0.1, 0.2, 0.3])
if spheres.contains_point(point):
    print("Collision detected!")

# Get clearance
distance = spheres.distance_to_point(point)
print(f"Distance to surface: {distance:.4f}m")
```

### Example 4: Interactive Visualization

```bash
# Convert and visualize
python scripts/xml_to_spheres.py robot.urdf --visualize

# Or visualize existing results
python -m mesh_spheres.visualization.visualizer_open3d \
    robot.urdf \
    --sphere-dir sphere_models
```

### Example Scripts

The `examples/` directory contains:

- **`examples.py`**: Various usage examples including collision detection, 2D projection, and method comparison
- **`example_render_depth.py`**: Depth map rendering examples with camera positioning
- **`convert_so100_robot.py`**: Complete SO-ARM100 robot conversion example

---

## Configuration

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_radius_ratio` | float | 0.012 | Min sphere radius as fraction of mesh size (1.2%) |
| `max_radius_ratio` | float | 0.30 | Max sphere radius as fraction of mesh size (30%) |
| `coverage_threshold` | float | 0.95 | Target surface coverage (95%) |
| `precision_threshold` | float | 0.70 | Target mesh occupancy (70%) |
| `min_radius_absolute` | float | None | Override with absolute min radius (meters) |
| `max_radius_absolute` | float | None | Override with absolute max radius (meters) |

### Tuning Guide

#### For High Accuracy

```python
config = {
    'coverage_threshold': 0.98,   # Nearly complete coverage
    'precision_threshold': 0.85,  # Very tight fit
    'min_radius_ratio': 0.010,   # Small spheres for detail
}
# Result: ~150+ spheres, excellent quality, slower
```

#### For Speed

```python
config = {
    'coverage_threshold': 0.85,   # Decent coverage
    'precision_threshold': 0.60,  # Looser fit
    'min_radius_ratio': 0.020,   # Larger minimum spheres
}
# Result: ~30-50 spheres, fast, acceptable quality
```

#### For Balance (Recommended)

```python
config = {
    'coverage_threshold': 0.95,   # Very good coverage
    'precision_threshold': 0.70,  # Good fit
    'min_radius_ratio': 0.015,   # Standard detail
}
# Result: ~60-100 spheres, good quality, reasonable speed
```

### Finding Optimal Configuration

1. **Start with defaults**: The default configuration works well for most objects
2. **Visualize results**: Use the interactive visualizer to inspect quality
3. **Check metrics**: Aim for 90-95% coverage and 65-80% precision
4. **Adjust parameters**:
   - If undercovered: Increase `coverage_threshold` or decrease `min_radius_ratio`
   - If too many spheres: Decrease `coverage_threshold` or increase `min_radius_ratio`
   - For fine details: Decrease `min_radius_ratio`
   - For speed: Increase `min_radius_ratio` and lower thresholds

---

## Testing

### Run Test Suite

```bash
python tests/test_demo.py
```

The test suite includes:
- Basic conversion functionality
- Quality metrics validation
- Collision detection tests
- Save/load functionality
- Edge cases (empty meshes, invalid inputs)

### Manual Testing

```python
from mesh_spheres import MeshToSpheresConverter

# Test basic conversion
converter = MeshToSpheresConverter()
collection = converter.convert('test_mesh.stl')

# Verify quality metrics
assert collection.metadata['coverage_ratio'] >= 0.90
assert collection.metadata['precision_ratio'] >= 0.60
assert len(collection) > 0

print("Tests passed!")
```

---

## MuJoCo Visualization

The MuJoCo visualizer creates a MuJoCo XML file that preserves the original robot structure while adding sphere representations alongside meshes.

### Overview

- **Preserves kinematic structure**: Maintains all joints, bodies, and hierarchy
- **Dual visualization**: Shows original meshes (Group 1) and sphere representations (Group 2)
- **Interactive control**: All joints remain controllable via MuJoCo actuators
- **Toggleable groups**: Use keyboard shortcuts to show/hide different visual groups

### Prerequisites

1. **Convert meshes to spheres first:**
   ```bash
   python scripts/xml_to_spheres.py your_robot.xml --output-dir sphere_models
   ```

2. **Install MuJoCo** (if not already installed):
   ```bash
   pip install mujoco
   ```

### Basic Usage

```bash
python -m mesh_spheres.visualization.mujoco_visualizer \
    <xml_file> \
    --sphere-dir <sphere_directory> \
    [--output <output_file>]
```

**Example**:
```bash
python -m mesh_spheres.visualization.mujoco_visualizer \
    ../../../mujoco_berry_sim/output/so_arm100.xml \
    --sphere-dir data/sphere_models/so100_sphere_models \
    --output data/mujoco/so100_visualization.xml
```

### Visual Groups

The generated XML has three visual groups:

- **Group 0**: Floor (checkerboard pattern)
- **Group 1**: Original meshes (semi-transparent blue)
- **Group 2**: Sphere representations (semi-transparent red)

### Viewing in MuJoCo

#### Option 1: MuJoCo Viewer (Interactive)

```bash
mujoco-viewer so100_visualization.xml
```

**Controls**:
- **0, 1, 2**: Toggle visual groups (floor, meshes, spheres)
- **Mouse**: Rotate camera (drag), zoom (scroll), pan (right-click drag)
- **Arrow keys**: Move camera
- **Space**: Pause/unpause simulation
- **Q or ESC**: Quit

#### Option 2: Python API

```python
import mujoco
import mujoco.viewer

# Load model
model = mujoco.MjModel.from_xml_path('so100_visualization.xml')
data = mujoco.MjData(model)

# Interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

### Complete Workflow

```bash
# Step 1: Convert meshes to spheres
python scripts/xml_to_spheres.py robot.urdf \
    --output-dir robot_spheres \
    --target-coverage 0.95 \
    --target-precision 0.70

# Step 2: Generate MuJoCo visualization
python -m mesh_spheres.visualization.mujoco_visualizer \
    robot.urdf \
    --sphere-dir robot_spheres \
    --output robot_visualization.xml

# Step 3: View in MuJoCo
mujoco-viewer robot_visualization.xml
```

---

## Depth Map Rendering

The library includes tools for rendering depth maps from mesh, MuJoCo spheres, and analytically synthesized spheres.

### Overview

Depth map rendering is useful for:
- Validating robot configurations
- Generating training data for vision-based control
- Testing camera setups
- Visualizing robot poses from different viewpoints

### Basic Usage

```bash
python examples/example_render_depth.py --compare \
    --scene ../../../mujoco_berry_sim/output/so_arm100.xml \
    --sphere-xml data/mujoco/so100_visualization.xml \
    --sphere-dir data/sphere_models/so100_sphere_models \
    --joints 0.0 1.57 1.57 0.0 0.0 \
    --camera-pos 0.5 0.3 0.4 \
    --size 640 480 \
    --output-dir my_comparison
```

### Quick Script

```bash
./scripts/run_sphere_depth_comparison.sh
```

This script runs a complete depth map comparison with pre-configured settings.

### Features

- **Three depth map sources**: Mesh (ground truth), MuJoCo spheres, analytical spheres
- **Interactive comparison**: Matplotlib visualization with side-by-side comparison
- **Camera positioning**: Automatic camera pose extraction from MuJoCo
- **Joint configuration**: Set robot to specified joint angles

---

## Troubleshooting

### Common Issues

#### "URDF file not found"

**Solution**: Check the path in the script. Update `URDF_FILE` variable in `demo_complete_workflow.sh` or use absolute paths.

#### "Sphere directory not found"

**Solution**: Make sure you've run `xml_to_spheres.py` first and specify the correct `--sphere-dir` path.

#### "Mesh file not found"

**Solution**: The script tries to resolve mesh paths relative to the XML file. If meshes are in a different location:
- Copy meshes to the same directory as the XML
- Or update mesh paths in the generated XML manually

#### "No joints found"

**Solution**: The script extracts joints from URDF/MuJoCo XML. If your XML doesn't have explicit joints, you may need to add them manually.

#### MuJoCo viewer doesn't show anything

**Solution**: 
- Press **1** and **2** to toggle mesh and sphere groups
- Check that the camera is positioned correctly (use mouse to rotate/zoom)
- Verify that sphere files were loaded (check console output)

#### Some meshes show in wrong group

**Solution**: The script forces all mesh geoms to Group 1. If you see meshes in other groups, check the console output for mesh matching warnings.

#### Spheres not showing for some parts

**Solution**: Check the console output for mesh name matching. The script tries to match mesh names from the XML to sphere collection files. If names don't match exactly, it will print warnings.

#### Import errors

**Solution**: Make sure you're running scripts from the correct directory and that the `mesh_spheres` package is in your Python path:

```bash
cd /path/to/mesh_spheres
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/xml_to_spheres.py ...
```

#### Low coverage or precision

**Solution**: 
- Decrease `min_radius_ratio` for finer detail
- Increase `coverage_threshold` or `precision_threshold`
- Check mesh quality (watertight, no holes)

---

## API Reference

### Classes

#### `MeshToSpheresConverter`

Main converter class.

```python
converter = MeshToSpheresConverter(config=None)
collection = converter.convert(mesh_path: str) -> SphereCollection
```

#### `SphereCollection`

Collection of spheres with metadata.

**Properties**:
- `spheres: List[Sphere]` - List of spheres
- `metadata: Dict[str, Any]` - Conversion metadata

**Methods**:
- `contains_point(point: np.ndarray) -> bool` - Check point collision
- `distance_to_point(point: np.ndarray) -> float` - Get distance to surface
- `project_to_2d(camera_matrix: np.ndarray, image_size: Tuple) -> List` - Project to 2D
- `bounds() -> Tuple[np.ndarray, np.ndarray]` - Get bounding box
- `total_volume() -> float` - Calculate total volume
- `save(filepath: str)` - Save to file
- `load(filepath: str) -> SphereCollection` - Load from file (classmethod)

#### `Sphere`

Single sphere in 3D space.

**Properties**:
- `center: np.ndarray` - [x, y, z] position
- `radius: float` - Sphere radius

**Methods**:
- `contains_point(point: np.ndarray) -> bool` - Check if point inside
- `distance_to_point(point: np.ndarray) -> float` - Distance to surface
- `intersects_sphere(other: Sphere) -> bool` - Check sphere collision
- `volume() -> float` - Calculate volume

---

## Performance

### Benchmark Results (SO-ARM100 Robot)

| Configuration | Spheres | Coverage | Precision | Time | Quality |
|--------------|---------|----------|-----------|------|---------|
| High Quality | 120-160 | 97-99% | 82-88% | 4-6s | ★★★★★ |
| Balanced | 60-100 | 93-96% | 70-78% | 2-3s | ★★★★☆ |
| Fast | 30-50 | 85-90% | 60-68% | 1-2s | ★★★☆☆ |

---

## Summary

The Mesh to Spheres library provides a comprehensive solution for converting 3D meshes to sphere collections. Key features:

- ✅ Criteria-driven conversion (coverage and precision targets)
- ✅ XML batch processing workflow
- ✅ Interactive visualization tools
- ✅ MuJoCo integration
- ✅ Depth map rendering
- ✅ Comprehensive quality metrics

For technical details about the algorithm, see `TECHNICAL_GUIDE.md`.

