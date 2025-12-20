#!/bin/bash
#
# Complete demonstration of XML-based mesh-to-spheres workflow
# for SO-ARM100 robot
#

set -e  # Exit on error

echo "========================================================================"
echo "           MESH TO SPHERES - COMPLETE WORKFLOW DEMO"
echo "========================================================================"
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Try MuJoCo XML first, then URDF
# From scripts/ directory: ../../../../ goes to src/, then mujoco_berry_sim/output/
URDF_FILE="$SCRIPT_DIR/../../../../mujoco_berry_sim/output/so_arm100.xml"
if [ ! -f "$URDF_FILE" ]; then
    URDF_FILE="$SCRIPT_DIR/../../../../isaac_assets/SO_ARM100/so100.urdf"
fi
OUTPUT_DIR="$SCRIPT_DIR/../data/sphere_models/so100_sphere_models"
MUJOCO_XML="$SCRIPT_DIR/../data/mujoco/so100_visualization.xml"

# Check if URDF exists
if [ ! -f "$URDF_FILE" ]; then
    echo "Error: URDF file not found: $URDF_FILE"
    echo "Please update URDF_FILE path in this script."
    exit 1
fi

echo "URDF file: $URDF_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Parse XML
echo "========================================================================"
echo "STEP 1: Parsing URDF file"
echo "========================================================================"
echo ""
python "$SCRIPT_DIR/run_xml_parser.py" "$URDF_FILE"
echo ""
read -p "Press Enter to continue to conversion..."
echo ""

# Step 2: Convert to spheres
echo "========================================================================"
echo "STEP 2: Converting all meshes to sphere collections"
echo "========================================================================"
echo ""
python "$SCRIPT_DIR/xml_to_spheres.py" "$URDF_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --target-coverage 0.95 \
    --target-precision 0.75 \
    --min-radius 0.01
echo ""
read -p "Press Enter to launch visualizer..."
echo ""

# Step 3: Launch visualizer
echo "========================================================================"
echo "STEP 3: Launching interactive visualizer"
echo "========================================================================"
echo ""
echo "Keyboard controls:"
echo "  TAB           : Switch between Individual and Complete mode"
echo "  LEFT/RIGHT    : Navigate meshes (Individual mode)"
echo "  M             : Toggle mesh visibility"
echo "  S             : Toggle spheres visibility"
echo "  B or 3        : Show both"
echo "  1             : Show mesh only"
echo "  2             : Show spheres only"
echo "  R             : Reset view"
echo "  H             : Show help"
echo "  Q or ESC      : Quit"
echo ""
read -p "Press Enter to start visualization..."
echo ""

python "$SCRIPT_DIR/../mesh_spheres/visualization/visualizer_open3d.py" "$URDF_FILE" \
    --sphere-dir "$OUTPUT_DIR"

echo ""
read -p "Press Enter to generate MuJoCo visualization..."
echo ""

# Step 4: Generate MuJoCo XML
echo "========================================================================"
echo "STEP 4: Generating MuJoCo visualization XML"
echo "========================================================================"
echo ""
python "$SCRIPT_DIR/../mesh_spheres/visualization/mujoco_visualizer.py" "$URDF_FILE" \
    --sphere-dir "$OUTPUT_DIR" \
    --output "$MUJOCO_XML"

echo ""
read -p "Press Enter to view in MuJoCo (or Ctrl+C to skip)..."
echo ""

# Step 5: View in MuJoCo
echo "========================================================================"
echo "STEP 5: Launching MuJoCo viewer"
echo "========================================================================"
echo ""
echo "MuJoCo Viewer Controls:"
echo "  0             : Toggle floor (Group 0)"
echo "  1             : Toggle original meshes (Group 1)"
echo "  2             : Toggle sphere representations (Group 2)"
echo "  Mouse         : Rotate camera (drag), zoom (scroll), pan (right-click)"
echo "  Arrow keys    : Move camera"
echo "  Space         : Pause/unpause simulation"
echo "  Q or ESC      : Quit"
echo ""
echo "Note: Make sure MuJoCo is installed: pip install mujoco"
echo ""

# Check if mujoco-viewer is available
if command -v mujoco-viewer &> /dev/null; then
    echo "Launching MuJoCo viewer..."
    mujoco-viewer "$MUJOCO_XML"
elif python -c "import mujoco.viewer" 2>/dev/null; then
    echo "Launching MuJoCo viewer via Python..."
    python -c "
import mujoco
import mujoco.viewer
import sys

model = mujoco.MjModel.from_xml_path('$MUJOCO_XML')
data = mujoco.MjData(model)

print('MuJoCo viewer opened. Close the window to exit.')
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
"
else
    echo "Warning: MuJoCo viewer not found."
    echo "Install MuJoCo with: pip install mujoco"
    echo ""
    echo "You can view the XML later with:"
    echo "  mujoco-viewer $MUJOCO_XML"
    echo ""
    echo "Or using Python:"
    echo "  python -c \"import mujoco; import mujoco.viewer; m=mujoco.MjModel.from_xml_path('$MUJOCO_XML'); d=mujoco.MjData(m); mujoco.viewer.launch_passive(m, d)\""
fi

echo ""
echo "========================================================================"
echo "DEMO COMPLETE!"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  Sphere models: $OUTPUT_DIR"
echo "  MuJoCo XML:    $MUJOCO_XML"
echo ""
echo "To load sphere collections in your code:"
echo ""
echo "  from mesh_spheres import SphereCollection"
echo "  collection = SphereCollection.load('$OUTPUT_DIR/Base_spheres.npz')"
echo ""
echo "To view MuJoCo visualization later:"
echo ""
echo "  mujoco-viewer $MUJOCO_XML"
echo ""
echo "See docs/COMPLETE_GUIDE.md for more information."
echo ""
echo "Note: Directory structure has been reorganized. See ORGANIZATION.md for details."
echo ""

