#!/bin/bash
# Run depth map comparison example

cd "$(dirname "$0")/.."
python examples/example_render_depth.py --compare \
    --scene ../../../mujoco_berry_sim/output/so_arm100.xml \
    --sphere-xml data/mujoco/so100_visualization.xml \
    --sphere-dir data/sphere_models/so100_sphere_models \
    --joints 0.0 1.57 1.57 0.0 0.0 \
    --camera-pos 0.5 0.3 0.4 \
    --size 640 480 \
    --output-dir data/my_comparison
