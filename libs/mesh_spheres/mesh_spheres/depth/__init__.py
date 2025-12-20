"""
Depth map rendering and comparison utilities.
"""

from .depth_comparison import (
    synthesize_depth_from_spheres,
    render_depth_from_mujoco_group,
    compare_depth_maps,
    visualize_depth_maps_interactive
)
from .render_depth_from_pose import (
    add_camera_to_model_xml,
    set_joint_angles,
    quaternion_look_at_origin
)

__all__ = [
    'synthesize_depth_from_spheres',
    'render_depth_from_mujoco_group',
    'compare_depth_maps',
    'visualize_depth_maps_interactive',
    'add_camera_to_model_xml',
    'set_joint_angles',
    'quaternion_look_at_origin',
]

