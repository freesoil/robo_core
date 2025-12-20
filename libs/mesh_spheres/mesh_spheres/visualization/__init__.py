"""
Visualization utilities for mesh and sphere collections.
"""

from .visualizer_open3d import MeshSphereVisualizer
from .mujoco_visualizer import MuJoCoVisualizer
from .visualization import visualize_sphere_collection, compare_methods

# Alias for backward compatibility
visualize_spheres = visualize_sphere_collection

# Convenience function
def create_mujoco_visualization_xml(xml_file: str, sphere_dir: str, output_file: str):
    """Create MuJoCo visualization XML from robot XML and sphere collections."""
    visualizer = MuJoCoVisualizer(xml_file, sphere_dir)
    visualizer.generate(output_file)

__all__ = [
    'MeshSphereVisualizer',
    'MuJoCoVisualizer',
    'create_mujoco_visualization_xml',
    'visualize_sphere_collection',
    'visualize_spheres',  # Alias
    'compare_methods',
]

