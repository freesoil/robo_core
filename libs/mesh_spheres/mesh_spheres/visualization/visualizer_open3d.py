#!/usr/bin/env python3
"""
Interactive Open3D visualizer for meshes and sphere collections.

Keyboard Controls:
- TAB: Switch between Individual mode and Complete mode
- M: Toggle mesh visibility
- S: Toggle spheres visibility
- B: Show both mesh and spheres
- LEFT/RIGHT ARROW: Navigate meshes in Individual mode
- 1/2/3: Switch visualization mode (mesh/spheres/both)
- R: Reset view
- H: Show help
- Q/ESC: Quit
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import trimesh

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d not installed. Install with: pip install open3d")
    sys.exit(1)

import sys
from pathlib import Path

# Handle imports for both script and module usage
script_dir = Path(__file__).parent
libs_dir = script_dir.parent.parent.parent  # robo_core/libs
if str(libs_dir) not in sys.path:
    sys.path.insert(0, str(libs_dir))

try:
    from mesh_spheres.xml_parser import RobotMeshParser
    from mesh_spheres import SphereCollection
except ImportError:
    # Fallback for direct script execution
    from ..xml_parser import RobotMeshParser
    from .. import SphereCollection


class MeshSphereVisualizer:
    """Interactive visualizer for meshes and their sphere approximations."""
    
    def __init__(self, xml_file: str, sphere_dir: str = 'sphere_models'):
        """
        Initialize visualizer.
        
        Args:
            xml_file: Path to URDF or MuJoCo XML file
            sphere_dir: Directory containing sphere collection files
        """
        self.xml_file = Path(xml_file)
        self.sphere_dir = Path(sphere_dir)
        
        # Parse XML to get mesh info
        parser = RobotMeshParser(str(xml_file))
        self.mesh_info = parser.get_mesh_info()
        self.stl_files = parser.get_unique_stl_files()
        
        # Load meshes and spheres
        self.meshes = []
        self.sphere_collections = []
        self.mesh_names = []
        
        print("Loading meshes and sphere collections...")
        for stl_file in self.stl_files:
            mesh_name = Path(stl_file).stem
            
            # Load mesh
            try:
                mesh = trimesh.load(stl_file)
                
                # Only add if mesh is valid
                if len(mesh.vertices) > 0:
                    self.meshes.append(mesh)
                    self.mesh_names.append(mesh_name)
                    
                    # Load corresponding sphere collection
                    sphere_file = self.sphere_dir / f"{mesh_name}_spheres.npz"
                    if sphere_file.exists():
                        try:
                            collection = SphereCollection.load(str(sphere_file))
                            self.sphere_collections.append(collection)
                        except Exception as e:
                            print(f"  Warning: Failed to load sphere collection for {mesh_name}: {e}")
                            self.sphere_collections.append(None)
                    else:
                        print(f"  Warning: No sphere collection for {mesh_name}")
                        self.sphere_collections.append(None)
                else:
                    print(f"  Warning: Empty mesh for {mesh_name}, skipping")
                    
            except Exception as e:
                print(f"  Error loading {mesh_name}: {e}")
        
        # Ensure lists are aligned
        if len(self.meshes) != len(self.sphere_collections):
            print(f"  Warning: Mismatch - {len(self.meshes)} meshes but {len(self.sphere_collections)} sphere collections")
            # Pad with None if needed
            while len(self.sphere_collections) < len(self.meshes):
                self.sphere_collections.append(None)
        
        if len(self.meshes) == 0:
            raise ValueError("No valid meshes loaded! Cannot visualize.")
        
        print(f"Loaded {len(self.meshes)} meshes and {sum(1 for s in self.sphere_collections if s is not None)} sphere collections")
        
        # Visualization state
        self.current_index = 0
        # Ensure current_index is valid
        if self.current_index >= len(self.meshes):
            self.current_index = 0
        self.mode = 'individual'  # 'individual' or 'complete'
        self.show_mesh = True
        self.show_spheres = True
        
        # Open3D visualization
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Mesh & Sphere Visualizer", width=1280, height=720)
        
        # Register keyboard callbacks
        self._register_callbacks()
        
        # Colors
        self.mesh_color = [0.8, 0.8, 0.9]  # Light blue-gray for mesh
        self.sphere_color = [0.3, 0.8, 0.3]  # Green for spheres
        self.sphere_alpha = 0.3  # Transparency
    
    def _register_callbacks(self):
        """Register keyboard callbacks."""
        # TAB (258 in GLFW/Open3D): Switch between individual and complete mode
        # Also try alternative codes in case of different key mapping
        self.vis.register_key_callback(258, self._toggle_mode)  # GLFW_KEY_TAB
        self.vis.register_key_callback(9, self._toggle_mode)    # Alternative
        # T key as alternative for mode toggle
        self.vis.register_key_callback(ord('T'), self._toggle_mode)
        
        # M (77): Toggle mesh visibility
        self.vis.register_key_callback(77, self._toggle_mesh)
        
        # S (83): Toggle spheres visibility
        self.vis.register_key_callback(83, self._toggle_spheres)
        
        # B (66): Show both
        self.vis.register_key_callback(66, self._show_both)
        
        # 1 (49): Mesh only
        self.vis.register_key_callback(49, self._show_mesh_only)
        
        # 2 (50): Spheres only
        self.vis.register_key_callback(50, self._show_spheres_only)
        
        # 3 (51): Both
        self.vis.register_key_callback(51, self._show_both)
        
        # LEFT ARROW (263): Previous mesh
        self.vis.register_key_callback(263, self._previous_mesh)
        
        # RIGHT ARROW (262): Next mesh
        self.vis.register_key_callback(262, self._next_mesh)
        
        # R (82): Reset view
        self.vis.register_key_callback(82, self._reset_view)
        
        # H (72): Show help
        self.vis.register_key_callback(72, self._show_help)
        
        # Q (81) / ESC (256): Quit
        self.vis.register_key_callback(81, self._quit)
        self.vis.register_key_callback(256, self._quit)
    
    def _toggle_mode(self, vis):
        """Toggle between individual and complete mode."""
        self.mode = 'complete' if self.mode == 'individual' else 'individual'
        print(f"\n{'='*60}")
        print(f"Mode: {self.mode.upper()}")
        print(f"{'='*60}")
        self._update_visualization()
        return False
    
    def _toggle_mesh(self, vis):
        """Toggle mesh visibility."""
        self.show_mesh = not self.show_mesh
        print(f"Mesh: {'ON' if self.show_mesh else 'OFF'}")
        self._update_visualization(preserve_view=True)
        return False
    
    def _toggle_spheres(self, vis):
        """Toggle spheres visibility."""
        self.show_spheres = not self.show_spheres
        print(f"Spheres: {'ON' if self.show_spheres else 'OFF'}")
        self._update_visualization(preserve_view=True)
        return False
    
    def _show_both(self, vis):
        """Show both mesh and spheres."""
        self.show_mesh = True
        self.show_spheres = True
        print("Showing: BOTH")
        self._update_visualization(preserve_view=True)
        return False
    
    def _show_mesh_only(self, vis):
        """Show mesh only."""
        self.show_mesh = True
        self.show_spheres = False
        print("Showing: MESH ONLY")
        self._update_visualization(preserve_view=True)
        return False
    
    def _show_spheres_only(self, vis):
        """Show spheres only."""
        self.show_mesh = False
        self.show_spheres = True
        print("Showing: SPHERES ONLY")
        self._update_visualization(preserve_view=True)
        return False
    
    def _previous_mesh(self, vis):
        """Navigate to previous mesh in individual mode."""
        if self.mode == 'individual' and len(self.meshes) > 0:
            self.current_index = (self.current_index - 1) % len(self.meshes)
            if 0 <= self.current_index < len(self.mesh_names):
                print(f"\nMesh {self.current_index + 1}/{len(self.meshes)}: {self.mesh_names[self.current_index]}")
            self._update_visualization()
        return False
    
    def _next_mesh(self, vis):
        """Navigate to next mesh in individual mode."""
        if self.mode == 'individual' and len(self.meshes) > 0:
            self.current_index = (self.current_index + 1) % len(self.meshes)
            if 0 <= self.current_index < len(self.mesh_names):
                print(f"\nMesh {self.current_index + 1}/{len(self.meshes)}: {self.mesh_names[self.current_index]}")
            self._update_visualization()
        return False
    
    def _reset_view(self, vis):
        """Reset camera view."""
        print("Resetting view...")
        vis.reset_view_point(True)
        return False
    
    def _show_help(self, vis):
        """Show help message."""
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS")
        print("="*60)
        print("TAB or T      : Switch between Individual and Complete mode")
        print("LEFT/RIGHT    : Navigate meshes (Individual mode)")
        print("M             : Toggle mesh visibility")
        print("S             : Toggle spheres visibility")
        print("B or 3        : Show both mesh and spheres")
        print("1             : Show mesh only")
        print("2             : Show spheres only")
        print("R             : Reset camera view")
        print("H             : Show this help")
        print("Q or ESC      : Quit")
        print("="*60)
        print(f"\nCurrent mode: {self.mode.upper()}")
        if self.mode == 'individual':
            print(f"Current mesh: {self.current_index + 1}/{len(self.meshes)} - {self.mesh_names[self.current_index]}")
        print(f"Mesh: {'ON' if self.show_mesh else 'OFF'}, Spheres: {'ON' if self.show_spheres else 'OFF'}")
        print("="*60)
        return False
    
    def _quit(self, vis):
        """Quit visualizer."""
        print("\nQuitting...")
        vis.close()
        return True
    
    def _trimesh_to_o3d(self, tmesh: trimesh.Trimesh, color: List[float]) -> o3d.geometry.TriangleMesh:
        """Convert trimesh to Open3D mesh."""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(tmesh.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(tmesh.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh
    
    def _create_sphere_geometry(self, collection: SphereCollection) -> List[o3d.geometry.TriangleMesh]:
        """Create Open3D sphere geometries from sphere collection."""
        spheres = []
        
        for sphere in collection.spheres:
            # Create sphere mesh
            sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(
                radius=sphere.radius,
                resolution=20
            )
            sphere_mesh.translate(sphere.center)
            sphere_mesh.compute_vertex_normals()
            
            # Set color with transparency (simulated via color)
            # Open3D doesn't support true transparency, so we lighten the color
            light_color = [
                self.sphere_color[0] + (1 - self.sphere_color[0]) * 0.5,
                self.sphere_color[1] + (1 - self.sphere_color[1]) * 0.5,
                self.sphere_color[2] + (1 - self.sphere_color[2]) * 0.5,
            ]
            sphere_mesh.paint_uniform_color(light_color)
            
            spheres.append(sphere_mesh)
        
        return spheres
    
    def _update_visualization(self, preserve_view=True):
        """Update visualization based on current state.
        
        Args:
            preserve_view: If True, save and restore camera view to prevent reset
        """
        # Save camera view if preserving
        view_control = None
        camera_params = None
        if preserve_view:
            view_control = self.vis.get_view_control()
            camera_params = view_control.convert_to_pinhole_camera_parameters()
        
        # Clear all geometries
        self.vis.clear_geometries()
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        self.vis.add_geometry(coord_frame)
        
        if self.mode == 'individual':
            # Show only current mesh and its spheres
            if len(self.meshes) == 0:
                return  # Nothing to show
            
            # Ensure current_index is valid
            if self.current_index >= len(self.meshes):
                self.current_index = 0
            if self.current_index < 0:
                self.current_index = len(self.meshes) - 1
            
            mesh = self.meshes[self.current_index]
            collection = None
            if self.current_index < len(self.sphere_collections):
                collection = self.sphere_collections[self.current_index]
            
            # Add mesh
            if self.show_mesh:
                o3d_mesh = self._trimesh_to_o3d(mesh, self.mesh_color)
                self.vis.add_geometry(o3d_mesh)
            
            # Add spheres
            if self.show_spheres and collection is not None:
                sphere_geoms = self._create_sphere_geometry(collection)
                for sphere_geom in sphere_geoms:
                    self.vis.add_geometry(sphere_geom)
        
        else:  # complete mode
            # Show all meshes and spheres
            # Use the shorter list length to avoid index errors
            max_len = min(len(self.meshes), len(self.sphere_collections))
            for i in range(max_len):
                mesh = self.meshes[i]
                collection = self.sphere_collections[i] if i < len(self.sphere_collections) else None
                
                # Add mesh
                if self.show_mesh:
                    o3d_mesh = self._trimesh_to_o3d(mesh, self.mesh_color)
                    self.vis.add_geometry(o3d_mesh)
                
                # Add spheres
                if self.show_spheres and collection is not None:
                    sphere_geoms = self._create_sphere_geometry(collection)
                    for sphere_geom in sphere_geoms:
                        self.vis.add_geometry(sphere_geom)
        
        # Restore camera view if preserving
        if preserve_view and camera_params is not None and view_control is not None:
            view_control.convert_from_pinhole_camera_parameters(camera_params)
        
        # Update rendering
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run(self):
        """Run the interactive visualizer."""
        print("\n" + "="*70)
        print(" "*15 + "INTERACTIVE MESH & SPHERE VISUALIZER")
        print("="*70)
        print(f"\nLoaded {len(self.meshes)} meshes")
        print(f"Mode: {self.mode.upper()}")
        
        if self.mode == 'individual':
            print(f"Current mesh: {self.mesh_names[self.current_index]}")
        
        print("\nPress H for help")
        print("="*70)
        
        # Initial visualization
        self._update_visualization()
        
        # Configure rendering options
        render_option = self.vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.mesh_show_wireframe = False
        render_option.background_color = np.array([0.1, 0.1, 0.15])
        
        # Run visualization loop
        self.vis.run()
        self.vis.destroy_window()


def launch_visualizer(xml_file: str, sphere_dir: str = 'sphere_models'):
    """
    Launch interactive visualizer.
    
    Args:
        xml_file: Path to URDF or MuJoCo XML file
        sphere_dir: Directory containing sphere collections
    """
    try:
        visualizer = MeshSphereVisualizer(xml_file, sphere_dir)
        visualizer.run()
    except Exception as e:
        print(f"Error launching visualizer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive visualizer for meshes and sphere collections'
    )
    parser.add_argument(
        'xml_file',
        help='Path to URDF or MuJoCo XML file'
    )
    parser.add_argument(
        '--sphere-dir',
        default='sphere_models',
        help='Directory containing sphere collections (default: sphere_models)'
    )
    
    args = parser.parse_args()
    
    return launch_visualizer(args.xml_file, args.sphere_dir)


if __name__ == '__main__':
    sys.exit(main())

