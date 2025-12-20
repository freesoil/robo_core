#!/usr/bin/env python3
"""
Example script demonstrating how to use render_depth_from_pose.py

This shows how to render depth maps for different joint configurations and camera poses,
and compare mesh-based, MuJoCo sphere-based, and analytically synthesized sphere-based depth maps.
"""

import subprocess
import sys
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation
import mujoco
from mujoco import Renderer

# Import comparison utilities
try:
    from mesh_spheres.depth import (
        synthesize_depth_from_spheres,
        render_depth_from_mujoco_group,
        compare_depth_maps
    )
    from mesh_spheres import SphereCollection
except ImportError:
    # If running as script, add libs directory to path
    script_dir = Path(__file__).parent
    libs_dir = script_dir.parent.parent  # robo_core/libs
    if str(libs_dir) not in sys.path:
        sys.path.insert(0, str(libs_dir))
    try:
        from mesh_spheres.depth import (
            synthesize_depth_from_spheres,
            render_depth_from_mujoco_group,
            compare_depth_maps
        )
        from mesh_spheres import SphereCollection
    except ImportError:
        # Fallback: import directly from module path
        import importlib.util
        depth_comp_path = Path(__file__).parent.parent / "mesh_spheres" / "depth" / "depth_comparison.py"
        spec = importlib.util.spec_from_file_location("depth_comparison", depth_comp_path)
        depth_comp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(depth_comp_module)
        synthesize_depth_from_spheres = depth_comp_module.synthesize_depth_from_spheres
        render_depth_from_mujoco_group = depth_comp_module.render_depth_from_mujoco_group
        compare_depth_maps = depth_comp_module.compare_depth_maps
        
        from mesh_spheres import SphereCollection

# Find the scene XML file
workspace_root = Path(__file__).parent.parent.parent.parent.parent
scene_xml = workspace_root / "src/mujoco_berry_sim/output/scene.xml"
sphere_xml = Path(__file__).parent.parent / "data" / "mujoco" / "so100_visualization.xml"
sphere_dir = Path(__file__).parent.parent / "data" / "sphere_models" / "so100_sphere_models"


def quaternion_look_at_origin(camera_pos, up_direction=None):
    """
    Compute quaternion for camera at camera_pos looking at origin (0, 0, 0).
    
    Args:
        camera_pos: Camera position [x, y, z]
        up_direction: Up direction vector [x, y, z] (default: [0, 0, 1] for world Z-up)
    
    Returns:
        Quaternion [w, x, y, z] (MuJoCo convention)
    """
    if up_direction is None:
        up_direction = np.array([0.0, 0.0, 1.0])
    
    # Camera forward direction (negative Z in camera frame) points toward origin
    forward = -np.array(camera_pos)  # Vector from camera to origin
    forward_norm = np.linalg.norm(forward)
    
    if forward_norm < 1e-6:
        # Camera is at origin, use identity rotation
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    forward = forward / forward_norm  # Normalize
    
    # Compute right vector (camera X axis)
    right = np.cross(forward, up_direction)
    right_norm = np.linalg.norm(right)
    
    if right_norm < 1e-6:
        # Forward and up are parallel, use default right
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm
    
    # Compute up vector (camera Y axis) - reorthogonalize
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Build rotation matrix: columns are camera X, Y, -Z in world coordinates
    # MuJoCo camera: X (right), Y (up), Z (forward, but -Z is forward direction)
    R = np.column_stack([right, up, -forward])
    
    # Convert to quaternion
    rot = Rotation.from_matrix(R)
    quat_xyzw = rot.as_quat()  # [x, y, z, w]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]
    
    return quat_wxyz


def load_sphere_collections(sphere_dir: Path) -> dict:
    """Load all sphere collections from the directory."""
    collections = {}
    
    if not sphere_dir.exists():
        print(f"Warning: Sphere directory not found: {sphere_dir}")
        return collections
    
    for npz_file in sphere_dir.glob("*_spheres.npz"):
        if "_metadata" in str(npz_file):
            continue
        
        body_name = npz_file.stem.replace("_spheres", "")
        try:
            collection = SphereCollection.load(str(npz_file))
            collections[body_name] = collection
            print(f"  Loaded {len(collection)} spheres for '{body_name}'")
        except Exception as e:
            print(f"  Warning: Could not load {npz_file}: {e}")
    
    return collections


def render_and_compare(
    scene_xml: Path,
    sphere_xml: Path,
    sphere_dir: Path,
    joint_angles: list,
    joint_names: list,
    camera_pos: list,
    camera_quat: list,
    camera_name: str,
    output_dir: Path,
    image_size: tuple = (640, 480),
    focal_length: float = 800.0
):
    """
    Render depth maps from mesh, MuJoCo spheres, and analytical spheres, then compare.
    """
    # Import functions from render_depth_from_pose module
    import importlib.util
    render_script_path = Path(__file__).parent.parent / "mesh_spheres" / "depth" / "render_depth_from_pose.py"
    spec = importlib.util.spec_from_file_location("render_depth_from_pose", render_script_path)
    render_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render_module)
    add_camera_to_model_xml = render_module.add_camera_to_model_xml
    set_joint_angles = render_module.set_joint_angles
    
    # Import interactive visualization
    try:
        from mesh_spheres.depth import visualize_depth_maps_interactive
    except ImportError:
        depth_comp_path = Path(__file__).parent.parent / "mesh_spheres" / "depth" / "depth_comparison.py"
        spec = importlib.util.spec_from_file_location("depth_comparison", depth_comp_path)
        depth_comp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(depth_comp_module)
        visualize_depth_maps_interactive = depth_comp_module.visualize_depth_maps_interactive
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Rendering and Comparing Depth Maps")
    print(f"{'='*70}")
    print(f"Joint angles: {joint_angles}")
    print(f"Camera position: {camera_pos}")
    print(f"Camera quaternion: {camera_quat}")
    print()
    
    # Step 1: Add camera to both XMLs
    temp_scene_xml = output_dir / "temp_scene_with_camera.xml"
    temp_sphere_xml = output_dir / "temp_sphere_with_camera.xml"
    
    print("Adding camera to scene XMLs...")
    add_camera_to_model_xml(
        scene_xml, camera_name, camera_pos, camera_quat,
        temp_scene_xml, framebuffer_size=image_size
    )
    
    if sphere_xml.exists():
        add_camera_to_model_xml(
            sphere_xml, camera_name, camera_pos, camera_quat,
            temp_sphere_xml, framebuffer_size=image_size
        )
    else:
        print(f"Warning: Sphere XML not found: {sphere_xml}")
        print("  Skipping sphere-based rendering. Run demo_complete_workflow.sh first.")
        temp_sphere_xml = None
    
    # Step 2: Load models and set joint angles
    print("Loading models...")
    model_mesh = mujoco.MjModel.from_xml_path(str(temp_scene_xml))
    data_mesh = mujoco.MjData(model_mesh)
    set_joint_angles(model_mesh, data_mesh, joint_names, joint_angles)
    
    if temp_sphere_xml and temp_sphere_xml.exists():
        model_sphere = mujoco.MjModel.from_xml_path(str(temp_sphere_xml))
        data_sphere = mujoco.MjData(model_sphere)
        set_joint_angles(model_sphere, data_sphere, joint_names, joint_angles)
        
        # Debug: Count geoms by group
        print(f"  Sphere model loaded: {model_sphere.ngeom} total geoms")
        for group_id in range(6):  # MuJoCo supports up to 6 groups
            count = sum(1 for i in range(model_sphere.ngeom) 
                       if model_sphere.geom_group[i] == group_id)
            if count > 0:
                print(f"    Group {group_id}: {count} geoms")
    else:
        model_sphere = None
        data_sphere = None
    
    # Step 3: Render depth from mesh (using original scene)
    print("Rendering depth from mesh...")
    renderer_mesh = Renderer(model_mesh, height=image_size[0], width=image_size[1])
    renderer_mesh.enable_depth_rendering()
    renderer_mesh.update_scene(data_mesh, camera=camera_name)
    depth_mesh = renderer_mesh.render()
    renderer_mesh.disable_depth_rendering()
    print(f"  Mesh depth range: [{depth_mesh.min():.3f}, {depth_mesh.max():.3f}] m")
    
    # Step 4: Render depth from MuJoCo spheres (group 2)
    depth_spheres_mujoco = None
    if model_sphere is not None:
        print("Rendering depth from MuJoCo spheres (group 2)...")
        # Verify camera exists in sphere model
        camera_id = mujoco.mj_name2id(model_sphere, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            print(f"  Error: Camera '{camera_name}' not found in sphere model")
            print(f"  Available cameras: {[mujoco.mj_id2name(model_sphere, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model_sphere.ncam)]}")
        else:
            try:
                depth_spheres_mujoco = render_depth_from_mujoco_group(
                    model_sphere, data_sphere, None, camera_name, geom_group=2, size=image_size
                )
                print(f"  MuJoCo spheres depth range: [{depth_spheres_mujoco.min():.3f}, {depth_spheres_mujoco.max():.3f}] m")
            except Exception as e:
                print(f"  Error rendering MuJoCo spheres: {e}")
                import traceback
                traceback.print_exc()
                depth_spheres_mujoco = None
    else:
        print("  Skipping MuJoCo sphere rendering (sphere model not loaded)")
    
    # Step 5: Synthesize depth analytically from spheres
    depth_spheres_analytic = None
    if sphere_dir.exists():
        print("Synthesizing depth analytically from spheres...")
        sphere_collections = load_sphere_collections(sphere_dir)
        
        if sphere_collections:
            # Debug: Print all body names from model for matching
            if model_mesh is not None:
                print("  Available MuJoCo body names:")
                body_names = []
                for i in range(model_mesh.nbody):
                    name = mujoco.mj_id2name(model_mesh, mujoco.mjtObj.mjOBJ_BODY, i)
                    if name:
                        body_names.append(name)
                print(f"    {body_names[:15]}...")  # Show first 15
                print(f"  Sphere collection names:")
                print(f"    {list(sphere_collections.keys())}")
            
            try:
                # Extract camera pose relative to robot base from MuJoCo
                # This ensures the camera-robot relationship matches exactly
                camera_id = mujoco.mj_name2id(model_mesh, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                if camera_id >= 0:
                    # Get camera pose in world coordinates
                    cam_pos_model = model_mesh.cam_pos[camera_id].copy()
                    cam_quat_model = model_mesh.cam_quat[camera_id].copy()  # [w, x, y, z]
                    
                    # If camera is attached to a body, transform to world coordinates
                    cam_bodyid = model_mesh.cam_bodyid[camera_id]
                    if cam_bodyid >= 0:
                        # Camera is attached to a body - transform to world coordinates
                        body_pos = data_mesh.xpos[cam_bodyid]
                        body_quat = data_mesh.xquat[cam_bodyid]  # [w, x, y, z]
                        
                        # Convert body quaternion to rotation matrix
                        from scipy.spatial.transform import Rotation
                        body_quat_xyzw = np.array([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
                        body_rot = Rotation.from_quat(body_quat_xyzw)
                        body_R = body_rot.as_matrix()
                        
                        # Transform camera position to world
                        cam_pos_world = body_pos + body_R @ cam_pos_model
                        
                        # Transform camera quaternion (compose rotations)
                        cam_quat_xyzw = np.array([cam_quat_model[1], cam_quat_model[2], 
                                                 cam_quat_model[3], cam_quat_model[0]])
                        cam_rot = Rotation.from_quat(cam_quat_xyzw)
                        world_rot = Rotation.from_matrix(body_R) * cam_rot
                        cam_quat_xyzw_world = world_rot.as_quat()  # [x, y, z, w]
                        cam_quat_world = np.array([cam_quat_xyzw_world[3], cam_quat_xyzw_world[0],
                                                   cam_quat_xyzw_world[1], cam_quat_xyzw_world[2]])  # [w, x, y, z]
                    else:
                        # Fixed camera in worldbody - world pose = local pose
                        cam_pos_world = cam_pos_model
                        cam_quat_world = cam_quat_model
                    
                    # Find robot base body (first body after 'world', or 'world' itself)
                    base_body_id = -1
                    base_body_name = None
                    for i in range(model_mesh.nbody):
                        body_name = mujoco.mj_id2name(model_mesh, mujoco.mjtObj.mjOBJ_BODY, i)
                        if body_name and body_name.lower() != 'world':
                            # Use first non-world body as base (typically the robot base)
                            base_body_id = i
                            base_body_name = body_name
                            break
                    
                    if base_body_id >= 0:
                        # Get base pose in world coordinates
                        base_pos_world = data_mesh.xpos[base_body_id]
                        base_quat_world = data_mesh.xquat[base_body_id]  # [w, x, y, z]
                        
                        # Convert base quaternion to rotation matrix
                        base_quat_xyzw = np.array([base_quat_world[1], base_quat_world[2], 
                                                  base_quat_world[3], base_quat_world[0]])
                        base_rot = Rotation.from_quat(base_quat_xyzw)
                        base_R = base_rot.as_matrix()
                        
                        # Compute camera pose relative to base
                        # cam_in_base = base_R^T @ (cam_world - base_world)
                        cam_pos_rel = base_R.T @ (cam_pos_world - base_pos_world)
                        
                        # Compute relative rotation: cam_rot_rel = base_R^T @ cam_R
                        cam_quat_xyzw_world_arr = np.array([cam_quat_world[1], cam_quat_world[2], 
                                                             cam_quat_world[3], cam_quat_world[0]])
                        cam_rot_world = Rotation.from_quat(cam_quat_xyzw_world_arr)
                        cam_rot_rel = Rotation.from_matrix(base_R.T) * cam_rot_world
                        cam_quat_xyzw_rel = cam_rot_rel.as_quat()  # [x, y, z, w]
                        cam_quat_rel = np.array([cam_quat_xyzw_rel[3], cam_quat_xyzw_rel[0],
                                                 cam_quat_xyzw_rel[1], cam_quat_xyzw_rel[2]])  # [w, x, y, z]
                        
                        print(f"  Extracted camera pose relative to base '{base_body_name}':")
                        print(f"    Relative position: {cam_pos_rel}")
                        print(f"    Relative quaternion: {cam_quat_rel}")
                        print(f"    Camera world position: {cam_pos_world}")
                        print(f"    Base world position: {base_pos_world}")
                        
                        # For analytical synthesis, we'll use the relative pose
                        # by computing camera pose from base pose + relative pose
                        # This ensures exact match with MuJoCo's coordinate system
                        camera_pos_use = cam_pos_world  # Use world position (same as MuJoCo)
                        camera_quat_use = cam_quat_world  # Use world quaternion (same as MuJoCo)
                    else:
                        # No base found, use camera world pose directly
                        print(f"  Using camera world pose (no base body found)")
                        camera_pos_use = cam_pos_world
                        camera_quat_use = cam_quat_world
                    
                    print(f"  Final camera pose for analytical synthesis:")
                    print(f"    Position: {camera_pos_use}")
                    print(f"    Quaternion: {camera_quat_use}")
                else:
                    print(f"  Warning: Camera '{camera_name}' not found, using provided pose")
                    camera_pos_use = np.array(camera_pos)
                    camera_quat_use = np.array(camera_quat)
                
                # Use the mesh model and data for body transformations
                # Sphere centers are in body-local coordinates, so we need to transform them
                # to world coordinates based on the current joint configuration
                depth_spheres_analytic = synthesize_depth_from_spheres(
                    sphere_collections,
                    camera_pos_use,
                    camera_quat_use,
                    image_size,
                    focal_length=focal_length,
                    model=model_mesh,
                    data=data_mesh
                )
                print(f"  Analytical spheres depth range: [{depth_spheres_analytic.min():.3f}, {depth_spheres_analytic.max():.3f}] m")
            except Exception as e:
                print(f"  Error synthesizing analytical depth: {e}")
                import traceback
                traceback.print_exc()
                depth_spheres_analytic = None
        else:
            print("  No sphere collections found")
    else:
        print(f"  Sphere directory not found: {sphere_dir}")
    
    # Step 6: Compare and visualize
    if depth_spheres_mujoco is not None or depth_spheres_analytic is not None:
        print("\nComparing depth maps...")
        
        # Use mesh depth as reference
        depth_ref = depth_mesh
        
        # Create comparison with available data
        if depth_spheres_mujoco is not None and depth_spheres_analytic is not None:
            metrics = compare_depth_maps(
                depth_ref, depth_spheres_mujoco, depth_spheres_analytic,
                output_dir, prefix="depth_comparison"
            )
        elif depth_spheres_mujoco is not None:
            # Only MuJoCo spheres available - create dummy analytic depth
            depth_dummy = np.full_like(depth_ref, 2.0)
            metrics = compare_depth_maps(
                depth_ref, depth_spheres_mujoco, depth_dummy,
                output_dir, prefix="depth_comparison"
            )
        elif depth_spheres_analytic is not None:
            # Only analytical spheres available - create dummy MuJoCo spheres depth
            depth_dummy = np.full_like(depth_ref, 2.0)
            metrics = compare_depth_maps(
                depth_ref, depth_dummy, depth_spheres_analytic,
                output_dir, prefix="depth_comparison"
            )
        
        # Save individual depth maps
        import matplotlib.pyplot as plt
        
        plt.imsave(output_dir / "depth_mesh.png", 
                  np.clip(depth_mesh, 0, 2.0) / 2.0, cmap='viridis')
        
        if depth_spheres_mujoco is not None:
            plt.imsave(output_dir / "depth_spheres_mujoco.png",
                      np.clip(depth_spheres_mujoco, 0, 2.0) / 2.0, cmap='viridis')
        
        if depth_spheres_analytic is not None:
            plt.imsave(output_dir / "depth_spheres_analytic.png",
                      np.clip(depth_spheres_analytic, 0, 2.0) / 2.0, cmap='viridis')
        
        print(f"\nDepth maps saved to: {output_dir}")
        
        # Show interactive visualization
        print("\nOpening interactive visualization...")
        try:
            visualize_depth_maps_interactive(
                depth_mesh,
                depth_spheres_mujoco,
                depth_spheres_analytic,
                title=f"Depth Comparison - Joints: {joint_angles}"
            )
        except Exception as e:
            print(f"  Error showing interactive visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo sphere-based depth maps available for comparison")
        print("  Make sure to run demo_complete_workflow.sh first to generate sphere models")
    
    # Cleanup
    if temp_scene_xml.exists():
        temp_scene_xml.unlink()
    if temp_sphere_xml and temp_sphere_xml.exists():
        temp_sphere_xml.unlink()


def main():
    parser = argparse.ArgumentParser(
        description='Render and compare depth maps from mesh, MuJoCo spheres, and analytical spheres'
    )
    parser.add_argument('--compare', action='store_true',
                       help='Enable comparison mode (renders all three depth maps and compares)')
    parser.add_argument('--scene', type=str, default=None,
                       help='Path to scene XML (default: auto-detect)')
    parser.add_argument('--sphere-xml', type=str, default=None,
                       help='Path to sphere visualization XML (default: so100_visualization.xml)')
    parser.add_argument('--sphere-dir', type=str, default=None,
                       help='Path to sphere models directory (default: so100_sphere_models)')
    parser.add_argument('--joints', type=float, nargs='+', default=[0.0, 1.57, 1.57, 0.0, 0.0],
                       help='Joint angles in radians')
    parser.add_argument('--joint-names', type=str, nargs='+',
                       default=['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll'],
                       help='Joint names')
    parser.add_argument('--camera-pos', type=float, nargs=3, default=[0.5, 0.3, 0.4],
                       help='Camera position [x, y, z]')
    parser.add_argument('--output-dir', type=str, default='depth_comparison_output',
                       help='Output directory for comparison results')
    parser.add_argument('--size', type=int, nargs=2, default=[640, 480],
                       help='Image size [height, width]')
    
    args = parser.parse_args()
    
    # Determine paths
    if args.scene:
        scene_xml_path = Path(args.scene)
    else:
        scene_xml_path = scene_xml
    
    if args.sphere_xml:
        sphere_xml_path = Path(args.sphere_xml)
    else:
        sphere_xml_path = sphere_xml
    
    if args.sphere_dir:
        sphere_dir_path = Path(args.sphere_dir)
    else:
        sphere_dir_path = sphere_dir
    
    # Compute camera quaternion looking at origin
    camera_quat = quaternion_look_at_origin(args.camera_pos)
    
    if args.compare:
        # Comparison mode
        render_and_compare(
            scene_xml_path, sphere_xml_path, sphere_dir_path,
            args.joints, args.joint_names,
            args.camera_pos, camera_quat.tolist(),
            "comparison_camera", Path(args.output_dir),
            tuple(args.size)
        )
    else:
        # Simple example mode (original behavior)
        print("=" * 70)
        print("Example: Rendering Depth Maps from Joint Angles and Camera Poses")
        print("=" * 70)
        print()
        print("Use --compare flag to enable depth map comparison mode")
        print("Example:")
        print("  python example_render_depth.py --compare \\")
        print("      --joints 0.0 1.57 1.57 0.0 0.0 \\")
        print("      --camera-pos 0.5 0.3 0.4")
        print()


if __name__ == '__main__':
    main()
