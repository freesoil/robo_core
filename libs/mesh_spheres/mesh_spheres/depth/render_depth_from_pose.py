#!/usr/bin/env python3
"""
Render depth map from joint angles and camera pose, then validate with MuJoCo viewer.

This script:
1. Takes joint angles and camera pose (position + quaternion) as input
2. Loads the MuJoCo scene
3. Sets the robot to the specified joint configuration
4. Creates/updates a camera with the specified pose
5. Renders a depth map from that camera
6. Optionally opens MuJoCo viewer for validation

Usage:
    python render_depth_from_pose.py \
        --scene path/to/scene.xml \
        --joints 0.0 1.57 1.57 0.0 0.0 \
        --camera-pos 0.5 0.0 0.3 \
        --camera-quat 1.0 0.0 0.0 0.0 \
        --output depth_map.png \
        --validate
"""

import argparse
import numpy as np
import mujoco
from mujoco import Renderer
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import sys
import os


def quaternion_to_xyaxes(quat_wxyz):
    """
    Convert quaternion (w, x, y, z) to MuJoCo camera xyaxes format.
    
    MuJoCo camera xyaxes is a 6-element vector [x_axis_x, x_axis_y, x_axis_z, y_axis_x, y_axis_y, y_axis_z]
    representing the X and Y axes of the camera frame in world coordinates.
    
    Args:
        quat_wxyz: Quaternion as [w, x, y, z] (MuJoCo convention)
    
    Returns:
        xyaxes: 6-element array [x_axis, y_axis]
    """
    # Convert to scipy Rotation (expects [x, y, z, w])
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = Rotation.from_quat(quat_xyzw)
    
    # Get rotation matrix
    R = rot.as_matrix()
    
    # Camera frame axes in world coordinates
    # MuJoCo camera: X axis (right), Y axis (up), Z axis (forward, negative)
    x_axis = R[:, 0]  # Right
    y_axis = R[:, 1]  # Up
    
    # Return as 6-element vector
    xyaxes = np.concatenate([x_axis, y_axis])
    return xyaxes


def set_camera_pose(model, data, camera_name, position, quaternion_wxyz):
    """
    Set camera pose in MuJoCo model/data.
    
    Note: This modifies the camera pose in the data structure, which affects rendering.
    The model structure itself is read-only, so we need to either:
    1. Create a new camera dynamically (not directly supported)
    2. Use a custom view matrix in the renderer
    3. Modify data.cam_xpos and data.cam_xquat (if camera is tracked)
    
    For now, we'll use the renderer with a custom view matrix.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        camera_name: Name of camera (or None to create custom view)
        position: Camera position [x, y, z]
        quaternion_wxyz: Camera orientation as [w, x, y, z]
    
    Returns:
        view_matrix: 4x4 view matrix for rendering
    """
    # Convert quaternion to rotation matrix
    quat_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    rot = Rotation.from_quat(quat_xyzw)
    R = rot.as_matrix()
    
    # Camera frame: X (right), Y (up), Z (forward, but MuJoCo uses -Z as forward)
    # Build view matrix: [R^T | -R^T * t]
    t = np.array(position)
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R.T  # Transpose for view matrix
    view_matrix[:3, 3] = -R.T @ t
    
    return view_matrix


def render_depth_with_custom_camera(model, data, renderer, position, quaternion_wxyz, size=(640, 480)):
    """
    Render depth map using a custom camera pose.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        renderer: MuJoCo Renderer instance
        position: Camera position [x, y, z]
        quaternion_wxyz: Camera orientation [w, x, y, z]
        size: Image size (height, width)
    
    Returns:
        depth: Depth image as float array [H, W] in meters
    """
    # Resize renderer if needed
    if renderer.height != size[0] or renderer.width != size[1]:
        renderer = Renderer(model, height=size[0], width=size[1])
    
    # Convert quaternion to rotation matrix
    quat_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    rot = Rotation.from_quat(quat_xyzw)
    R = rot.as_matrix()
    
    # Build view matrix
    t = np.array(position)
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R.T
    view_matrix[:3, 3] = -R.T @ t
    
    # Enable depth rendering
    renderer.enable_depth_rendering()
    
    # Update scene with custom view matrix
    # Note: MuJoCo Renderer.update_scene doesn't directly accept view matrix,
    # so we need to create a temporary camera or use a different approach.
    # For now, we'll create a camera in the model dynamically by modifying XML,
    # or use the renderer's scene update with a custom camera.
    
    # Alternative: Create a temporary camera by modifying the scene
    # Actually, we can use mujoco.mjvCamera to set custom view
    from mujoco import mjvCamera
    
    # Create a camera object
    cam = mjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1  # Not using a fixed camera
    cam.trackbodyid = -1  # Not tracking a body
    
    # Set camera pose
    # MuJoCo camera uses lookat point and distance
    # We need to convert position + quaternion to lookat format
    # For now, let's use a simpler approach: add a temporary camera to the model
    
    # Actually, the easiest way is to temporarily modify data.cam_xpos and data.cam_xquat
    # But this only works if the camera is tracked (has a body)
    
    # Best approach: Create a new camera in the XML temporarily, or use the renderer
    # with a scene that has the camera at the desired pose
    
    # For this implementation, we'll add a temporary camera to the model
    # by creating a new XML with the camera at the desired pose
    
    # Simpler: Use the first available camera and modify its pose in data
    # But cameras in worldbody are fixed...
    
    # Workaround: Use renderer with scene update and manually set camera transform
    # We'll need to use the low-level rendering API
    
    # Actually, let's use a different approach: create a camera body and set its pose
    # Or modify an existing camera's data
    
    # For now, let's try to use the renderer's update_scene with a custom camera
    # by creating a temporary camera in the model
    
    # Let's use a simpler method: find or create a camera, then render
    # We'll add a temporary camera to the scene XML and reload, or
    # we can use the renderer's internal camera settings
    
    # Best solution: Use mujoco's mjvOption and mjvScene with custom camera
    from mujoco import mjvOption, mjvScene, mjrContext
    
    # Create scene and context
    scn = mjvScene(model, maxgeom=10000)
    con = mjrContext(model, mujoco.mjFONTSCALE_150)
    
    # Set up camera
    cam = mjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    
    # Set camera pose using lookat (we'll compute lookat from position + quaternion)
    # Camera forward is -Z in camera frame
    forward_dir = -R[:, 2]  # Negative Z axis in world frame
    lookat_point = position + forward_dir * 1.0  # Look 1m forward
    
    cam.lookat[:] = lookat_point
    cam.distance = 1.0
    cam.azimuth = 0.0
    cam.elevation = 0.0
    
    # Update scene
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scn
    )
    
    # Render depth
    depth = np.zeros((size[0], size[1]), dtype=np.float32)
    mujoco.mjr_renderDepth(scn, con, depth, size[1], size[0])
    
    return depth


def render_depth_simple(model, data, renderer, camera_name, position, quaternion_wxyz, size=(640, 480)):
    """
    Simplified depth rendering: add a temporary camera to the model or use existing one.
    
    For now, we'll modify an existing camera's data if possible, or create a new one.
    Actually, the simplest is to temporarily add a camera to the XML.
    
    Let's use a workaround: modify the scene XML to add/update a camera, then reload.
    Or better: use the renderer with a custom view by creating a temporary camera body.
    """
    # Resize renderer if needed
    if renderer.height != size[0] or renderer.width != size[1]:
        renderer = Renderer(model, height=size[0], width=size[1])
    
    # Try to find or create a camera named "custom_camera"
    camera_name_to_use = camera_name if camera_name else "custom_camera"
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_to_use)
    
    if camera_id < 0:
        # Camera doesn't exist - we need to add it to the model
        # Since we can't modify the model at runtime, we'll use a workaround:
        # Use the first available camera and note that the pose won't match
        # Or create a new model with the camera added
        
        print(f"Warning: Camera '{camera_name_to_use}' not found. Using first available camera.")
        if model.ncam > 0:
            camera_id = 0
            camera_name_to_use = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
            print(f"Using camera: {camera_name_to_use}")
        else:
            raise ValueError("No cameras found in model. Please add a camera to the scene XML.")
    
    # For cameras attached to bodies, we can modify data.cam_xpos and data.cam_xquat
    # But for fixed cameras in worldbody, we need to modify the model (read-only)
    
    # Workaround: Create a temporary model with the camera at the desired pose
    # This requires XML manipulation, which is complex
    
    # Simpler approach: Use the renderer's low-level API with custom view matrix
    # We'll render using mjvScene and mjrContext with a custom camera
    
    # Convert quaternion to xyaxes for MuJoCo camera
    xyaxes = quaternion_to_xyaxes(quaternion_wxyz)
    
    # Temporarily modify camera pose in data (only works for tracked cameras)
    # For fixed cameras, we need to recreate the model
    
    # Best solution: Use the renderer with update_scene and a custom camera object
    # But Renderer.update_scene doesn't support custom cameras directly
    
    # Let's use the mjv/mjr API directly
    from mujoco import mjvCamera, mjvOption, mjvScene, mjrContext
    
    # Create rendering objects
    scn = mjvScene(model, maxgeom=10000)
    con = mjrContext(model, mujoco.mjFONTSCALE_150)
    opt = mjvOption()
    
    # Create camera with custom pose
    cam = mjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    
    # Convert position + quaternion to lookat format
    quat_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    rot = Rotation.from_quat(quat_xyzw)
    R = rot.as_matrix()
    
    # Camera looks along -Z axis
    forward = -R[:, 2]
    lookat_dist = 1.0  # Distance to lookat point
    lookat_point = position + forward * lookat_dist
    
    cam.lookat[:] = lookat_point
    cam.distance = lookat_dist
    
    # Compute azimuth and elevation from rotation
    # This is approximate - for exact control, we'd need to use the quaternion directly
    # But mjvCamera uses lookat + distance + azimuth + elevation
    
    # Update scene
    mujoco.mjv_updateScene(
        model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scn
    )
    
    # Render depth
    depth = np.zeros((size[0], size[1]), dtype=np.float32)
    mujoco.mjr_renderDepth(scn, con, depth, size[1], size[0])
    
    return depth


def add_camera_to_model_xml(xml_path, camera_name, position, quaternion_wxyz, output_path=None, framebuffer_size=None):
    """
    Add or update a camera in the MuJoCo XML file.
    
    Args:
        xml_path: Path to input XML file
        camera_name: Name of camera to add/update
        position: Camera position [x, y, z]
        quaternion_wxyz: Camera orientation [w, x, y, z]
        output_path: Path to output XML (if None, modifies in place)
        framebuffer_size: Tuple (height, width) for framebuffer size (optional)
    
    Returns:
        Path to modified XML file
    """
    import xml.etree.ElementTree as ET
    
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Convert quaternion to xyaxes
    xyaxes = quaternion_to_xyaxes(quaternion_wxyz)
    xyaxes_str = ' '.join([f'{v:.6f}' for v in xyaxes])
    pos_str = ' '.join([f'{v:.6f}' for v in position])
    
    # Preserve and fix compiler meshdir and include paths if output is in different location
    if output_path is not None:
        output_path = Path(output_path)
        if output_path != xml_path:
            # Output is in a different location - need to fix relative paths
            # Fix compiler meshdir
            compiler = root.find('compiler')
            if compiler is not None:
                meshdir = compiler.get('meshdir', '')
                if meshdir:
                    # Resolve the original meshdir path relative to input XML
                    original_meshdir = (xml_path.parent / meshdir).resolve()
                    
                    # Calculate relative path from output XML location to original meshdir
                    try:
                        relative_meshdir = Path(os.path.relpath(original_meshdir, output_path.parent))
                        # Ensure path uses forward slashes and ends with / if original did
                        meshdir_str = str(relative_meshdir).replace('\\', '/')
                        if meshdir.endswith('/'):
                            meshdir_str = meshdir_str + '/'
                        compiler.set('meshdir', meshdir_str)
                    except ValueError:
                        # If paths are on different drives (Windows) or can't compute relative path,
                        # use absolute path
                        compiler.set('meshdir', str(original_meshdir).replace('\\', '/') + '/')
            
            # Fix include file paths
            for include_elem in root.findall('include'):
                include_file = include_elem.get('file', '')
                if include_file:
                    # Resolve the original include path relative to input XML
                    original_include = (xml_path.parent / include_file).resolve()
                    
                    # Calculate relative path from output XML location to original include file
                    try:
                        relative_include = Path(os.path.relpath(original_include, output_path.parent))
                        # Ensure path uses forward slashes
                        include_str = str(relative_include).replace('\\', '/')
                        include_elem.set('file', include_str)
                    except ValueError:
                        # If paths are on different drives (Windows) or can't compute relative path,
                        # use absolute path
                        include_elem.set('file', str(original_include).replace('\\', '/'))
    
    # Set framebuffer size if specified
    if framebuffer_size is not None:
        visual = root.find('visual')
        if visual is None:
            visual = ET.SubElement(root, 'visual')
        
        global_elem = visual.find('global')
        if global_elem is None:
            global_elem = ET.SubElement(visual, 'global')
        
        # Set offscreen framebuffer size
        global_elem.set('offheight', str(framebuffer_size[0]))
        global_elem.set('offwidth', str(framebuffer_size[1]))
    
    # Find worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("No worldbody found in XML")
    
    # Check if camera already exists
    existing_camera = None
    for camera in worldbody.findall('camera'):
        if camera.get('name') == camera_name:
            existing_camera = camera
            break
    
    if existing_camera:
        # Update existing camera
        existing_camera.set('pos', pos_str)
        existing_camera.set('xyaxes', xyaxes_str)
    else:
        # Add new camera
        camera_elem = ET.SubElement(worldbody, 'camera')
        camera_elem.set('name', camera_name)
        camera_elem.set('pos', pos_str)
        camera_elem.set('xyaxes', xyaxes_str)
    
    # Write to output
    if output_path is None:
        output_path = xml_path
    else:
        output_path = Path(output_path)
    
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    return output_path


def set_joint_angles(model, data, joint_names, joint_angles):
    """
    Set joint angles in MuJoCo model.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_names: List of joint names
        joint_angles: Array of joint angles [n_joints]
    """
    if len(joint_names) != len(joint_angles):
        raise ValueError(f"Number of joint names ({len(joint_names)}) doesn't match number of angles ({len(joint_angles)})")
    
    for name, angle in zip(joint_names, joint_angles):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' not found in model")
        
        qpos_addr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_addr] = angle
    
    # Zero velocities
    data.qvel[:] = 0
    
    # Forward kinematics
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(
        description='Render depth map from joint angles and camera pose',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python render_depth_from_pose.py \\
      --scene scene.xml \\
      --joints 0.0 1.57 1.57 0.0 0.0 \\
      --camera-pos 0.5 0.0 0.3 \\
      --camera-quat 1.0 0.0 0.0 0.0

  # With validation viewer
  python render_depth_from_pose.py \\
      --scene scene.xml \\
      --joints 0.0 1.57 1.57 0.0 0.0 \\
      --camera-pos 0.5 0.0 0.3 \\
      --camera-quat 1.0 0.0 0.0 0.0 \\
      --validate

  # Custom output and size
  python render_depth_from_pose.py \\
      --scene scene.xml \\
      --joints 0.0 1.57 1.57 0.0 0.0 \\
      --camera-pos 0.5 0.0 0.3 \\
      --camera-quat 1.0 0.0 0.0 0.0 \\
      --output my_depth.png \\
      --size 1280 720
        """
    )
    
    parser.add_argument('--scene', type=str, required=True,
                       help='Path to MuJoCo scene XML file')
    parser.add_argument('--joints', type=float, nargs='+', required=True,
                       help='Joint angles (in radians)')
    parser.add_argument('--joint-names', type=str, nargs='+', default=None,
                       help='Joint names (default: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll)')
    parser.add_argument('--camera-pos', type=float, nargs=3, required=True,
                       help='Camera position [x, y, z] in meters')
    parser.add_argument('--camera-quat', type=float, nargs=4, required=True,
                       help='Camera quaternion [w, x, y, z]')
    parser.add_argument('--camera-name', type=str, default='custom_camera',
                       help='Name of camera to use/create (default: custom_camera)')
    parser.add_argument('--output', type=str, default='depth_map.png',
                       help='Output depth map image path (default: depth_map.png)')
    parser.add_argument('--size', type=int, nargs=2, default=[640, 480],
                       help='Image size [height, width] (default: 640 480)')
    parser.add_argument('--validate', action='store_true',
                       help='Open MuJoCo viewer for validation')
    parser.add_argument('--temp-xml', type=str, default=None,
                       help='Path to temporary XML with camera (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Default joint names for SO-ARM100
    if args.joint_names is None:
        args.joint_names = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll']
    
    if len(args.joints) != len(args.joint_names):
        parser.error(f"Number of joint angles ({len(args.joints)}) must match number of joint names ({len(args.joint_names)})")
    
    # Load scene
    scene_path = Path(args.scene)
    if not scene_path.exists():
        parser.error(f"Scene file not found: {scene_path}")
    
    print(f"Loading scene: {scene_path}")
    
    # Add camera to XML temporarily
    if args.temp_xml is None:
        temp_xml_path = scene_path.parent / f"{scene_path.stem}_with_camera{scene_path.suffix}"
    else:
        temp_xml_path = Path(args.temp_xml)
    
    print(f"Adding camera '{args.camera_name}' to scene XML...")
    try:
        # Add camera and set framebuffer size to match requested render size
        add_camera_to_model_xml(
            scene_path,
            args.camera_name,
            args.camera_pos,
            args.camera_quat,
            temp_xml_path,
            framebuffer_size=(args.size[0], args.size[1])
        )
        print(f"✓ Created temporary XML: {temp_xml_path}")
    except Exception as e:
        print(f"Error adding camera to XML: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Load model
    import mujoco
    try:
        model = mujoco.MjModel.from_xml_path(str(temp_xml_path))
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"✓ Model loaded successfully")
    print(f"  Number of joints: {model.njnt}")
    print(f"  Number of cameras: {model.ncam}")
    print(f"  Framebuffer size: {model.vis.global_.offheight}x{model.vis.global_.offwidth}")
    
    # Check if requested size exceeds framebuffer
    if args.size[0] > model.vis.global_.offheight or args.size[1] > model.vis.global_.offwidth:
        print(f"⚠ Warning: Requested size ({args.size[0]}x{args.size[1]}) exceeds framebuffer ({model.vis.global_.offheight}x{model.vis.global_.offwidth})")
        print(f"  Adjusting size to fit framebuffer...")
        args.size = (min(args.size[0], model.vis.global_.offheight), 
                     min(args.size[1], model.vis.global_.offwidth))
        print(f"  Using size: {args.size[0]}x{args.size[1]}")
    
    # Set joint angles
    print(f"\nSetting joint angles...")
    try:
        set_joint_angles(model, data, args.joint_names, args.joints)
        print(f"✓ Joint angles set:")
        for name, angle in zip(args.joint_names, args.joints):
            print(f"  {name}: {angle:.4f} rad ({np.degrees(angle):.2f} deg)")
    except Exception as e:
        print(f"Error setting joint angles: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create renderer
    print(f"\nCreating renderer ({args.size[0]}x{args.size[1]})...")
    try:
        renderer = Renderer(model, height=args.size[0], width=args.size[1])
    except ValueError as e:
        print(f"Error creating renderer: {e}")
        print(f"\nTrying to fix by adjusting framebuffer size in XML...")
        # Re-add camera with larger framebuffer
        add_camera_to_model_xml(
            scene_path,
            args.camera_name,
            args.camera_pos,
            args.camera_quat,
            temp_xml_path,
            framebuffer_size=(max(args.size[0], 1024), max(args.size[1], 1024))
        )
        # Reload model
        model = mujoco.MjModel.from_xml_path(str(temp_xml_path))
        data = mujoco.MjData(model)
        # Set joint angles again after reload
        set_joint_angles(model, data, args.joint_names, args.joints)
        # Retry renderer
        renderer = Renderer(model, height=args.size[0], width=args.size[1])
    
    # Render depth
    print(f"Rendering depth map from camera '{args.camera_name}'...")
    try:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=args.camera_name)
        depth = renderer.render()  # Returns float array in meters
        renderer.disable_depth_rendering()
        print(f"✓ Depth map rendered: shape {depth.shape}, range [{depth.min():.3f}, {depth.max():.3f}] m")
    except Exception as e:
        print(f"Error rendering depth: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save depth map
    output_path = Path(args.output)
    print(f"\nSaving depth map to: {output_path}")
    
    # Normalize depth for visualization (0-2m range)
    depth_normalized = np.clip(depth, 0, 2.0) / 2.0
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Save as image
    plt.imsave(output_path, depth_uint8, cmap='viridis')
    print(f"✓ Depth map saved")
    
    # Also save raw depth as numpy array
    npz_path = output_path.with_suffix('.npz')
    np.savez_compressed(npz_path, depth=depth, position=args.camera_pos, quaternion=args.camera_quat)
    print(f"✓ Raw depth data saved to: {npz_path}")
    
    # Validate with viewer
    if args.validate:
        print(f"\nOpening MuJoCo viewer for validation...")
        print(f"  Camera: {args.camera_name}")
        print(f"  Position: {args.camera_pos}")
        print(f"  Quaternion: {args.camera_quat}")
        print(f"\n  Press ESC or close window to exit")
        
        import mujoco.viewer
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set camera to match our custom camera
            viewer.cam.lookat[:] = np.array(args.camera_pos) + np.array([0, 0, -1])  # Approximate
            viewer.cam.distance = 1.0
            
            # Keep viewer open
            while viewer.is_running():
                # Update scene if needed
                mujoco.mj_step(model, data)
                viewer.sync()
    
    # Cleanup temporary XML if it was auto-generated
    if args.temp_xml is None and temp_xml_path.exists():
        print(f"\nCleaning up temporary XML: {temp_xml_path}")
        temp_xml_path.unlink()
    
    print(f"\n✓ Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

