#!/usr/bin/env python3
"""
Depth map comparison utilities for comparing MuJoCo mesh depth, MuJoCo sphere depth,
and analytically synthesized sphere depth.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from scipy.spatial.transform import Rotation
import mujoco
from mujoco import Renderer

try:
    from ..converter import SphereCollection
except ImportError:
    import sys
    script_dir = Path(__file__).parent
    libs_dir = script_dir.parent
    if str(libs_dir) not in sys.path:
        sys.path.insert(0, str(libs_dir))
    from mesh_spheres import SphereCollection


def extract_camera_intrinsics_from_mujoco(
    model: mujoco.MjModel,
    camera_id: int,
    image_size: Tuple[int, int]
) -> Tuple[np.ndarray, float]:
    """
    Extract camera intrinsics from MuJoCo model.
    
    Returns:
        K: 3x3 camera intrinsics matrix
        focal_length: Focal length in pixels
    """
    height, width = image_size
    
    # Get camera FOV from model (if available)
    # MuJoCo cameras have fovy (vertical field of view in degrees)
    if hasattr(model, 'cam_fovy') and model.cam_fovy is not None:
        fovy_deg = model.cam_fovy[camera_id]
        fovy_rad = np.deg2rad(fovy_deg)
        # Focal length from FOV: f = (height/2) / tan(fovy/2)
        focal_length = (height / 2.0) / np.tan(fovy_rad / 2.0)
    else:
        # Default: use provided focal length or estimate from image size
        # Assume ~60 degree FOV
        fovy_rad = np.deg2rad(60.0)
        focal_length = (height / 2.0) / np.tan(fovy_rad / 2.0)
    
    # Principal point at image center
    cx, cy = width / 2.0, height / 2.0
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    
    return K, focal_length


def synthesize_depth_from_spheres(
    sphere_collections: Dict[str, SphereCollection],
    camera_pos: np.ndarray,
    camera_quat: np.ndarray,
    image_size: Tuple[int, int],
    focal_length: float = 800.0,
    max_depth: float = 2.0,
    model: Optional[mujoco.MjModel] = None,
    data: Optional[mujoco.MjData] = None
) -> np.ndarray:
    """
    Synthesize depth map analytically from sphere collections.
    
    Args:
        sphere_collections: Dictionary mapping body names to SphereCollection objects
        camera_pos: Camera position [x, y, z] in world coordinates
        camera_quat: Camera quaternion [w, x, y, z] (MuJoCo convention)
        image_size: Image size (height, width)
        focal_length: Camera focal length in pixels
        max_depth: Maximum depth value in meters
        model: MuJoCo model (optional, for body transformations)
        data: MuJoCo data (optional, for body poses after forward kinematics)
    
    Returns:
        Depth map as float array [H, W] in meters
    
    Note:
        If model and data are provided, spheres are transformed from body-local to world
        coordinates using the current robot configuration. Otherwise, spheres are assumed
        to be already in world coordinates.
    """
    height, width = image_size
    depth_map = np.full((height, width), max_depth, dtype=np.float32)
    
    # Extract camera pose from MuJoCo if available (to ensure exact match)
    # Note: MuJoCo cameras are stored in the model, not data
    if model is not None and data is not None:
        # Debug: print camera info if available
        if model.ncam > 0:
            print(f"  [Analytical synthesis] Model has {model.ncam} camera(s)")
            for i in range(model.ncam):
                cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                if cam_name:
                    # Get camera pose from model
                    cam_pos_model = model.cam_pos[i].copy()
                    cam_quat_model = model.cam_quat[i].copy()
                    
                    # Transform to world if attached to body
                    cam_bodyid = model.cam_bodyid[i]
                    if cam_bodyid >= 0:
                        body_pos = data.xpos[cam_bodyid]
                        body_quat = data.xquat[cam_bodyid]
                        body_quat_xyzw = np.array([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
                        body_rot = Rotation.from_quat(body_quat_xyzw)
                        body_R = body_rot.as_matrix()
                        cam_pos_world = body_pos + body_R @ cam_pos_model
                        print(f"    Camera '{cam_name}': model_pos={cam_pos_model}, world_pos={cam_pos_world}")
                    else:
                        print(f"    Camera '{cam_name}': pos={cam_pos_model}, quat={cam_quat_model}")
        
        print(f"  [Analytical synthesis] Using provided camera pose:")
        print(f"    Position: {camera_pos}")
        print(f"    Quaternion: {camera_quat}")
    
    # Convert quaternion to rotation matrix
    # MuJoCo camera: X=right, Y=up, Z axis in camera frame
    # MuJoCo camera looks along -Z axis in camera frame (negative Z = forward)
    quat_xyzw = np.array([camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]])
    rot = Rotation.from_quat(quat_xyzw)
    R_cam_to_world = rot.as_matrix()
    R_world_to_cam = R_cam_to_world.T
    
    # Camera frame axes in world coordinates
    cam_x_axis = R_cam_to_world[:, 0]  # Right
    cam_y_axis = R_cam_to_world[:, 1]  # Up
    cam_z_axis = R_cam_to_world[:, 2]  # Camera Z axis (forward is -Z)
    cam_forward = -cam_z_axis  # Forward direction in world
    
    # Debug: Print camera orientation
    print(f"  [Analytical synthesis] Camera orientation:")
    print(f"    Position: {camera_pos}")
    print(f"    Forward direction (world): {cam_forward}")
    print(f"    Right direction (world): {cam_x_axis}")
    print(f"    Up direction (world): {cam_y_axis}")
    
    # Check if origin is in view
    origin_cam = R_world_to_cam @ (np.zeros(3) - camera_pos)
    print(f"    Origin in camera frame: {origin_cam}")
    print(f"    Origin depth (Z): {origin_cam[2]:.3f} (negative = in front)")
    
    # Check direction to origin
    origin_dir = -camera_pos / np.linalg.norm(camera_pos) if np.linalg.norm(camera_pos) > 1e-6 else np.zeros(3)
    forward_dot_origin = np.dot(cam_forward, origin_dir)
    print(f"    Camera forward · origin direction: {forward_dot_origin:.3f} (1.0 = looking at origin)")
    
    # Extract camera intrinsics from MuJoCo if available
    if model is not None:
        # Try to find camera in model to get actual FOV
        camera_id = -1
        for i in range(model.ncam):
            camera_id = i
            break
        
        if camera_id >= 0:
            K, focal_length_actual = extract_camera_intrinsics_from_mujoco(model, camera_id, image_size)
            print(f"  [Analytical synthesis] Extracted camera intrinsics from MuJoCo:")
            print(f"    Focal length: {focal_length_actual:.1f} pixels (provided: {focal_length:.1f})")
            print(f"    Principal point: ({K[0,2]:.1f}, {K[1,2]:.1f})")
            focal_length = focal_length_actual
        else:
            # Fallback to provided focal length
            cx, cy = width / 2, height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ])
            print(f"  [Analytical synthesis] Using provided focal length: {focal_length:.1f}")
    else:
        # No model - use provided focal length
        cx, cy = width / 2, height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        print(f"  [Analytical synthesis] Using provided focal length: {focal_length:.1f}")
    
    # Build body transformation map if model/data provided
    body_transforms = {}
    if model is not None and data is not None:
        print(f"  [Analytical synthesis] Applying body transformations based on joint configuration")
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                # Get body pose in world frame
                body_pos = data.xpos[i]  # Body position in world
                body_quat_wxyz = data.xquat[i]  # Body quaternion [w, x, y, z]
                
                # Convert to rotation matrix
                body_quat_xyzw = np.array([body_quat_wxyz[1], body_quat_wxyz[2], 
                                          body_quat_wxyz[3], body_quat_wxyz[0]])
                body_rot = Rotation.from_quat(body_quat_xyzw)
                body_R = body_rot.as_matrix()
                
                body_transforms[body_name] = (body_pos, body_R)
        print(f"  [Analytical synthesis] Built transforms for {len(body_transforms)} bodies")
        print(f"  [Analytical synthesis] Available body names: {list(body_transforms.keys())[:10]}...")  # Show first 10
        
        # Debug: Show where some bodies are relative to camera
        print(f"  [Analytical synthesis] Sample body positions relative to camera:")
        for i, (body_name, (body_pos, body_R)) in enumerate(list(body_transforms.items())[:5]):
            body_cam = R_world_to_cam @ (body_pos - camera_pos)
            body_depth = -body_cam[2] if body_cam[2] < 0 else 0
            print(f"    {body_name}: world={body_pos}, cam={body_cam}, depth={body_depth:.3f}")
    else:
        print(f"  [Analytical synthesis] Warning: No MuJoCo model/data provided - assuming spheres in world coordinates")
    
    # Process all spheres from all collections
    total_spheres = sum(len(c.spheres) for c in sphere_collections.values())
    print(f"  [Analytical synthesis] Processing {total_spheres} spheres from {len(sphere_collections)} collections")
    print(f"  [Analytical synthesis] Sphere collection names: {list(sphere_collections.keys())}")
    
    spheres_processed = 0
    
    for body_name, collection in sphere_collections.items():
        # Get body transform if available
        # Try exact match first, then try variations (removing suffixes, etc.)
        transform_found = False
        matched_body = None
        if body_name in body_transforms:
            body_pos, body_R = body_transforms[body_name]
            transform_found = True
            matched_body = body_name
        else:
            # Try to find a matching body name (handle mesh name variations)
            # Common patterns: mesh name might be "Base" but body is "base" or "Base_Body"
            for bt_name in body_transforms.keys():
                # Try various matching strategies
                mesh_lower = body_name.lower().replace('_', '').replace('-', '')
                body_lower = bt_name.lower().replace('_', '').replace('-', '')
                
                if (body_name.lower() in bt_name.lower() or 
                    bt_name.lower() in body_name.lower() or
                    mesh_lower == body_lower or
                    mesh_lower in body_lower or
                    body_lower in mesh_lower):
                    body_pos, body_R = body_transforms[bt_name]
                    transform_found = True
                    matched_body = bt_name
                    print(f"  [Analytical synthesis] Matched '{body_name}' to body '{bt_name}'")
                    break
        
        if not transform_found:
            # No transform available - assume identity (spheres in world coords)
            # This will happen for the base body which is at the origin
            body_pos = np.zeros(3)
            body_R = np.eye(3)
            if model is not None:
                print(f"  [Analytical synthesis] Warning: No transform found for '{body_name}', using identity")
                print(f"    Tried matching against: {list(body_transforms.keys())[:5]}...")
        
        spheres_in_front = 0
        spheres_in_image = 0
        spheres_behind = 0
        spheres_too_far = 0
        min_cam_z = float('inf')
        max_cam_z = float('-inf')
        
        for sphere in collection.spheres:
            spheres_processed += 1
            # Transform sphere center from body-local to world coordinates
            center_local = sphere.center
            center_world = body_pos + body_R @ center_local
            
            # Transform sphere center to camera frame
            center_cam = R_world_to_cam @ (center_world - camera_pos)
            
            # Track min/max Z for debugging
            min_cam_z = min(min_cam_z, center_cam[2])
            max_cam_z = max(max_cam_z, center_cam[2])
            
            # MuJoCo camera looks along -Z axis in camera frame
            # Convert to standard camera frame (+Z forward) for easier math
            center_cam_standard = np.array([center_cam[0], center_cam[1], -center_cam[2]])
            
            # Check if sphere is in front of camera and within reasonable distance
            if center_cam_standard[2] <= 0:  # Behind camera or at camera plane
                spheres_behind += 1
                continue
            
            depth = center_cam_standard[2]  # Depth = Z in standard frame
            if depth > max_depth:  # Too far away
                spheres_too_far += 1
                continue
            
            spheres_in_front += 1
            
            # Debug: print first few sphere positions
            if spheres_in_front <= 3:
                print(f"    Sphere {spheres_in_front} from '{body_name}': world={center_world}, cam={center_cam}, depth={depth:.3f}")
            
            # Project sphere center to image plane
            # MuJoCo camera coordinate system:
            # - Camera looks along -Z (negative Z = forward)
            # - X = right, Y = up in camera frame
            # - For projection: we need to use standard pinhole but account for -Z convention
            # 
            # Standard pinhole: u = f*X/Z, v = f*Y/Z where Z > 0 is forward
            # MuJoCo: Z < 0 is forward, so we use: u = f*X/(-Z), v = f*Y/(-Z)
            # But we also need to check MuJoCo's image coordinate convention
            # MuJoCo images: (0,0) is top-left, u increases right, v increases down
            # Standard: (0,0) is top-left, u increases right, v increases down
            # So the convention matches, but we need to handle the -Z
            
            # Transform to camera frame where forward is +Z (standard convention)
            # MuJoCo camera frame: forward = -Z, so we flip Z
            center_cam_standard = np.array([center_cam[0], center_cam[1], -center_cam[2]])
            
            # Now use standard pinhole projection (Z > 0 is forward)
            if center_cam_standard[2] <= 0:
                continue  # Behind camera
            
            # Standard projection: u = f*X/Z + cx, v = f*Y/Z + cy
            # MuJoCo image coordinates: (0,0) at top-left, Y increases downward
            # Camera Y points up, so we need to flip Y: v = cy - (f*Y/Z) instead of cy + (f*Y/Z)
            u = (focal_length * center_cam_standard[0] / center_cam_standard[2]) + K[0, 2]
            v = K[1, 2] - (focal_length * center_cam_standard[1] / center_cam_standard[2])  # Flip Y axis
            u, v = int(u), int(v)
            
            # Debug first few projections
            if spheres_in_front <= 5:  # Show more projections
                print(f"      Projection: cam_mujoco={center_cam}, cam_standard={center_cam_standard}, pixel=({u}, {v}), bounds=({width}, {height})")
                if not (0 <= u < width and 0 <= v < height):
                    print(f"        OUT OF BOUNDS! u={u} not in [0, {width}), v={v} not in [0, {height})")
            
            # Check if center is within image bounds
            if not (0 <= u < width and 0 <= v < height):
                continue
            
            spheres_in_image += 1
            
            # Compute depth at sphere center
            # In standard camera frame (+Z forward), depth = Z
            # In MuJoCo frame (-Z forward), we converted to standard, so depth = center_cam_standard[2]
            depth_center = center_cam_standard[2]
            
            # Approximate sphere radius in camera frame (projection of radius)
            # For small angles, radius in image ≈ (focal_length * sphere_radius) / depth
            radius_2d = (focal_length * sphere.radius) / depth_center
            
            # Compute depth range for this sphere
            depth_near = depth_center - sphere.radius
            depth_far = depth_center + sphere.radius
            
            # Rasterize sphere: for each pixel, compute if it's inside sphere
            # and update depth if this sphere is closer
            radius_pixels = int(np.ceil(radius_2d))
            u_min = max(0, u - radius_pixels)
            u_max = min(width, u + radius_pixels + 1)
            v_min = max(0, v - radius_pixels)
            v_max = min(height, v + radius_pixels + 1)
            
            for v_pix in range(v_min, v_max):
                for u_pix in range(u_min, u_max):
                    # Pixel to camera ray
                    # Convert pixel (u, v) to camera ray direction
                    # Standard: pixel -> normalized camera coordinates
                    # u = f*X/Z + cx  =>  X/Z = (u - cx)/f
                    # v = cy - f*Y/Z  =>  Y/Z = (cy - v)/f  (Y flipped for MuJoCo)
                    # So direction in camera frame (standard, +Z forward): [X/Z, Y/Z, 1]
                    x_norm = (u_pix - K[0, 2]) / focal_length
                    y_norm = (K[1, 2] - v_pix) / focal_length  # Flip Y to match projection
                    
                    # Direction in standard camera frame (+Z forward)
                    pixel_cam_standard = np.array([x_norm, y_norm, 1.0])
                    pixel_cam_standard = pixel_cam_standard / np.linalg.norm(pixel_cam_standard)
                    
                    # Convert to MuJoCo camera frame (-Z forward)
                    # MuJoCo: flip Z component
                    pixel_cam_mujoco = np.array([pixel_cam_standard[0], pixel_cam_standard[1], -pixel_cam_standard[2]])
                    
                    # Transform to world coordinates
                    ray_dir_world = R_cam_to_world @ pixel_cam_mujoco
                    ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)
                    
                    # Sphere intersection: ||p - center|| = radius
                    # p = camera_pos + t * ray_dir
                    # ||camera_pos + t*ray_dir - center||^2 = radius^2
                    oc = camera_pos - center_world
                    a = np.dot(ray_dir_world, ray_dir_world)
                    b = 2 * np.dot(oc, ray_dir_world)
                    c = np.dot(oc, oc) - sphere.radius ** 2
                    
                    discriminant = b * b - 4 * a * c
                    if discriminant < 0:
                        continue  # No intersection
                    
                    sqrt_disc = np.sqrt(discriminant)
                    t1 = (-b - sqrt_disc) / (2 * a)
                    t2 = (-b + sqrt_disc) / (2 * a)
                    
                    # Use closest intersection in front of camera
                    t = None
                    if t1 > 0:
                        t = t1
                    elif t2 > 0:
                        t = t2
                    else:
                        continue  # Sphere behind camera
                    
                    # Depth in camera frame (distance along ray)
                    point_world = camera_pos + t * ray_dir_world
                    point_cam_mujoco = R_world_to_cam @ (point_world - camera_pos)
                    # Convert to standard frame (+Z forward) for depth
                    point_cam_standard = np.array([point_cam_mujoco[0], point_cam_mujoco[1], -point_cam_mujoco[2]])
                    depth = point_cam_standard[2]  # Depth = Z in standard frame
                    
                    # Update depth if this sphere is closer and valid
                    if 0 < depth < depth_map[v_pix, u_pix] and depth < max_depth:
                        depth_map[v_pix, u_pix] = depth
    
        # Debug output per collection
        if spheres_processed > 0:
            print(f"  [Analytical synthesis] '{body_name}': {spheres_processed} total, "
                  f"{spheres_in_front} in front, {spheres_in_image} in image, "
                  f"{spheres_behind} behind, {spheres_too_far} too far")
            if spheres_in_front == 0 and spheres_processed > 0:
                print(f"    Z range in camera frame: [{min_cam_z:.3f}, {max_cam_z:.3f}]")
    
    # Debug output
    valid_pixels = np.sum((depth_map > 0) & (depth_map < max_depth))
    print(f"  [Analytical synthesis] Processed {spheres_processed} spheres total")
    print(f"  [Analytical synthesis] Valid depth pixels: {valid_pixels} / {height * width}")
    if valid_pixels == 0:
        print(f"  [Analytical synthesis] WARNING: No valid depth pixels generated!")
        print(f"    Depth range: [{depth_map.min():.6f}, {depth_map.max():.6f}]")
        print(f"    Camera position: {camera_pos}")
        print(f"    Camera quaternion: {camera_quat}")
        print(f"    Try checking:")
        print(f"      1. Are body names matching correctly?")
        print(f"      2. Are spheres in front of the camera?")
        print(f"      3. Is the camera pointing in the right direction?")
    
    return depth_map


def render_depth_from_mujoco_group(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: Optional[Renderer],
    camera_name: str,
    geom_group: int,
    size: Tuple[int, int] = (640, 480)
) -> np.ndarray:
    """
    Render depth map from MuJoCo with only specific geometry group visible.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        renderer: MuJoCo Renderer (optional, will create if None)
        camera_name: Name of camera
        geom_group: Geometry group to render (0=floor, 1=meshes, 2=spheres)
        size: Image size (height, width)
    
    Returns:
        Depth map as float array [H, W] in meters
    
    Note:
        Uses Renderer API which renders all groups. For group filtering, the model
        should be set up with only the desired group visible, or use a separate
        XML file with only that group.
    """
    # Verify camera exists
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found in model")
    
    # Use Renderer API (most reliable across MuJoCo versions)
    # Note: Renderer API doesn't support group filtering directly
    # We render all groups - if you need only group 2, use a separate XML
    if renderer is None:
        renderer = Renderer(model, height=size[0], width=size[1])
    elif renderer.height != size[0] or renderer.width != size[1]:
        renderer = Renderer(model, height=size[0], width=size[1])
    
    # Render depth using Renderer API
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render()
    renderer.disable_depth_rendering()
    
    # Check depth range for debugging
    depth_min, depth_max = depth.min(), depth.max()
    valid_pixels = np.sum((depth > 0) & (depth < 10.0))  # Reasonable range for robotics
    
    # Count geoms in the requested group for info
    geoms_in_group = sum(1 for i in range(model.ngeom) 
                       if model.geom_group[i] == geom_group)
    
    if valid_pixels == 0:
        print(f"  Warning: No valid depth pixels found")
        print(f"    Depth range: [{depth_min:.6f}, {depth_max:.6f}]")
        print(f"    Camera: {camera_name} (ID: {camera_id})")
        print(f"    Total geoms in model: {model.ngeom}")
        print(f"    Geoms in group {geom_group}: {geoms_in_group}")
        print(f"    Note: Renderer API renders all groups, not just group {geom_group}")
    else:
        print(f"  Rendered {valid_pixels} valid depth pixels (range: [{depth_min:.3f}, {depth_max:.3f}] m)")
        if geoms_in_group > 0:
            print(f"    Note: Rendered all groups ({model.ngeom} geoms), group {geom_group} has {geoms_in_group} geoms")
    
    return depth


def compare_depth_maps(
    depth_mesh: np.ndarray,
    depth_spheres_mujoco: np.ndarray,
    depth_spheres_analytic: np.ndarray,
    output_dir: Path,
    prefix: str = "comparison"
) -> Dict[str, float]:
    """
    Compare three depth maps and generate visualization.
    
    Args:
        depth_mesh: Depth from MuJoCo mesh rendering
        depth_spheres_mujoco: Depth from MuJoCo sphere rendering
        depth_spheres_analytic: Depth from analytical sphere synthesis
        output_dir: Directory to save comparison images
        prefix: Prefix for output filenames
    
    Returns:
        Dictionary with comparison metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    valid_mask = (depth_mesh > 0) & (depth_mesh < 2.0)
    
    metrics = {}
    
    # Mesh vs MuJoCo spheres
    diff_mesh_spheres_mj = np.abs(depth_mesh - depth_spheres_mujoco)
    metrics['mesh_vs_spheres_mujoco_mean'] = np.mean(diff_mesh_spheres_mj[valid_mask])
    metrics['mesh_vs_spheres_mujoco_max'] = np.max(diff_mesh_spheres_mj[valid_mask])
    metrics['mesh_vs_spheres_mujoco_std'] = np.std(diff_mesh_spheres_mj[valid_mask])
    
    # MuJoCo spheres vs analytical spheres
    valid_analytic = (depth_spheres_analytic > 0) & (depth_spheres_analytic < 2.0)
    valid_both = valid_mask & valid_analytic
    if np.any(valid_both):
        diff_spheres_mj_analytic = np.abs(depth_spheres_mujoco - depth_spheres_analytic)
        metrics['spheres_mujoco_vs_analytic_mean'] = np.mean(diff_spheres_mj_analytic[valid_both])
        metrics['spheres_mujoco_vs_analytic_max'] = np.max(diff_spheres_mj_analytic[valid_both])
        metrics['spheres_mujoco_vs_analytic_std'] = np.std(diff_spheres_mj_analytic[valid_both])
    else:
        metrics['spheres_mujoco_vs_analytic_mean'] = np.nan
        metrics['spheres_mujoco_vs_analytic_max'] = np.nan
        metrics['spheres_mujoco_vs_analytic_std'] = np.nan
    
    # Mesh vs analytical spheres
    if np.any(valid_both):
        diff_mesh_analytic = np.abs(depth_mesh - depth_spheres_analytic)
        metrics['mesh_vs_analytic_mean'] = np.mean(diff_mesh_analytic[valid_both])
        metrics['mesh_vs_analytic_max'] = np.max(diff_mesh_analytic[valid_both])
        metrics['mesh_vs_analytic_std'] = np.std(diff_mesh_analytic[valid_both])
    else:
        metrics['mesh_vs_analytic_mean'] = np.nan
        metrics['mesh_vs_analytic_max'] = np.nan
        metrics['mesh_vs_analytic_std'] = np.nan
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Depth maps
    im1 = axes[0, 0].imshow(depth_mesh, cmap='viridis', vmin=0, vmax=2.0)
    axes[0, 0].set_title('MuJoCo Mesh Depth')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(depth_spheres_mujoco, cmap='viridis', vmin=0, vmax=2.0)
    axes[0, 1].set_title('MuJoCo Spheres Depth')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(depth_spheres_analytic, cmap='viridis', vmin=0, vmax=2.0)
    axes[0, 2].set_title('Analytical Spheres Depth')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Row 2: Difference maps
    im4 = axes[1, 0].imshow(diff_mesh_spheres_mj, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title(f'Mesh - MuJoCo Spheres\n(Mean: {metrics["mesh_vs_spheres_mujoco_mean"]:.4f}m)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])
    
    if np.any(valid_both):
        im5 = axes[1, 1].imshow(diff_spheres_mj_analytic, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[1, 1].set_title(f'MuJoCo Spheres - Analytical\n(Mean: {metrics["spheres_mujoco_vs_analytic_mean"]:.4f}m)')
        axes[1, 1].axis('off')
        plt.colorbar(im5, ax=axes[1, 1])
        
        im6 = axes[1, 2].imshow(diff_mesh_analytic, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[1, 2].set_title(f'Mesh - Analytical\n(Mean: {metrics["mesh_vs_analytic_mean"]:.4f}m)')
        axes[1, 2].axis('off')
        plt.colorbar(im6, ax=axes[1, 2])
    else:
        axes[1, 1].text(0.5, 0.5, 'No valid pixels', ha='center', va='center')
        axes[1, 1].axis('off')
        axes[1, 2].text(0.5, 0.5, 'No valid pixels', ha='center', va='center')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / f"{prefix}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison saved to: {output_path}")
    print("\nMetrics:")
    print(f"  Mesh vs MuJoCo Spheres:")
    print(f"    Mean error: {metrics['mesh_vs_spheres_mujoco_mean']:.4f}m")
    print(f"    Max error:  {metrics['mesh_vs_spheres_mujoco_max']:.4f}m")
    print(f"    Std error:  {metrics['mesh_vs_spheres_mujoco_std']:.4f}m")
    
    if not np.isnan(metrics['spheres_mujoco_vs_analytic_mean']):
        print(f"  MuJoCo Spheres vs Analytical:")
        print(f"    Mean error: {metrics['spheres_mujoco_vs_analytic_mean']:.4f}m")
        print(f"    Max error:  {metrics['spheres_mujoco_vs_analytic_max']:.4f}m")
        print(f"    Std error:  {metrics['spheres_mujoco_vs_analytic_std']:.4f}m")
    
    return metrics


def visualize_depth_maps_interactive(
    depth_mesh: np.ndarray,
    depth_spheres_mujoco: Optional[np.ndarray] = None,
    depth_spheres_analytic: Optional[np.ndarray] = None,
    title: str = "Depth Map Comparison"
):
    """
    Create an interactive matplotlib visualization of depth maps.
    
    Args:
        depth_mesh: Depth from MuJoCo mesh rendering
        depth_spheres_mujoco: Depth from MuJoCo sphere rendering (optional)
        depth_spheres_analytic: Depth from analytical sphere synthesis (optional)
        title: Window title
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    
    # Determine number of subplots
    n_plots = 1
    if depth_spheres_mujoco is not None:
        n_plots += 1
    if depth_spheres_analytic is not None:
        n_plots += 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Normalize depth for visualization - cap at 2m as requested
    depth_max = 2.0  # Fixed maximum depth scale
    
    # Plot mesh depth
    im0 = axes[0].imshow(depth_mesh, cmap='viridis', vmin=0, vmax=depth_max)
    axes[0].set_title('MuJoCo Mesh Depth', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], label='Depth (m)')
    
    # Plot MuJoCo spheres depth
    plot_idx = 1
    if depth_spheres_mujoco is not None:
        im1 = axes[plot_idx].imshow(depth_spheres_mujoco, cmap='viridis', vmin=0, vmax=depth_max)
        axes[plot_idx].set_title('MuJoCo Spheres Depth', fontsize=12)
        axes[plot_idx].axis('off')
        plt.colorbar(im1, ax=axes[plot_idx], label='Depth (m)')
        plot_idx += 1
    
    # Plot analytical spheres depth
    if depth_spheres_analytic is not None:
        im2 = axes[plot_idx].imshow(depth_spheres_analytic, cmap='viridis', vmin=0, vmax=depth_max)
        axes[plot_idx].set_title('Analytical Spheres Depth', fontsize=12)
        axes[plot_idx].axis('off')
        plt.colorbar(im2, ax=axes[plot_idx], label='Depth (m)')
    
    # Add statistics text
    stats_text = []
    stats_text.append(f"Mesh: range [{depth_mesh.min():.3f}, {depth_mesh.max():.3f}] m")
    if depth_spheres_mujoco is not None:
        valid_mj = np.sum((depth_spheres_mujoco > 0) & (depth_spheres_mujoco < depth_max))
        stats_text.append(f"MuJoCo Spheres: {valid_mj} valid pixels")
    if depth_spheres_analytic is not None:
        valid_an = np.sum((depth_spheres_analytic > 0) & (depth_spheres_analytic < depth_max))
        stats_text.append(f"Analytical: {valid_an} valid pixels")
    
    fig.text(0.5, 0.02, ' | '.join(stats_text), ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Make it interactive - show on screen
    plt.show(block=True)
    
    return fig



