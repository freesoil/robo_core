#!/usr/bin/env python3
"""
Stereo Geometry Utilities for Fiducial Calibration

This module provides utilities for improving marker pose estimation using stereo disparity.
"""

import numpy as np
import cv2
import cv2.aruco as aruco
from typing import Dict, Optional, Tuple, Any
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path
import rolog

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import transforms as tf
    from calibration.markers import detect_marker_poses
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")


class StereoGeometry:
    """Utility class for stereo geometry calculations."""
    
    @staticmethod
    def disparity_to_depth(disparity: np.ndarray, baseline: float, focal_length: float) -> np.ndarray:
        """Convert disparity to depth using stereo geometry."""
        # Avoid division by zero
        valid_disparity = np.where(disparity > 0, disparity, np.inf)
        return (baseline * focal_length) / valid_disparity
    
    @staticmethod
    def pixel_to_3d(pixels: np.ndarray, depths: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates and depths to 3D points.
        
        Args:
            pixels: Array of shape (N, 2) with [u, v] coordinates
            depths: Array of shape (N,) with depth values
            intrinsics: Camera intrinsic matrix
            
        Returns:
            Array of shape (N, 3) with 3D points [x, y, z]
        """
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Vectorized calculation
        x = (pixels[:, 0] - cx) * depths / fx
        y = (pixels[:, 1] - cy) * depths / fy
        z = depths
        
        return np.column_stack([x, y, z])
    
    @staticmethod
    def correct_poses_with_positions(poses: Dict[int, Any], 
                                   marker_ids: np.ndarray, 
                                   new_positions: np.ndarray) -> Dict[int, Any]:
        """
        Update marker poses with new 3D positions while preserving rotations.
        
        Args:
            poses: Dictionary of marker poses (transforms)
            marker_ids: Array of marker IDs
            new_positions: Array of new 3D positions
            
        Returns:
            Dictionary of corrected poses
        """
        corrected_poses = {}
        for i, marker_id in enumerate(marker_ids):
            # Convert numpy scalar to Python int to make it hashable
            marker_id = int(marker_id)
            if marker_id in poses:
                corrected_poses[marker_id] = tf.create_tf(
                    new_positions[i].astype(np.float64), 
                    poses[marker_id].R
                )
        return corrected_poses

    @staticmethod
    def extract_stereo_camera_params(left_cam_info, right_cam_info):
        """
        Extract stereo camera parameters from camera info messages.
        
        Args:
            left_cam_info: Left camera info message
            right_cam_info: Right camera info message
            
        Returns:
            Tuple of (baseline, focal_length, left_intrinsics, right_intrinsics, left_distortion, right_distortion)
        """
        # Extract intrinsics
        left_K = np.array(left_cam_info.k).reshape(3, 3)
        right_K = np.array(right_cam_info.k).reshape(3, 3)
        
        # Extract distortion coefficients
        left_D = np.array(left_cam_info.d)
        right_D = np.array(right_cam_info.d)
        
        # Extract baseline from projection matrices
        left_P = np.array(left_cam_info.p).reshape(3, 4)
        right_P = np.array(right_cam_info.p).reshape(3, 4)
        
        # Baseline is the translation between cameras
        # For rectified stereo: baseline = -right_P[0, 3] / right_P[0, 0]
        baseline = abs(right_P[0, 3] / right_P[0, 0])
        
        # Focal length (use left camera)
        focal_length = left_K[0, 0]
        
        return baseline, focal_length, left_K, right_K, left_D, right_D


class StereoMarkerDetector:
    """Marker detector with stereo disparity correction."""
    
    def __init__(self, baseline: float, focal_length: float):
        """
        Initialize stereo marker detector.
        
        Args:
            baseline: Distance between left and right cameras (meters)
            focal_length: Focal length of the cameras (pixels)
        """
        self.baseline = baseline
        self.focal_length = focal_length
        self._geometry = StereoGeometry()
        
        # Initialize ArUco detector - use older API for compatibility
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
        parameters = aruco.DetectorParameters_create()
        self.detector = (aruco_dict, parameters)
    
    def detect_stereo_markers(
        self, 
        left_image: np.ndarray,
        right_image: np.ndarray,
        marker_length: float,
        left_intrinsics: np.ndarray, 
        left_distortion: np.ndarray,
        right_intrinsics: Optional[np.ndarray] = None,
        right_distortion: Optional[np.ndarray] = None,
        fallback_to_left_only: bool = False,
        **kwargs
    ) -> Dict[int, Any]:
        """
        Detect markers using stereo disparity correction.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            marker_length: Physical marker length
            left_intrinsics: Left camera intrinsic matrix
            left_distortion: Left camera distortion coefficients
            right_intrinsics: Right camera intrinsic matrix (optional, uses left if None)
            right_distortion: Right camera distortion coefficients (optional, uses left if None)
            fallback_to_left_only: Whether to fall back to left-only detection when stereo fails (default: False)
            
        Returns:
            Dictionary mapping marker IDs to stereo-corrected poses
        """
        rolog.stat("detect_markers")
        # Use left camera parameters for right if not provided
        if right_intrinsics is None:
            right_intrinsics = left_intrinsics
        if right_distortion is None:
            right_distortion = left_distortion
        
        # Detect markers in both images
        left_data = self._detect_markers_with_centers(
            left_image, marker_length, left_intrinsics, left_distortion)
        right_data = self._detect_markers_with_centers(
            right_image, marker_length, right_intrinsics, right_distortion)

        rolog.stat("find_common")
        if not left_data or not right_data:
            rolog.warn("No markers detected in one or both stereo images")
            # Fall back to mono detection from left image only if enabled
            if fallback_to_left_only and left_data:
                return left_data['poses']
            return {}
        
        # Find common markers
        common_markers = self._find_common_markers(left_data, right_data)
        if not common_markers:
            rolog.warn("No common markers found between left and right images")
            # Fall back to mono detection from left image only if enabled
            if fallback_to_left_only:
                return left_data['poses']
            return {}
        
        rolog.stat("Process markers")
        # Process markers with stereo correction
        ret = self._process_markers_vectorized(common_markers, left_intrinsics)
        rolog.stat("done")
        rolog.warn(rolog.cycle_profile())
        return ret
    
    def _detect_markers_with_centers(
        self, 
        image: np.ndarray, 
        marker_length: float,
        intrinsics: np.ndarray, 
        distortion: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Detect markers and return poses and pixel centers.
        
        Returns:
            Dictionary with 'poses', 'centers', and 'ids' or None if no markers found
        """
        # Use the existing marker detection function
        corners, ids, rvecs, tvecs = detect_marker_poses(image, marker_length, intrinsics, distortion)
        
        if ids is None or len(ids) == 0:
            return None
        
        # Convert to poses and extract centers
        marker_poses = {}
        marker_centers = {}
        marker_ids = []
        
        for i, marker_id in enumerate(ids):
            # Create pose from rvec and tvec
            rotation = R.from_rotvec(rvecs[i].flatten())
            translation = tvecs[i].flatten()
            pose = tf.create_tf(translation, rotation)
            marker_poses[marker_id] = pose
            
            # Calculate center as mean of corners
            if len(corners) > i:
                marker_corners = corners[i].reshape((4, 2))
                center = np.mean(marker_corners, axis=0)
                marker_centers[marker_id] = center
                marker_ids.append(marker_id)
        
        return {
            'poses': marker_poses,
            'centers': marker_centers,
            'ids': marker_ids
        }
    
    def _find_common_markers(self, left_data: Dict[str, Any], right_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find markers that exist in both left and right images."""
        left_ids = set(left_data['ids'])
        right_ids = set(right_data['ids'])
        common_ids = list(left_ids.intersection(right_ids))

        rolog.debug(f"left markers: {left_ids}, right_makers: {right_ids}")
        
        if not common_ids:
            return None
        
        return {
            'ids': common_ids,
            'left_poses': {mid: left_data['poses'][mid] for mid in common_ids},
            'left_centers': np.array([left_data['centers'][mid] for mid in common_ids]),
            'right_centers': np.array([right_data['centers'][mid] for mid in common_ids])
        }
    
    def _process_markers_vectorized(self, common_markers: Dict[str, Any], intrinsics: np.ndarray) -> Dict[int, Any]:
        """Process all common markers using vectorized operations."""
        marker_ids = np.array(common_markers['ids'])
        left_centers = common_markers['left_centers']
        right_centers = common_markers['right_centers']
        left_poses = common_markers['left_poses']
        
        # Calculate disparities vectorized
        disparities = left_centers[:, 0] - right_centers[:, 0]
        
        # Filter valid disparities
        valid_mask = disparities > 0
        if not np.any(valid_mask):
            rolog.warn("No valid disparities found")
            return left_poses
        
        # Keep only valid markers
        valid_ids = marker_ids[valid_mask]
        valid_disparities = disparities[valid_mask]
        valid_left_centers = left_centers[valid_mask]
        
        # Calculate depths vectorized
        depths = self._geometry.disparity_to_depth(valid_disparities, self.baseline, self.focal_length)
        
        # Convert to 3D positions vectorized
        positions_3d = self._geometry.pixel_to_3d(valid_left_centers, depths, intrinsics)
        
        # Update poses with new positions
        corrected_poses = self._geometry.correct_poses_with_positions(left_poses, valid_ids, positions_3d)
        
        # Add uncorrected poses for markers without valid disparity
        invalid_ids = marker_ids[~valid_mask]
        for marker_id in invalid_ids:
            corrected_poses[marker_id] = left_poses[marker_id]
        
        rolog.info(f"Stereo correction applied to {len(valid_ids)} markers, {len(invalid_ids)} markers used mono detection")
        
        return corrected_poses


def split_stereo_image(stereo_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a stereo image into left and right components.
    
    Args:
        stereo_image: Stereo image with left and right images side by side
    
    Returns:
        Tuple of (left_image, right_image)
    """
    h, w = stereo_image.shape[:2]
    mid_w = w // 2
    
    left_image = stereo_image[:, :mid_w]
    right_image = stereo_image[:, mid_w:]
    
    return left_image, right_image


def detect_stereo_marker_frames(left_images_by_ts, right_images_by_ts, marker_length, 
                               left_intrinsics, left_distortion, baseline, focal_length,
                               right_intrinsics=None, right_distortion=None):
    """
    Detect marker frames using stereo images with disparity correction.
    
    Args:
        left_images_by_ts: Dictionary of left images by timestamp
        right_images_by_ts: Dictionary of right images by timestamp
        marker_length: Physical marker length
        left_intrinsics: Left camera intrinsic matrix
        left_distortion: Left camera distortion coefficients
        baseline: Stereo baseline distance
        focal_length: Camera focal length
        right_intrinsics: Right camera intrinsic matrix (optional)
        right_distortion: Right camera distortion coefficients (optional)
    
    Returns:
        Tuple of (marker_traces, frame_traces) similar to detect_marker_frames
    """
    import collections
    
    # Initialize detector
    detector = StereoMarkerDetector(baseline, focal_length)
    
    # marker_traces[marker_id] = {'ts': [...], 'tvecs': [...], 'rvecs': [...]}
    marker_traces = collections.defaultdict(lambda: collections.defaultdict(list))
    # frame_traces[ts] = {marker_id: {'rvec': ..., 'tvec': ...}}
    frame_traces = collections.OrderedDict()
    
    # Find common timestamps
    common_timestamps = set(left_images_by_ts.keys()).intersection(set(right_images_by_ts.keys()))
    
    rolog.info(f"Processing {len(common_timestamps)} stereo image pairs...")
    
    for ts in sorted(common_timestamps):
        left_image = left_images_by_ts[ts]
        right_image = right_images_by_ts[ts]
        
        # Detect markers with stereo correction
        marker_poses = detector.detect_stereo_markers(
            left_image, right_image, marker_length,
            left_intrinsics, left_distortion,
            right_intrinsics, right_distortion
        )
        
        marker_info = {}
        for marker_id, pose in marker_poses.items():
            tvec = pose.T
            rvec = pose.R.as_rotvec()
            marker_traces[marker_id]['tvecs'].append(tvec)
            marker_traces[marker_id]['rvecs'].append(rvec)
            marker_traces[marker_id]['ts'].append(ts)
            marker_info[marker_id] = {
                'rvec': rvec,
                'tvec': tvec,
            }
        frame_traces[ts] = marker_info
    
    return marker_traces, frame_traces 
