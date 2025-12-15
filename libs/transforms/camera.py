import numpy as np
from scipy import interpolate
from .cartesian import inverse_tf, project_frame_A2B

def map_2d_to_3d(keypoints, xyz, keypoints_shape=None, method='nearest'):
    """Convert 2D keypoints to 3D keypoints given location info.

    Args:
        keypoints: N x 2, where keypoints[i][:] is the col and row of the i-th keypoint
        xyz: The xyz of each pixel
        shape: The shape of the image upon which keypoints are found. None if equals to xyz
    """
    assert(len(xyz.shape) == 3)
    h, w = xyz.shape[:2]
    x = np.arange(w)
    y = np.arange(h)

    if keypoints_shape is not None:
        kh, kw = keypoints_shape[:2]
        keypoints = keypoints.copy()
        keypoints[:, 0] *= (xyz.shape[1] - 1) / float(kw - 1)
        keypoints[:, 1] *= (xyz.shape[0] - 1) / float(kh - 1)

    # Note that the last dimension of keypoints is [x, y], revert it to have [y, x]
    new_xyz = interpolate.interpn((y, x), xyz, keypoints[:, ::-1], method=method)
    return new_xyz


def project_to_image_plane(xyz, intrinsics,
                           tf_map_to_camera=None,
                           image_shape_width_height=None):
    # tf_map_to_camera is the pose from the submap's frame to the camera's frame
    if tf_map_to_camera:
        tf_cam_to_map = inverse_tf(tf_map_to_camera)
        xyz = project_frame_A2B(xyz, tf_cam_to_map)

    assert(xyz.shape[1] == 3)

    if len(xyz.shape) > 2:
        xyz = xyz.reshape((-1, xyz.shape[-1]))

    # [x/z, y/z, 1]
    xy1z = xyz / np.expand_dims(xyz[:, 2], -1)
    # Convert it into image plane
    uv1 = np.matmul(intrinsics, xy1z.T).T
    # Crop by image
    if image_shape_width_height:
        image_width, image_height = image_shape_width_height
        mask = ((0 <= uv1[:, 0]) & (uv1[:, 0] < image_width) &
                (0 <= uv1[:, 1]) & (uv1[:, 1] < image_height))
        xyz = xyz[mask]
        uv1 = uv1[mask]

    return xyz, uv1
