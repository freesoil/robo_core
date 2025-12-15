"""
tf_A_to_B: Can be interpreted in 3 ways:
1. The transformation from Frame A to Frame B.
2. The origin position of Frame B in Frame A
3. The transformation matrix to convert coordinates in Frame B to Frame A.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import rolog
import munch
from math import sin, cos, pi
import numpy_utils as nu
import skimage

def is_valid_tf(transform):
    return not (np.any(np.isnan(transform.T)) or np.any(np.isnan(transform.R.as_quat())))

def numpify_transform(translation, rotation):
    if isinstance(translation, dict):
        translation = [translation['x'], translation['y'], translation['z']]
    else:
        translation = [
            translation.x,
            translation.y,
            translation.z
        ]

    if isinstance(rotation, dict):
        rotation = [rotation['x'], rotation['y'], rotation['z'], rotation['w']]
    else:
        rotation = [rotation.x, rotation.y, rotation.z, rotation.w]

    return munch.munchify({
        'T': np.array(translation),
        'R': Rotation.from_quat(rotation)
    })

def create_tf(translation, rotation=None):
    if isinstance(translation, (list, tuple)):
        translation = np.array(translation)
    if rotation is None:
        rotation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
    translation = translation.flatten()
    assert translation.size == 3
    assert isinstance(rotation, Rotation)
    return munch.munchify({'T': np.array(translation), 'R': rotation})

def create_tf_from_eulers(axis_names, angles, degrees, pos=None):
    if pos is None:
        pos = [0., 0., 0.]
    return munch.munchify({
        'T': np.array(pos),
        'R': Rotation.from_euler(axis_names, angles, degrees=degrees)
    })


def identity_tf():
    return munch.munchify({
        'T': np.array([0.0, 0.0, 0.0]),
        'R': Rotation.from_quat([0.0, 0.0, 0.0, 1.0])})

def set_R_by_normal(z_dir, x_or_y, axis_ref):
    z_dir = nu.normalize(z_dir, axis=0)
    if x_or_y == 'x':
        y_dir = nu.normalize(np.cross(z_dir, axis_ref), axis=0)
        x_dir = np.cross(y_dir, z_dir)
    elif x_or_y == 'y':
        x_dir = nu.normalize(np.cross(axis_ref, z_dir), axis=0)
        y_dir = np.cross(z_dir, x_dir)
    else:
        assert False
    matrix = np.zeros((3, 3))
    matrix[:, 0] = x_dir
    matrix[:, 1] = y_dir
    matrix[:, 2] = z_dir
    return Rotation.from_matrix(matrix)

def set_R_by_x_axis(x_dir, other_R):
    x = other_R.as_matrix()[:, 0]
    y = other_R.as_matrix()[:, 1]
    z = other_R.as_matrix()[:, 2]

    lowest_inner = best_ref_name = best_ref_dir = None
    for name, axis in zip(['y', 'z'], [y, z]):
        inner_prod = abs(np.dot(axis, x_dir))
        if lowest_inner is None or lowest_inner > inner_prod:
            lowest_inner = inner_prod
            best_ref_name = name
            best_ref_dir = axis 

    if best_ref_name == 'y':
        z_dir = nu.normalize(np.cross(x_dir, best_ref_dir), axis=0)
        y_dir = np.cross(z_dir, x_dir)
    elif best_ref_name == 'z':
        y_dir = nu.normalize(np.cross(best_ref_dir, x_dir), axis=0)
        z_dir = np.cross(x_dir, y_dir)
    else:
        assert False

    matrix = np.zeros((3, 3))
    matrix[:, 0] = x_dir
    matrix[:, 1] = y_dir
    matrix[:, 2] = z_dir
    return Rotation.from_matrix(matrix)
    
def print_tf(tf):
    if tf is None:
        return 'Invalid TF'
    T = np.array(tf.T).flatten()
    T = T.tolist()
    # xyzw
    quat = tf.R.as_quat().tolist()
    # zxy
    eulers = tf.R.as_euler('zxy', degrees=True)

    return f'T: [{T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f}], Quat (xyzw): [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}], Euler(zxy) deg: [{eulers[0]:.1f}, {eulers[1]:.1f}, {eulers[2]:.1f}]'

def complement_tf(bridge_tf, target_tf):
    """ Figure out the axis transform going from A to B,
    given the axis tranform from A to C (target_tf) and B to C (bridge_tf).
    """
    # Say we transform the frame twice to get to a target frame.
    # In other words, for a given 3D point, its coord in target frame
    # is Xt. Its coord in the bridge frame is Xb.
    # We want to derive the base frame, which will eventually align Xb to Xt.
    # Ht * Xt = H * Hb * Xb
    # => Rt * Xt + Tt = R * (Rb * Xb + Tb) + T
    Rt = target_tf.R.as_matrix()
    Rb = bridge_tf.R.as_matrix()
    R = np.matmul(Rt, Rb.T)
    T = target_tf.T - np.matmul(R, bridge_tf.T)
    return create_tf(T, Rotation.from_matrix(R))

def relative_tf(src_tf, dest_tf):
    """ Figure out the axis transform going from B to C,
    given the axis tranform from A to C (dest_tf) and A to B (src_tf).
    """
    # src frame is A_to_B, dest frame is A_to_C
    # Rd * x + Td = Rs * (Ro * x + To) + Ts
    # Rs' * Rd * x + Rs' * (Td - Ts) = Ro * x + To
    # Ro = Rs' * Rd
    # To = Rs' * (Td - Ts)
    """
    Say Xd and Xs corresponds to the same point, but in a different frame, D and S.
    """
    # We need to derive the transform from frame S to D.
    # H * Xd = Xs
    # Hd * Xd = Hs * Xs
    # => Rd * xd + Td = Rs * xs + Ts
    # => Rs' * Rd * xd + Rs' * (Td - Ts) = xs
    src_R= src_tf.R.as_matrix()
    R = np.matmul(src_R.T, dest_tf.R.as_matrix())
    T = np.matmul(src_R.T, (dest_tf.T - src_tf.T))
    return create_tf(T, Rotation.from_matrix(R))

def diff_tfs(src_tf, dest_tf):
    offset_tf = relative_tf(src_tf, dest_tf)
    pos_err = np.linalg.norm(offset_tf.T)
    rad_err = np.linalg.norm(offset_tf.R.as_rotvec())
    return pos_err, rad_err

def is_similar_tf(src_tf, dest_tf, pos_epsilon, axis_epsilon, angle_epsilon, is_deg=False):
    pos_err, rad_err = diff_tfs(src_tf, dest_tf)
    if is_deg:
        angle_epsilon = np.deg2rad(angle_epsilon)
    return (pos_err < pos_epsilon and rad_err < angle_epsilon)


def average_tf(transforms):
    center = np.mean([transform.T for transform in transforms], axis=0)
    rotations = [transform.R.as_matrix() for transform in transforms]

    axes = [[R[:, d] for R in rotations] for d in [0, 1, 2]]
    mean_axes = [np.mean(axes[d], axis=0) for d in [0, 1, 2]]
    norm_axes = [np.linalg.norm(mean_axes[d]) for d in [0, 1, 2]]
    errors = [1.0 - norm_axes[d] for d in [0, 1, 2]]

    best_dims = np.argsort(np.array(errors))

    best_dim = best_dims[0]
    if norm_axes[best_dim] < 1e-5:
        return None
    best_axis = mean_axes[best_dim] / norm_axes[best_dim]

    next_dim = best_dims[1]
    if norm_axes[next_dim] < 1e-5:
        return None
    next_axis = mean_axes[next_dim] / norm_axes[next_dim]

    if (best_dim + 1) % 3 == next_dim:
        worst_axis = np.cross(best_axis, next_axis)
    else:
        worst_axis = np.cross(next_axis, best_axis)

    worst_axis_norm = np.linalg.norm(worst_axis)
    if worst_axis_norm < 1e-5:
        return None
    else:
        worst_axis = worst_axis / worst_axis_norm

    if (best_dim + 1) % 3 == next_dim:
        next_axis = np.cross(worst_axis, best_axis)
    else:
        next_axis = np.cross(best_axis, worst_axis)

    best_axes = [best_axis, next_axis, worst_axis]
    est_axes = [None, None, None]
    for i, axis in enumerate(best_axes):
        dim = best_dims[i]
        est_axes[dim] = axis

    est_R = np.zeros((3, 3))
    est_R[:, 0] = est_axes[0]
    est_R[:, 1] = est_axes[1]
    est_R[:, 2] = est_axes[2]
    return create_tf(center, Rotation.from_matrix(est_R))


def estimate_tf(poses, residual_pos_threshold=0.05,
                residual_angle_threshold=np.deg2rad(10),
                max_trials=100, is_degree=False):
    class TfModel(skimage.measure.fit.BaseModel):
        def __init__(self):
            self.params = None
            self._residual_pos_threshold = residual_pos_threshold
            self._residual_angle_threshold = (
                np.deg2rad(residual_angle_threshold) if is_degree
                else residual_angle_threshold)

        def estimate(self, data):
            mean_tf = average_tf(data)
            if mean_tf is None:
                return False
            self.params = mean_tf
            return True

        def residuals(self, data):
            residuals = []
            for transform in data:
                pos_err, angle_err = diff_tfs(self.params, transform)
                residual = max(
                    pos_err / self._residual_pos_threshold,
                    angle_err / self._residual_angle_threshold)
                residuals.append(residual)
            return np.array(residuals)

    assert(residual_angle_threshold >= 0.0)
    assert(residual_pos_threshold >= 0.0)
    if len(poses) == 0:
        return None, None
    elif len(poses) == 1:
        return poses[0], np.array([True], dtype=bool)
    elif len(poses) == 2:
        return average_tf(poses), np.array([True, True], dtype=bool)
    else:
        model_robust, inliers = skimage.measure.ransac(
            np.array(poses), TfModel, min_samples=min(1, len(poses)-1),
            residual_threshold=1.0, max_trials=max_trials)
        return model_robust.params, inliers

def project_frame_A2B(xyz_a, transform_B2A):
    """Given coordinates in frame A, project it to frame B"""
    shape = xyz_a.shape
    xyz_b = transform_B2A.R.apply(xyz_a.reshape((-1, 3))) + transform_B2A.T
    return xyz_b.reshape(shape)

def rotate_A2B(xyz_a, R_B2A):
    """Given coordinates in frame A, project it to frame B"""
    shape = xyz_a.shape
    xyz_b = R_B2A.apply(xyz_a.reshape((-1, 3)))
    return xyz_b.reshape(shape)

def tf_to_projection_matrix(tf):
    """
    P = [ R  T ]
        [ 0  1 ]

    To rotate a point by the origin and then translate it, or
    To project a point in the transformed frame to the original frame:
    P * [X 1]' = [RX + T, 1]
    """
    P = np.zeros((4, 4), dtype=np.float64)
    P[3, 3] = 1.0
    P[:3, 3] = tf.T
    P[:3, :3] = tf.R.as_matrix()
    return P

def projection_matrix_to_tf(P):
    T = P[:3, 3]
    R = Rotation.from_matrix(P[:3, :3])
    return munch.munchify(dict(T=T, R=R))

def axes_to_R(axis_names, vecs, normalize=False, dtype=np.float64, is_orthogonal=True):
    """
    vecs: Axis of the output frame w.r.t the map frame.
    normalize: Normalize the vecs

    R_m2o rotates the map's frame to the output's frame,
        or aligns a point coord in frame O to that in frame M.
        R_m2o is made of [X', Y', Z']

    R_m2o * X_o' = X_m'

    E.g. for X axis
    R * [1, 0, 0]' = X'
    R * [0, 1, 0]' = Y'
    R * [0, 0, 1]' = Z'
    """
    assert len(axis_names) == 3
    assert 2 <= len(vecs) <= 3

    unit_vecs = []
    for i, axis_name in enumerate(axis_names):
        if i == 2 and (len(vecs) == 2 or not is_orthogonal):
            vec = nu.normalize(
                np.cross(unit_vecs[0], unit_vecs[1]), axis=0)
            if not is_orthogonal:
                unit_vecs[1] = nu.normalize(
                    np.cross(vec, unit_vecs[0]), axis=0)
        else:
            vec = vecs[i]
        if normalize:
            vec = nu.normalize(vec, axis=0)
        unit_vecs.append(vec)

    R = np.zeros((3, 3), dtype=dtype)
    for i, axis_name in enumerate(axis_names):
        axis = 0 if axis_name == 'x' else \
               (1 if axis_name == 'y' else 2)
        R[:, axis] = unit_vecs[i]
    return Rotation.from_matrix(R)

def R_to_axes(R, scale=1.0):
    matrix = R.as_matrix() * scale
    x = matrix[:, 0]
    y = matrix[:, 1]
    z = matrix[:, 2]
    return x, y, z

def combine_tfs(*tfs):
    if len(tfs) == 0:
        return identity_tf()
    elif len(tfs) == 1:
        return tfs[-1]
    else:
        P = tf_to_projection_matrix(tfs[-1])
        for tf in tfs[-2::-1]:
            P = np.matmul(tf_to_projection_matrix(tf), P)
        return projection_matrix_to_tf(P)

def inverse_tf(tf):
    # Rx + T = y
    # R'y -R'T = x
    R_inv = tf.R.inv()
    T_inv = -R_inv.apply(tf.T)
    return munch.munchify(dict(
        T=T_inv,
        R=R_inv,
    ))

def euler_to_quaternion(roll, pitch, yaw):
    qx = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2)
    qy = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2)
    qz = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2)
    qw = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2)
    return munch.munchify(dict(x=qx, y=qy, z=qz, w=qw))


def rotation_between(pose1, pose2):
    def _rotation_matrix_between_poses(pose1, pose2):
      """Finds the rotation matrix that aligns the two poses."""
      R1 = pose1.R.as_matrix()
      R2 = pose2.R.as_matrix()

      R = R1 @ R2.T
      return R

    def _axis_of_rotation(R):
      """Finds the axis of rotation of a rotation matrix."""

      theta = np.arccos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
      if theta == 0:
        return np.array([0, 0, 1])
      else:
        v = np.array([R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0]])
        v /= np.linalg.norm(v)
        return v

    def _rotation_angle(Rm):
      """Finds the degree of rotation between two poses."""
      theta = np.arccos((Rm.trace() - 1) / 2)
      if Rm.trace() == -1:
        theta = np.pi
      return theta

    Rm = _rotation_matrix_between_poses(pose1, pose2)
    axis = _axis_of_rotation(Rm)
    angle_rad = _rotation_angle(Rm)
    p1 = pose1.T
    p2 = pose2.T
    t = (p2 - p1) @ axis
    rotation_center = p1 + t * axis
    return rotation_center, axis, angle_rad

def rotation_in_between(pose1, pose2):
    transform = relative_tf(pose1, pose2)
    # Obtain the axis and angle in the pose1 frame
    axis_angle = transform.R.as_rotvec()
    angle = np.linalg.norm(axis_angle)
    axis_pose1 = axis_angle / angle
    # Translate the axis to the global frame.
    axis = project_frame_A2B(axis_pose1, create_tf([0, 0, 0], pose1.R))

    pos1 = pose1.T
    pos2 = pose2.T
    # Find the axis along which to find the center of rotation
    extend_axis = np.cross(axis, nu.normalize(pos2 - pos1, axis=0))
    # Then, define a plane perpendicular to extend_axis and passes through the middle point.
    middle_point = (pos1 + pos2) * 0.5
    # Compute the pose1 and pose2's projection on this plane.
    #pos1_plane = pos1 - np.dot(pos1 - middle_point, extend_axis)
    #pos2_plane = pos2 - np.dot(pos2 - middle_point, extend_axis)
    plane_axis = np.cross(axis, extend_axis)
    plane_distance = np.dot(pos1 - pos2, plane_axis)
    # Extend from the middle point along the extend_axis direction up to a point where
    # the angle between the two projects are `angle`.
    extend_distance = plane_distance * 0.5 / np.tan(angle * 0.5)
    center = middle_point + extend_axis * extend_distance
    return center, axis, angle


def rotation_between_directions(A, B):
    """Get the rotation that can rotate direction A to direction B"""
    # Normalize A and B
    A_norm = A / np.linalg.norm(A)
    B_norm = B / np.linalg.norm(B)

    # Calculate the rotation axis (cross product of A and B)
    rotation_axis = np.cross(A_norm, B_norm)

    # Calculate the angle of rotation
    # Use clip to avoid floating point errors leading to invalid inputs for arccos
    cos_angle = np.dot(A_norm, B_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # Normalize the rotation axis\n",
    rotation_axis_norm = rotation_axis / np.linalg.norm(rotation_axis)

    # Construct the rotation vector (magnitude represents the angle)
    rotvec = rotation_axis_norm * angle
    return rotvec


def rotate_by_axis(src_pose, center, axis, angle):
    center = create_tf(center, Rotation.from_euler('xyz', [0, 0, 0]))
    tf_center_to_src = relative_tf(center, src_pose)
    axis_center = project_frame_A2B(center.T + axis, inverse_tf(center))
    R = Rotation.from_rotvec(axis_center * -angle)
    tf_center_to_dest = relative_tf(create_tf([0, 0, 0], R), tf_center_to_src)
    return combine_tfs(center, tf_center_to_dest)


def random_tf(x_limit, y_limit, z_limit,
              roll_deg_limit=(-180, 180),
              pitch_deg_limit=(-180, 180),
              yaw_deg_limit=(-180, 180)):
    T = np.array([
        np.random.uniform(*x_limit),
        np.random.uniform(*y_limit),
        np.random.uniform(*z_limit)])
    pitch = np.random.uniform(*pitch_deg_limit)
    roll = np.random.uniform(*roll_deg_limit)
    yaw = np.random.uniform(*yaw_deg_limit)

    # Create a rotation object using the random angles
    rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return create_tf(T, rotation)

"""
    rot = transforms.rotation_from_axis_angle(axis, angle)
    tf_center_to_src = transforms.create_tf(src_pose.T - center.T, src_pose.R)
    dest_pose = transforms.combine_tfs(
        transforms.create_tf([0.0, 0.0, 0.0], rot), tf_center_to_src)
    dest_pose.T += center.T
    return dest_pose



def rotation_from_axis_angle(axis, angle):
    ""
    Compute the rotation matrix from rotation axis and angle.

    :param axis: Rotation axis (3D vector)
    :param angle: Rotation angle in radians
    :return: 3x3 Rotation Matrix
    ""
    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    cross_product_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    Rm = (np.eye(3) +
         np.sin(angle) * cross_product_matrix +
         (1 - np.cos(angle)) * np.dot(cross_product_matrix, cross_product_matrix))
    return Rotation.from_matrix(Rm)
"""
