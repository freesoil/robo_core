"""
import skgeom as sg

def closest_points(line1: sg.Line3, line2: sg.Line3):
    dir1 = line1.direction()
    dir2 = line2.direction()
    dir3 = np.cross(dir1, dir2)
    plane1 = sg.Plane3(line1.a, dir3)
    plane2 = sg.Plane3(line2.a, dir3)
    pt2 = sg.do_intersect(line2, plane1)
    pt1 = sg.do_intersect(line1, plane2)
    return pt1, pt2
"""


from scipy.spatial.transform import Rotation
import Geometry3D as g3
import numpy_utils as nu
import numpy as np
import sympy
import transforms as tf

def clamp_scalar(v, low, high):
    return low if v < low else (high if v > high else v)

def closest_points(line1: g3.Line, line2: g3.Line):
    dir1 = nu.normalize(np.array(line1.dv._v), 0)
    dir2 = nu.normalize(np.array(line2.dv._v), 0)
    dir3 = np.cross(dir1, dir2)
    if np.linalg.norm(dir3) < 1e-4:
        return None, None
    plane1 = g3.Plane(g3.Point(line1.sv), g3.Vector(np.cross(dir3, dir1)))
    plane2 = g3.Plane(g3.Point(line2.sv), g3.Vector(np.cross(dir3, dir2)))
    pt2 = g3.intersection(plane1, line2)
    pt1 = g3.intersection(plane2, line1)
    return pt1, pt2

def closest_point(line: g3.Line, point: g3.Point) -> g3.Point:
  """Finds the point on the given line that is closest to the given point.

  Args:
    line: The line to find the closest point on.
    point: The point to find the closest point to.

  Returns:
    A g3.Point object representing the closest point on the line to the given point.
  """

  # Create a vector pointing from the given point to the start of the line.
  v = g3.Vector(point) - line.sv

  # Calculate the dot product of this vector with the direction vector of the line.
  dot_product = np.array(v._v).dot(np.array(line.dv._v))

  # Multiply the direction vector of the line by the dot product from step 2.
  closest_point_vector = line.dv * dot_product

  # Add the result from step 3 to the start of the line.
  closest_point = line.sv + closest_point_vector

  return g3.Point(closest_point)

def rotate_point_around_line(point: g3.Point, line: g3.Line, rotation_angle: float) -> g3.Point:
  """Rotates a point around a given line by a specified angle.

  Args:
    point: The point to be rotated.
    line: The line to rotate the point around.
    rotation_angle: The angle of rotation in radians.

  Returns:
    A g3.Point object representing the rotated point.
  """

  # Create a rotation matrix for the given rotation angle and the direction vector of the line.
  rotation_matrix = Rotation.from_rotvec(np.array(line.dv._v) * rotation_angle).as_matrix()

  # Multiply the rotation matrix by the vector from the point to the origin.
  rotated_point_vector = g3.Vector(rotation_matrix @ np.array((g3.Vector(point) - line.sv)._v))

  # Add the rotated point vector to the start of the line to get the new position of the rotated point.
  rotated_point = line.sv + rotated_point_vector

  return g3.Point(rotated_point)


"""
def intersect_line_with_sphere(line: g3.Line, center, radius):
    # Define the sphere (center at origin, radius 5)
    center = g3.Point(center[0], center[1], center[2])
    sphere = g3.Sphere(center, radius)

    # Find the intersection points
    intersection_points = sphere.intersection(line)
    intersection_coords = [[p.x, p.y, p.z] for p in intersection_points] if intersection_points else []
    return np.array(intersection_coords)
"""

def line_sphere_intersections(line_point, line_direction, sphere_center, sphere_radius):
    """
    Find the intersection of a line and a sphere in 3D space.

    Parameters:
    sphere_center (np.array): The center of the sphere (x, y, z).
    sphere_radius (float): The radius of the sphere.
    line_point (np.array): A point on the line (x, y, z).
    line_direction (np.array): The direction vector of the line (x, y, z).

    Returns:
    list: The points of intersection, if any.
    """
    # Calculate coefficients of the quadratic equation
    line_direction = nu.normalize(line_direction, axis=-1)
    a = 1.0
    b = 2 * np.dot(line_point - sphere_center, line_direction)
    c = np.sum((line_point - sphere_center) ** 2, axis=-1) - np.array(sphere_radius) ** 2
    # c = np.dot(line_point - sphere_center, line_point - sphere_center) - sphere_radius ** 2

    # Calculate the discriminant
    discriminant = b ** 2 - 4 * a * c

    if len(discriminant.shape) == 0:
        discriminant = np.array([discriminant])

    # No real roots, no intersection
    discriminant[discriminant < 0] = float('nan')

    # Calculate the two roots (points of intersection)
    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    # Two points of intersection
    results = np.array([line_point + np.outer(t1, line_direction), line_point + np.outer(t2, line_direction)])
    # (Number of spheres, solution id, 3D vector)
    return results.transpose(1, 0, 2)

def solve_offsets(xyz, u, v, w):
    """
    xyz: The XYZ coordinate (N, 3) for some base (doesn't have to be unit or orthogonal)
    u, v, w: The coordinate (3,) for the same base, but as a new frame.
    Returns the coordinates (N, 3) as if the same points are expressed in u, v, w.
    """
    A = np.array([u, v, w]).T
    A_inv = np.linalg.inv(A)
    offsets = np.matmul(A_inv, xyz.T).T
    return offsets

def cartesian_to_sph(pt):
    if len(pt.shape) == 1:
        r = np.linalg.norm(pt)
        azi = np.arctan2(pt[1], pt[0])
        xy = np.linalg.norm(pt[:2])
        ele = np.arctan2(pt[2], xy)
    else:
        r = np.linalg.norm(pt, axis=-1)
        azi = np.arctan2(pt[:, 1], pt[:, 0])
        xy = np.linalg.norm(pt[:, :2], axis=-1)
        ele = np.arctan2(pt[:, 2], xy)
    return azi, ele, r

def sph_to_cartesian(azi, ele, r):
    assert(not isinstance(azi, np.ndarray))
    z = r * np.sin(ele)
    xy = r * np.cos(ele)
    x = xy * np.cos(azi)
    y = xy * np.sin(azi)
    return np.array([x, y, z])

def R_from_sph(azi, ele, gamma=0):
    R = Rotation.from_euler('xyz', (gamma, -ele, azi), degrees=False)
    return R

def sph_from_R(R):
    gamma, ele, azi = R.as_euler('xyz', degrees=False)
    return azi, -ele, gamma

from sympy.vector import CoordSys3D, BodyOrienter, Point, express
def solver_sph_attachment_rotation():
    """In spherical frame, given an attachment at relatively
    (azi_attach, ele_attach, r), and then if the whole sph frame is rotated
    for (azi, ele), the attachment ends up at (azi_target, ele_target). Find the
    (azi, ele) that meets this condition.
    """
    azi_attach, ele_attach, r, azi, ele, azi_target, ele_target = sympy.symbols('azi_attach ele_attach r azi ele azi_target ele_target', real=True)

    origin_frame = CoordSys3D('origin')
    #sph_frame = origin_frame.create_new('sph', transformation='spherical')
    orienter = BodyOrienter(azi, -ele, 0.0, 'ZYX')
    rotated_frame = origin_frame.orient_new('rotated', (orienter,))

    attach_orienter = BodyOrienter(azi_attach, -ele_attach, 0.0, 'ZYX')
    attach_frame = rotated_frame.orient_new('attach', (attach_orienter,))
    target_frame = attach_frame.locate_new('target', r * attach_frame.i)

    target_pt_o = express(target_frame.position_wrt(origin_frame), origin_frame)

    orienter_target = BodyOrienter(azi_target, -ele_target, 0.0, 'ZYX')
    target_rotated_frame = origin_frame.orient_new('target_rotated', (orienter_target,))
    target_obs_frame = target_rotated_frame.locate_new('target_obs', r * target_rotated_frame.i)
    target_obs_o = express(target_obs_frame.position_wrt(origin_frame), origin_frame)

    diff = target_pt_o - target_obs_o
    eqs = [diff.dot(origin_frame.i), diff.dot(origin_frame.j), diff.dot(origin_frame.k)]

    solutions = sympy.solve(eqs, [azi, ele], dict=True)
    return solutions

def solver_approx_sph_attachment_rotation():
    """In spherical frame, given an attachment at relatively
    (azi_attach, ele_attach, r), and then if the whole sph frame is rotated
    for (azi, ele), the attachment ends up at (azi_target, ele_target). Find the
    (azi, ele) that meets this condition.

    This approximation calculates the elevation and azimuth change
    on the attachment caused by the elevation movement, then add it to the
    overall azi and ele
    """
    azi_attach, ele_attach, r, azi, ele, azi_target, ele_target = sympy.symbols('azi_attach ele_attach r azi ele azi_target ele_target', real=True)
    attach_z = r * sympy.sin(ele_attach)
    # radius at the attachment's projection at Y axis for its elevation rotation.
    r_attach_y = r * sympy.Abs(sympy.cos(azi_attach))
    ele_attach_y_old = sympy.asin(attach_z / r_attach_y)
    ele_attach_y = ele_attach_y_old + ele
    new_z = r_attach_y * sympy.sin(ele_attach_y)
    new_x = r_attach_y * sympy.cos(ele_attach_y)
    target_ele = sympy.asin(new_z / r)
    new_xy_r = sympy.sqrt(r * r - new_z * new_z)
    target_azi = azi + sympy.acos(new_x / new_xy_r)
    solutions = sympy.solve([target_ele - ele_target, target_azi - azi_target], [azi, ele], dict=True)
    result = []
    variables = (azi_attach, ele_attach, r, azi_target, ele_target)
    for sol in solutions:
        result.append({
            'azi': sympy.lambdify(variables, sol[azi]),
            'ele': sympy.lambdify(variables, sol[ele]),
        })
    return result

#est_sph_attach_rotation_solvers = solver_approx_sph_attachment_rotation()
def est_sph_attach_rotation(attach_point, target_point):
    #global est_sph_attach_rotation_solver
    # sol_azi and sol_ele is obtained from solver_approx_sph_attachment_rotation[1]
    def sol_ele(azi_attach, ele_attach, r, azi_target, ele_target):
        if isinstance(azi_target, np.ndarray) and len(azi_target) == 0:
            return np.array([])
        sin1 = np.sin(ele_attach)/np.abs(np.cos(azi_attach))
        sin2 = np.sin(ele_target)/np.abs(np.cos(azi_attach))
        sin1_clipped = np.clip(sin1, -1.0, 1.0)
        sin2_clipped = np.clip(sin2, -1.0, 1.0)
        if not np.any(np.abs(sin1 - sin1_clipped) < 0.01):
            rolog.error(f'Potential numerical error in sin1! {np.abs(sin1 - sin1_clipped)} should < 0.01')
        if not np.any(np.abs(sin2 - sin2_clipped) < 0.01):
            rolog.error(f'Potential numerical error in sin2! {np.abs(sin2 - sin2_clipped)} should < 0.01')
        return -np.arcsin(sin1_clipped) + np.arcsin(sin2_clipped)

    def sol_azi(azi_attach, ele_attach, r, azi_target, ele_target):
        if isinstance(azi_target, np.ndarray) and len(azi_target) == 0:
            return np.array([])
        sq = -np.sin(ele_target)**2/np.cos(azi_attach)**2 + 1
        sq_clipped = np.clip(sq, 0.0, None)
        cos = r*np.sqrt(sq_clipped)*np.abs(np.cos(azi_attach)/(r*np.cos(ele_target)))
        cos_clipped = np.clip(cos, -1.0, 1.0)
        if not np.any(np.abs(sq - sq_clipped) < 0.02):
            rolog.error(f'Potential numerical error in sq! {np.abs(sq - sq_clipped)} should < 0.02')
        if not np.any(np.abs(cos - cos_clipped) < 0.01):
            rolog.error(f'Potential numerical error in cos! {np.abs(cos - cos_clipped)} should < 0.01')
        return azi_target - np.arccos(cos_clipped)

    azi_attach, ele_attach, r = tf.cartesian_to_sph(attach_point)
    azi_target, ele_target, _ = tf.cartesian_to_sph(target_point)
    subs = (azi_attach, ele_attach, r, azi_target, ele_target)
    return sol_azi(*subs), sol_ele(*subs)
    #results = []
    #for solver in est_sph_attach_rotation_solvers:
    #    results.append({angle: solver[angle](*subs) for angle in ['azi', 'ele']})
    #return results

def est_sph_attach_rotation_as_addition(attach_point, target_point):
    """In spherical frame, given an attachment at relatively
    (azi_attach, ele_attach, r), and then if the whole sph frame is rotated
    for (azi, ele), the attachment ends up at (azi_target, ele_target). Find the
    (azi, ele) that meets this condition.

    This approximate solve assumes the solution is to add the two
    azi and ele angles from the attachment's offset and the rotation.
    """
    azi_attach, ele_attach, _ = tf.cartesian_to_sph(attach_point)
    azi_target, ele_target, _ = tf.cartesian_to_sph(target_point)
    ele = ele_target - ele_attach
    azi = azi_target - azi_attach
    return azi, ele

def corner_3d_axis(gamma, alpha, beta):
    """Given two axis, x-axis and another axis on the XY plane with gamma angle to
    the x-axis, find the 3D axis that are alpha-angle away from x-axis
    and beta-angle away from the other axis.
    Note all angles are in radians.
    See https://drive.google.com/file/d/1XkZlv4JYw02KD4JsM_bG-m4cpWf9lDca/view?usp=drive_link
    """
    AC = np.cos(alpha)
    AB = AC / np.cos(gamma)
    AE = np.cos(beta)
    BE = AB - AE
    BF = BE / np.cos(np.pi * 0.5 - gamma)
    BC = np.cos(alpha) * np.tan(gamma)
    CF = BC - BF
    AF = np.sqrt(AC * AC + CF * CF)
    AD = 1.0
    if AD*AD - AF*AF < 0:
        # DEBUG: why would this happen
        return []
    DF = np.sqrt(AD * AD - AF * AF)
    if DF != 0:
        return [
            nu.normalize(np.array([AC, CF, DF]), axis=-1),
            nu.normalize(np.array([AC, CF, -DF]), axis=-1),
        ]
    else:
        return [nu.normalize(np.array([AC, CF, DF]), axis=-1)]

def triangle_cos_ABC(AB, BC, CA):
    return np.arccos((BC * BC + AB * AB - CA * CA) / (2. * BC * AB))

def est_sph_frame_by_azi(points, azis_prior, z_dirs):
    """Given a set of points and their corresponding expected azimuth,
       and a list of z_dir candidates, find the best spherical frame
       that meets the expected azis.
    """
    def _est_spherical_x(points, azis_prior, z_dir):
        """Find the frame with z_dir that best reflects the expected azimuths provided."""
        # Identify a good seed.
        seed_x_candidates = np.array([
            [1, 0, 0],
            [0, 1, 0],
        ])
        cross_amplitude = np.linalg.norm(np.cross(seed_x_candidates, z_dir), axis=-1)
        best_index = np.argmax(cross_amplitude)
        seed_x_axis = seed_x_candidates[best_index]

        # Get azimuth error
        seed_R = tf.set_R_by_normal(z_dir, 'x', seed_x_axis)
        tf_origin_to_seed_sph = tf.create_tf([0, 0, 0], seed_R)
        points_sph = tf.project_frame_A2B(points, tf.inverse_tf(tf_origin_to_seed_sph))
        azis_est, eles_est, rs = cartesian_to_sph(points_sph)
        
        errors = nu.wrap_autobase(azis_prior - azis_est)
        mean_azi_offset = np.mean(errors)
        azi_R = Rotation.from_euler('z', -mean_azi_offset, degrees=False)
        sph_frame = tf.combine_tfs(
            tf_origin_to_seed_sph, tf.create_tf([0, 0, 0], azi_R))
        # TOOD: return only the R part since t is always 0
        return sph_frame, errors - mean_azi_offset

    best_sph_frame = None
    best_error = float('inf')
    for z_dir in z_dirs:
        sph_frame, errors = _est_spherical_x(
            points, azis_prior, z_dir)
        error = np.mean(np.abs(errors))
        if error < best_error:
            best_sph_frame = sph_frame
            best_error = error

    return best_sph_frame

def est_spherical_z(point0, point1, ele0, ele1):
    """Given two points (B for pt0, C for pt1), and their corresponding
    elevation, find the Z axis of the spherical frame.
    https://drive.google.com/file/d/1pEyPr9nv9C_fJ5fZ5u7m-TAcUkZJQVq_/view?usp=drive_link

    If ele0 ~= ele1, the Z dir may be one of the flips.
    """
    # Define the local frame
    x_l = nu.normalize(point1 - point0, axis=-1)
    z_l = nu.normalize(np.cross(x_l, -point0), axis=-1)
    R = tf.set_R_by_normal(z_l, 'x', x_l)
    tf_origin_to_localB = tf.create_tf(
        point0, R)

    # Solve for z_axis in the above local frame
    AB = np.linalg.norm(point0)
    AC = np.linalg.norm(point1)
    BD = AB * np.sin(ele0)
    CE = AC * np.sin(ele1)
    BC = np.linalg.norm(point0 - point1)

    if BC < 1e-5:
        return np.zeros((0, 3))

    height_diff = BD - CE
    # angle(CBD)
    alpha = np.arccos(np.clip(height_diff / BC, -1.0, 1.0))
    # angle(ABD)
    beta = np.pi * 0.5 - ele0
    # angle(ABC)
    gamma = triangle_cos_ABC(AB, BC, AC)
    # Z-axis for spherical frame, assuming vec(BC) is along x-axis
    normals_at_B = corner_3d_axis(gamma, alpha, beta)

    height_diff = CE - BD
    # angle(BCE)
    alpha = np.pi - np.arccos(np.clip(height_diff / BC, -1.0, 1.0))
    # angle(ACE)
    beta = np.pi * 0.5 - ele1
    # angle(ACB)
    gamma = np.pi - triangle_cos_ABC(BC, AC, AB)
    # Z-axis for spherical frame, assuming vec(BC) is along x-axis
    normals_at_C = corner_3d_axis(gamma, alpha, beta)

    normals_in_localB = []
    for normal_B in normals_at_B:
        is_shared = False
        for normal_C in normals_at_C:
            if (1.0 - abs(np.dot(normal_B, normal_C))) < 1e5:
                is_shared = True
        if not is_shared:
            continue
        normals_in_localB.append(normal_B)

    if len(normals_in_localB) == 0:
        return np.zeros((0, 3))
    normals_in_origin = tf.project_frame_A2B(np.array(normals_in_localB),
                                             tf.create_tf([0, 0, 0], tf_origin_to_localB.R))
    # flip: normals are computed from pt to the sph base plane, it should
    # actually go from the sph base plane to the pt
    normals_in_origin = -normals_in_origin
    return normals_in_origin


def sym_corner_3d_axis():
    """ NOT working! """
    x0, y0, z0 = sympy.symbols('x0 y0 z0')
    x1, y1, z1 = sympy.symbols('x1 y1 z1')
    x2, y2, z2 = sympy.symbols('x2 y2 z2')
    eq0 = sympy.Eq(x0 * x0 + y0 * y0 + z0 * z0, 1.0)
    eq1 = sympy.Eq(x1 * x1 + y1 * y1 + z1 * z1, 1.0)
    eq2 = sympy.Eq(x2 * x2 + y2 * y2 + z2 * z2, 1.0)
    alpha, beta = sympy.symbols('alpha beta')
    eq3 = sympy.Eq(x0 * x2 + y0 * y2 + z0 * z2, sympy.cos(alpha))
    eq4 = sympy.Eq(x1 * x2 + y1 * y2 + z1 * z2, sympy.cos(beta))

    solutions = sympy.solve([eq0, eq1, eq2, eq3, eq4], [x2, y2, z2], dict=True)
    sol_funcs = []
    arguments = [x0, y0, z0, x1, y1, z1, alpha, beta]
    for sol in solutions:
        sol_x2 = sympy.lambdify(arguments, sol[x2].simplify())
        sol_y2 = sympy.lambdify(arguments, sol[y2].simplify())
        sol_z2 = sympy.lambdify(arguments, sol[z2].simplify())
        def corner_3d_axis(axis0, axis1, alpha, beta):
            axis0 = nu.normalize(axis0, axis=-1)
            axis1 = nu.normalize(axis1, axis=-1)
            arguments = axis0.tolist() + axis1.tolist() + [alpha, beta]
            return np.array([sol_x2(*arguments),
                             sol_y2(*arguments),
                             sol_z2(*arguments)])
        sol_funcs.append(corner_3d_axis)
    def all_sols(axis0, axis1, alpha, beta):
        results = []
        for f in sol_funcs:
            try:
                result = f(axis0, axis1, alpha, beta)
            except:
                pass
            else:
                results.append(result)
        return results
    return all_sols


def locations_along_axes(start_pos_g, vel_g,
                           lateral_dir_g, vertical_dir_g, *xyz_gs):
    """Distance along the gantry directions. "_a" means gantry frame.
    """
    longitude_dir_g = nu.normalize(vel_g, axis=0)
    return [solve_offsets(xyz_g - start_pos_g, longitude_dir_g, lateral_dir_g, vertical_dir_g)
            for xyz_g in xyz_gs]

#def project_image_to_grid(grid_base, voxel_size, camera_pose, camera_info, image_pts):
def project_image_to_plane(plane, camera_pose, camera_intrinsics, image_points):
    """
    Project 2D image points onto a 3D plane in a vectorized way.

    :param plane: Geometry3D.Plane object representing the 3D plane.
    :param camera_translation: numpy array of shape (3,) representing the camera translation.
    :param camera_rotation: scipy Rotation object representing the camera rotation.
    :param camera_intrinsics: numpy array of shape (3, 3) representing the camera intrinsics matrix.
    :param image_points: list of (u, v) tuples representing the image coordinates.
    :return: numpy array of shape (N, 3) where N is the number of image points, representing the 3D coordinates of the projected points on the plane.
    """

    # Convert camera_rotation to a rotation matrix
    R_matrix = camera_pose.R.as_matrix()

    # Invert the intrinsics matrix
    K_inv = np.linalg.inv(camera_intrinsics)

    # Convert image points to numpy array and add homogeneous coordinate
    image_points = np.array(image_points)
    pixel_homogeneous = np.hstack((image_points, np.ones((image_points.shape[0], 1))))

    # Convert pixel coordinates to camera coordinates
    # K @ camera_coords = [px, py, 1]'
    # [fx, 0, cx]' * [x/z, y/z, 1] = fx * x / z + cx = px
    # x/z = (px - cx) / fx
    # camera_coords = [x/z, y/z, 1]
    camera_coords = K_inv @ pixel_homogeneous.T

    # Transform to world coordinates
    world_coords = (R_matrix @ camera_coords).T + camera_pose.T

    # Direction vectors from camera to points in world coordinates
    directions = world_coords - camera_pose.T

    # Plane equation components
    point, normal = plane.point_normal()
    plane_normal = np.array(normal._v)
    plane_point = np.array(point._v)

    # Calculate intersections with the plane
    ndotu = plane_normal.dot(directions.T)
    w = camera_pose.T - plane_point
    si = -plane_normal.dot(w) / ndotu
    intersection_points = camera_pose.T + (si[:, np.newaxis] * directions)

    return intersection_points

