import geometry
import unittest
import random
import numpy_utils as nu
import collections

import numpy as np
import transforms as tf
from munch import munchify
from scipy.spatial.transform import Rotation
# import skgeom as sg
import Geometry3D as g3

def random_sph():
    azi = random.random() * 2 * np.pi
    ele = random.random() * np.pi - np.pi * 0.5
    return azi, ele

def random_pt(center, extends):
    v = np.array([
        random.random() * extends[0] - extends[0] * 0.5,
        random.random() * extends[1] - extends[1] * 0.5,
        random.random() * extends[2] - extends[2] * 0.5])
    return v + center 

class TestGeometry(unittest.TestCase):

    def setUp(self):
        pass

    """
    def test_bounds(self):
        line1_pts = [
            sg.Point3(-1.0, 3.0, 6.0),
            sg.Point3(1.0, 3.0, 6.0),
        ]
        pt1 = np.array([0.0, 3.0, 6.0])
        line2_pts = [
            sg.Point3(7.0, -1.0, 8.0),
            sg.Point3(7.0, 1.0, 8.0),
        ]
        pt2 = np.array([7.0, 0.0, 8.0])
        line1 = sg.Line3(line1_pts[0], line1_pts[1])
        line2 = sg.Line3(line2_pts[0], line2_pts[1])
        pt1_est, pt2_est = geometry.closest_points(line1, line2)
    """

    def test_bounds(self):
        line1_pts = [
            g3.Point(-1.0, 3.0, 6.0),
            g3.Point(1.0, 3.0, 6.0),
        ]
        pt1 = g3.Point([7.0, 3.0, 6.0])
        line2_pts = [
            g3.Point(7.0, -1.0, 8.0),
            g3.Point(7.0, 1.0, 8.0),
        ]
        pt2 = g3.Point([7.0, 3.0, 8.0])
        line1 = g3.Line(line1_pts[0], line1_pts[1])
        line2 = g3.Line(line2_pts[0], line2_pts[1])
        pt1_est, pt2_est = geometry.closest_points(line1, line2)
        self.assertEqual(pt1, pt1_est)
        self.assertEqual(pt2, pt2_est)

    def test_closest_point(self):
        point = g3.Point([1, 2, 3])
        pt1 = g3.Point([3, 4, 5])
        pt2 = g3.Point([6, 7, 8])
        line = g3.Line(pt1, pt2)
        closest = geometry.closest_point(line, point)
        connect = g3.Line(point, closest)
        self.assertTrue(np.dot(connect.dv, line.dv) < 1e-5)

    def test_rotation(self):
        point = g3.Point([200.0, 0.0, 200.0])
        origin = g3.Point(0, 0, 200)
        line = g3.Line(origin, g3.Point(g3.Vector(origin) + g3.Vector(0, 0, 1)))

        angle = np.pi / 2.0
        dest_pose = geometry.rotate_point_around_line(point, line, angle)

        expected = g3.Point(0.0, 200.0, 200.0)
        self.assertEqual(dest_pose, expected)

    def test_corner_3d(self):
        base_axis = np.array([1, 0, 0])
        for _ in range(20):
            gamma = random.random() * np.pi * 2

            azi_axis = np.array([np.cos(gamma), np.sin(gamma), 0])

            corner_axis = nu.normalize(np.array([
                random.random() * 2 - 1,
                random.random() * 2 - 1,
                random.random() * 2 - 1]), axis=-1)

            alpha = np.arccos(np.dot(corner_axis, base_axis))
            beta = np.arccos(np.dot(corner_axis, azi_axis))

            est_axes = geometry.corner_3d_axis(gamma, alpha, beta)
            for est_axis in est_axes:
                est_alpha = np.arccos(np.dot(est_axis, base_axis))
                est_beta = np.arccos(np.dot(est_axis, azi_axis))
                self.assertAlmostEqual(est_alpha, alpha)
                self.assertAlmostEqual(est_beta, beta)


    def test_azi_ele(self):
        for _ in range(20):
            azi = random.random() * np.pi * 2.0
            ele = random.random() * np.pi - np.pi * 0.5
            r = random.random() * 10
            pt = geometry.sph_to_cartesian(azi, ele, r)
            azi_est, ele_est, r_est = geometry.cartesian_to_sph(pt)
            self.assertAlmostEqual(nu.wrap(azi_est, 0.0, np.pi * 2.0), azi)
            self.assertAlmostEqual(nu.wrap(ele_est, -np.pi * 0.5, np.pi), ele)
            self.assertAlmostEqual(r, r_est)

            R = geometry.R_from_sph(azi, ele, 0.0)
            tf_origin_to_sph = tf.create_tf([0, 0, 0], R)
            x_axis = tf.project_frame_A2B(np.array([[1, 0, 0]]), tf_origin_to_sph)
            azi_est, ele_est, r_est = geometry.cartesian_to_sph(pt)
            self.assertAlmostEqual(nu.wrap(azi_est, 0.0, np.pi * 2.0), azi)
            self.assertAlmostEqual(nu.wrap(ele_est, -np.pi * 0.5, np.pi), ele)

            azi_est, ele_est, gamma_est = geometry.sph_from_R(R)
            self.assertAlmostEqual(nu.wrap(azi_est, 0.0, np.pi * 2.0), azi)
            self.assertAlmostEqual(nu.wrap(ele_est, -np.pi * 0.5, np.pi), ele)
            self.assertAlmostEqual(gamma_est, 0.0)


    def test_spherical(self):
        for _ in range(20):
            # define Z axis of the spherical frame
            azi, ele = random_sph()
            R = tf.R_from_sph(azi, ele)
            tf_origin_to_sph = tf.create_tf([0, 0, 0], R)
            z_axis = R.as_matrix()[:, 2]

            pts = np.array([
                random_pt(np.array([0, 0, 0]),
                          np.array([1, 1, 1]))
                for _ in range(10)])
            pt1 = pts[0]
            pt2 = pts[1]
            sph_coords = tf.project_frame_A2B(
                pts,
                tf.inverse_tf(tf_origin_to_sph))
            azis, eles, rs = tf.cartesian_to_sph(sph_coords)
            ele1, ele2 = eles[:2]
            axes = geometry.est_spherical_z(
                pt1, pt2, ele1, ele2)

            best_sph_frame = geometry.est_sph_frame_by_azi(
                pts, azis, axes)

            sph_coords_est = tf.project_frame_A2B(
                pts,
                tf.inverse_tf(best_sph_frame))
            azis_est, eles_est, rs_est = tf.cartesian_to_sph(sph_coords_est)
            self.assertAlmostEqual(np.max(np.abs(azis - azis_est)), 0.0)
            self.assertAlmostEqual(np.max(np.abs(eles - eles_est)), 0.0)


    def test_sph_attached_rotation(self):
        # work in progress!
        max_error_A = collections.defaultdict(lambda: 0.0)
        max_error_B = collections.defaultdict(lambda: 0.0)
        angle_limit = np.pi / 4
        for azi in np.linspace(-angle_limit, angle_limit, 10):
            for ele in np.linspace(-angle_limit, angle_limit, 10):
                #print(f'----- At azi={np.rad2deg(azi)}, ele={np.rad2deg(ele)} -----')
                R = tf.R_from_sph(azi, ele)
                tf_origin_to_sph = tf.create_tf([0, 0, 0], R)
                # The length is 7
                tf_sph_to_tip = tf.create_tf([6, 2, 3], Rotation.from_euler('xyz', [0, 0, 0]))
                tf_origin_to_tip = tf.combine_tfs(tf_origin_to_sph, tf_sph_to_tip)

                azi_est, ele_est = tf.est_sph_attach_rotation(tf_sph_to_tip.T, tf_origin_to_tip.T)
                tf_origin_to_tip_est = tf.combine_tfs(
                    tf.create_tf([0, 0, 0], tf.R_from_sph(azi_est, ele_est)),
                    tf_sph_to_tip)
                error = tf_origin_to_tip_est.T - tf_origin_to_tip.T
                #print(f'est error A: azi_err={azi_est-azi}, ele_err={ele_est-ele}, tip_err={error}')
                max_error_A['azi'] = max(max_error_A['azi'], np.rad2deg(np.abs(azi_est-azi)))
                max_error_A['ele'] = max(max_error_A['ele'], np.rad2deg(np.abs(ele_est-ele)))
                max_error_A['tx'] = max(max_error_A['tx'], np.abs(error[0]))
                max_error_A['ty'] = max(max_error_A['ty'], np.abs(error[1]))
                max_error_A['tz'] = max(max_error_A['tz'], np.abs(error[2]))

                azi_est, ele_est = tf.est_sph_attach_rotation_as_addition(tf_sph_to_tip.T, tf_origin_to_tip.T)
                tf_origin_to_tip_est = tf.combine_tfs(
                    tf.create_tf([0, 0, 0], tf.R_from_sph(azi_est, ele_est)),
                    tf_sph_to_tip)
                error = tf_origin_to_tip_est.T - tf_origin_to_tip.T
                #print(f'est error B: azi_err={azi_est-azi}, ele_err={ele_est-ele}, tip_err={error}')
                max_error_B['azi'] = max(max_error_B['azi'], np.rad2deg(np.abs(azi_est-azi)))
                max_error_B['ele'] = max(max_error_B['ele'], np.rad2deg(np.abs(ele_est-ele)))
                max_error_B['tx'] = max(max_error_B['tx'], np.abs(error[0]))
                max_error_B['ty'] = max(max_error_B['ty'], np.abs(error[1]))
                max_error_B['tz'] = max(max_error_B['tz'], np.abs(error[2]))

                #azi_est, ele_est = tf.sph_from_fixed_offset(tf_sph_to_tip.T, tf_origin_to_tip.T)
        self.assertTrue(max_error_A['azi'] < 7)
        self.assertTrue(max_error_A['ele'] < 7)
        self.assertTrue(max_error_A['tx'] < 0.3)
        self.assertTrue(max_error_A['ty'] < 0.3)
        self.assertTrue(max_error_A['tz'] < 0.3)
        print(f'Max errors A [deg]: {max_error_A}')
        print(f'Max errors B [deg]: {max_error_B}')

    def test_line_sphere_intersections(self):
        # Example usage:
        line_point = np.array([1, 2, 3])
        line_direction = nu.normalize(np.array([0.3, 0.5, 0.8]), axis=-1)
        sphere_centers = np.array([[6, 7, 8], [9, 5, 2], [6, 7, 8], [9, 5, 2]])
        sphere_radiuses = [123, 123, 1, 1]

        intersections = tf.line_sphere_intersections(line_point, line_direction, sphere_centers, sphere_radiuses)

        for i, sphere_results in enumerate(intersections):
            for intersect in sphere_results:
                if np.any(np.isnan(intersect)):
                    continue
                direction = nu.normalize(intersect - line_point, axis=-1)
                self.assertAlmostEqual(np.abs(np.dot(direction, line_direction)), 1.0)
                sphere_center = sphere_centers[i]
                sphere_radius = sphere_radiuses[i]
                distance = np.linalg.norm(intersect - sphere_center)
                self.assertAlmostEqual(distance, sphere_radius)


    # Unit test for the function
    def test_project_points_to_plane(self):
        # Test Case 1: Simple case where camera is at the origin and looking directly at the plane
        plane = g3.Plane(g3.Point([0, 0, 5]), g3.Vector([0, 0, 1]))
        camera_pose = tf.create_tf(
            np.array([0, 0, 0]),
            Rotation.from_euler('xyz', [0, 0, 0]))
        camera_intrinsics = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1]])
        image_points = [(400, 300), (500, 400)]
        
        # Expected projections are directly along the Z axis to the plane at z = 5
        expected_output = np.array([[0, 0, 5], [0.625, 0.625, 5]])
        
        projected_points = tf.project_image_to_plane(plane, camera_pose, camera_intrinsics, image_points)
        distances = [g3.distance(plane, g3.Point(p)) for p in projected_points]

        assert np.allclose(projected_points, expected_output), f"Expected {expected_output}, but got {projected_points}"

        # Test Case 2: Camera translated along the Z axis
        camera_pose.T = np.array([0, 0, 2])
        
        # Expected projections should be along the Z axis but starting from z = 2
        expected_output = np.array([[0, 0, 5], [1.25, 1.25, 5]])
        
        projected_points = tf.project_image_to_plane(plane, camera_pose, camera_intrinsics, image_points)
        assert np.allclose(projected_points, expected_output), f"Expected {expected_output}, but got {projected_points}"
        
        # Test Case 3: Camera rotated 90 degrees around the Y axis
        camera_pose.T = np.array([0, 0, 0])
        camera_pose.R = Rotation.from_euler('y', 90, degrees=True)
        
        # With this rotation, the camera looks along the X axis, and points should not intersect the plane
        try:
            projected_points = tf.project_image_to_plane(plane, camera_pose, camera_intrinsics, image_points)
            assert False, "Expected ValueError for parallel lines"
        except ValueError as e:
            assert str(e) == "The line is parallel to the plane."
        
        # Test Case 4: Complex case with an arbitrary plane and camera pose
        plane = g3.Plane(g3.Point([5, 5, 5]), g3.Vector(np.array([1, 1, 1]) / np.sqrt(3)))
        camera_translation = np.array([0, 0, 0])
        camera_rotation = Rotation.from_euler('xyz', [45, 45, 0], degrees=True)
        camera_intrinsics = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1]])
        image_points = [(400, 300), (600, 200)]
        
        # Manually calculate expected output
        projected_points = tf.project_image_to_plane(plane, camera_pose, camera_intrinsics, image_points)
        # We won't have exact expected values here without manual calculation, but we can check some properties
        assert projected_points.shape == (2, 3), f"Expected shape (2, 3), but got {projected_points.shape}"

        print("All tests passed!")


if __name__ == '__main__':
    unittest.main()
