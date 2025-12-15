import unittest

import numpy_utils as nu
import numpy as np
import transforms
from munch import munchify
from scipy.spatial.transform import Rotation


class TestCartesian(unittest.TestCase):

    def setUp(self):
        pass

    def test_bounds(self):
        rot_mat = [
            [-0.19, -0.98,  0.02],
            [ 0.96, -0.18,  0.2 ],
            [-0.2,   0.06,  0.98]
        ]
        T = [0.0, 0.0, 0.0]
        tf = munchify(dict(T=T, R=Rotation.from_matrix(rot_mat)))
        x = np.array([1.0, 0.0, 0.0])
        print(tf.R.as_matrix())
        print(tf.R.apply(x))
        print(tf.R.as_matrix().dot(x))

    def test_rotation(self):
        T = np.array([200.0, 0.0, 200.0])
        R = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
        src_pose = munchify(dict(T=T, R=R))

        center = np.array([0, 0, 200])
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi / 2.0
        dest_pose = transforms.rotate_by_axis(src_pose, center, axis, angle)

        T = np.array([0.0, 200.0, 200.0])
        R = Rotation.from_euler('xyz', [0, 0, 90], degrees=True)
        est_dest_pose = munchify(dict(T=T, R=R))
        self.assertTrue(transforms.is_similar_tf(dest_pose, est_dest_pose, 0.1, 0.1, 0.1))


    def test_rotation_calc(self):
        T = np.array([100.0, 200.0, 300.0])
        R = Rotation.from_euler('xyz', [40, 50, 60], degrees=True)
        src_pose = munchify(dict(T=T, R=R))
        
        center = np.array([77, 88, 99])
        axis = nu.normalize(np.array([55, 66, 77]), axis=0)
        angle = np.deg2rad(43.21)

        dest_pose = transforms.rotate_by_axis(src_pose, center, axis, angle)
        est_center, est_axis, est_angle = transforms.rotation_in_between(src_pose, dest_pose)
        recovered_pose = transforms.rotate_by_axis(src_pose, est_center, est_axis, est_angle)
        self.assertTrue(transforms.is_similar_tf(dest_pose, recovered_pose, 0.1, 0.1, 0.1))



if __name__ == '__main__':
    unittest.main()
