from sympy.vector import CoordSys3D, BodyOrienter, AxisOrienter, SpaceOrienter
from sympy import symbols
import numpy as np

def sym_frame(prev_frame, pose, name):
    T = pose.T
    rotvec = pose.R.as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        axis = np.array([0.0, 0.0 ,1.0])

    P = prev_frame
    orienter = AxisOrienter(angle, rotvec[0] * P.i + rotvec[1] * P.j + rotvec[2] * P.k)
    return P.orient_new(name, (orienter,),
                        location=T[0] * P.i + T[1] * P.j + T[2] * P.k)


"""
def create_
q1, q2, q3 = symbols('q1 q2 q3')
N = CoordSys3D('N')

body_orienter = BodyOrienter(q1, q2, q3, '123')
D = N.orient_new('D', (body_orienter, ))


from sympy.vector import AxisOrienter
orienter = AxisOrienter(q1, N.i + 2 * N.j)
B = N.orient_new('B', (orienter, ))
"""
