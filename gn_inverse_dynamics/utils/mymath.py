import numpy as np
import math
from scipy.spatial.transform import Rotation
from scipy import spatial


### MATH functions
def get_mul(a,b):
    return np.matmul(a, b)

def get_mul_T(T_list):
    Tout = np.identity(4)
    for T in T_list:
        Tout = np.matmul(Tout, T)
    return Tout

def get_diff_T(T1,T2):
    T = np.matmul(inv_T(T1), T2)
    return T

def inv_T(T):
    R = T[0:3, 0:3]
    p = T[0:3,3].tolist()

    invR = inv_R(R)
    invp = - np.matmul(invR, p)

    return T_from_R_p(invR, invp)

def inv_R(R):
    invR = np.transpose(R)
    return invR

def T_from_R_p(R,p):
    T = np.identity(4)
    T[0:3,0:3] = R
    T[0:3,3] = p
    return T

def rot_p_from_T(T):
    R, p = R_p_from_T(T)
    
    rot = Rotation.from_matrix(R)
    rotvec = rot.as_rotvec()

    return rotvec, p    

def R_p_from_T(T):
    R = T[0:3, 0:3]
    p = T[0:3,3].tolist()
    return R,p

def zyx_p_from_T(T):
    R, p = R_p_from_T(T)
    
    rot = Rotation.from_matrix(R)
    zyx = rot.as_euler('zyx', degrees=False)

    return zyx, p

def zyx_to_rot(ZYX):

    # R = Rotation.from_euler('zyx', ZYX, degrees=False).as_matrix()
    Rz = Rotation.from_euler('z', ZYX[0], degrees=False).as_matrix()
    Ry = Rotation.from_euler('y', ZYX[1], degrees=False).as_matrix()
    Rx = Rotation.from_euler('x', ZYX[2], degrees=False).as_matrix()
    R = np.matmul(np.matmul(Rz, Ry),Rx)

    return R

def quat_to_rot(quat):
  # input arg quat : w,x,y,z
  # output rotation matrix : 3x3 array
  xyzw = quat[1:4] + quat[0]
  r = spatial.transform.Rotation.from_quat(xyzw)
  return r

def quat_to_rot_axis(quat, axis):
  # quat : w,x,y,z
  # axis : 'x','y','z'
  axis_dict = {'x':0,'y':1,'z':2}
  r = quat_to_rot(quat).as_matrix()
  # print(r)
  rz = r[0:3, axis_dict[axis]]
  return rz


def normalize(arr):
    "arr : array"
    output_arr = []
    array_norm = 0
    for element in arr:
        array_norm += element**2

    if array_norm > 1e-3:
        for element in arr:
            output_arr.append(element / array_norm)

    return output_arr

def cwise_product(arr1, arr2):    
    if(len(arr1) == len(arr2)):
        output = []
        for a1, a2 in zip(arr1, arr2):
            output.append(a1*a2)
        return output
    else:
        raise ValueError('input array size is wrong')

def angle_between_axes(axis1, axis2):
    EPS = 1e-3
    if(np.linalg.norm(axis1) < EPS or np.linalg.norm(axis2) < EPS  ):
        raise ValueError('input array size is wrong')

    cos_theta = np.dot(axis1, axis2) / np.linalg.norm(axis1) / np.linalg.norm(axis2)
  
    if(cos_theta>1.0-EPS):
        return 0
    elif(cos_theta<-1.0+EPS):
        return np.pi
    else:
        return math.acos(cos_theta)


def mod_angle(angle, minmax):
    mod_min = minmax[0]
    mod_max = minmax[1]
    mod_size = mod_min - mod_max
    
    return np.remainder(angle - mod_min, mod_size)  + mod_min

def cross_product(a,b):
    return np.cross(a,b)