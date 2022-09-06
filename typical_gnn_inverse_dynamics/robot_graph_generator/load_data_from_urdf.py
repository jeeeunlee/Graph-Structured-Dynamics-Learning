from urdfpy import URDF
from geometry import axis_angle_from_rotation
from scipy.spatial.transform import Rotation as R
from typical_gnn_inverse_dynamics.utils import mymath as mymath
import numpy as np
# links, joints, actuated_joints

PRINT_COMPONENTS =  False
# PRINT_COMPONENTS =  True

def loadRobotURDF(fn):
    return URDF.load(fn)

class RobotComponent():
    '''Super class'''
    def rotmat2rpy(self, rotmat):
        r = R.from_matrix(rotmat)
        ori = r.as_euler('zyx', degrees=False)
        return ori

    def rotmat2so3(self,rotmat):
        axis, angle = axis_angle_from_rotation(rotmat)
        ori = axis.tolist()
        ori.append(angle)
        return ori

    def check_actuated(self, joint, actuatedJoints = []):
        joint_actuated = 0
        for aj in actuatedJoints:
            if( aj in joint.name ):
                joint_actuated = 1
        return joint_actuated

    def check_print(self, obj):
        if(PRINT_COMPONENTS):
            print(obj)

            
class LegData(RobotComponent):
    def __init__(self, joint, joint1, joint2):
        
        
        self.check_print(joint1.name + '- <' + joint.name + '> -' + joint2.name )
        self.check_print(self.data)

    def get_data(self):
        return self.data

