from urdfpy import URDF
from geometry import axis_angle_from_rotation
from scipy.spatial.transform import Rotation as R
from gn_inverse_dynamics.utils import mymath as mymath
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
        T1 = mymath.get_diff_T(joint.origin, joint1.origin)
        T2 = mymath.get_diff_T(joint.origin, joint2.origin)

        ori1, pos1 = mymath.rot_p_from_T(T1) #zyx_p_from_T(T1)
        ori2, pos2 = mymath.rot_p_from_T(T2) #zyx_p_from_T
        self.data = list()
        self.data.extend(pos1) #3
        self.data.extend(ori1) #3
        self.data.extend(pos2) #3
        self.data.extend(ori2) #3
        
        self.check_print(joint1.name + '- <' + joint.name + '> -' + joint2.name )
        self.check_print(self.data)

    def get_data(self):
        return self.data

    def extract_data(self, idx_list = [0,2,4,6,8,10]):
        extracted_data = list()
        for extract_idx in idx_list:
            extracted_data.append(self.data[extract_idx])
        return extracted_data

class JointData(RobotComponent):
    def __init__(self, joint, actuatedJoints = []):
        JointTypeDict = {'fixed':0, 'revolute':1, 'prismatic':2, 'continuous':3}
        # actuatedJoints = ['coxa', 'femur', 'tibia']

        self.parent = joint.parent
        self.child = joint.child
        self.joint_type = joint.joint_type

        self.joint_type_idx = JointTypeDict[self.joint_type]
        self.joint_actuated = self.check_actuated(joint, actuatedJoints)
        
        self.axis = joint.axis
        self.pos = joint.origin[0:3,3].tolist()
        self.ori = self.rotmat2rpy(joint.origin[0:3,0:3])

        # stacked_informative_data        
        self.data = [self.joint_actuated]        
        self.data.extend(self.axis)
        self.data.extend(self.pos)
        self.data.extend(self.ori)
        # self.data.append = self.joint_type_idx

        self.check_print(self.data)

class JointScrewData(RobotComponent):
    def __init__(self, joint, parentlink, childlink, actuatedJoints = []):
        JointTypeDict = {'fixed':0, 'revolute':1, 'prismatic':2, 'continuous':3}
        # actuatedJoints = ['coxa', 'femur', 'tibia']

        self.joint_actuated = self.check_actuated(joint, actuatedJoints)
        self.axis = joint.axis
      
        # M_i-1,i : parent link to child link
        Tc1L1 = mymath.inv_T(parentlink.origin)
        TL1c2 = np.matmul(joint.origin, childlink.origin)
        self.Mii = np.matmul(Tc1L1, TL1c2) #Tc1c2
        self.pos = self.Mii[0:3,3].tolist()
        self.ori = self.rotmat2rpy(self.Mii[0:3,0:3])

        # Ai=(wi,vi) : joint screw expressed from child link CoM frame
        Tc2L2 = mymath.inv_T(childlink.origin)
        self.wi = np.matmul(Tc2L2[0:3,0:3], self.axis)     
        self.qi = Tc2L2[0:3,3]
        self.vi = np.cross(self.qi, self.wi)

        # stacked_informative_data        
        self.data = [self.joint_actuated]        
        self.data.extend(self.wi.tolist())
        self.data.extend(self.vi.tolist())
        self.data.extend(self.pos)
        self.data.extend(self.ori)
        # self.data.append = self.joint_type_idx

        self.check_print(self.data)


class LinkData(RobotComponent):
    def __init__(self, inertial):
        self.mass = inertial.mass
        self.inertia = [ inertial.inertia[0,0], inertial.inertia[1,1], 
                    inertial.inertia[2,2], inertial.inertia[0,1],  
                    inertial.inertia[1,2], inertial.inertia[0,2] ]
                    # Ixx, Iyy, Izz, Ixy, Iyz, Izx        
        self.pos = inertial.origin[0:3,3].tolist()
        self.ori = self.rotmat2rpy(inertial.origin[0:3,0:3])
        
        # stacked_informative_data
        self.data = [self.mass]
        self.data.extend(self.inertia)
        self.data.extend(self.pos)
        self.data.extend(self.ori)

        self.check_print(self.data)

class LinkInertiaData(RobotComponent):
    def __init__(self, inertial):
        self.mass = inertial.mass
        self.inertia = [ inertial.inertia[0,0], inertial.inertia[1,1], 
                    inertial.inertia[2,2], inertial.inertia[0,1],  
                    inertial.inertia[1,2], inertial.inertia[0,2] ]
                    # Ixx, Iyy, Izz, Ixy, Iyz, Izx        
        self.pos = inertial.origin[0:3,3].tolist()
        self.ori = self.rotmat2rpy(inertial.origin[0:3,0:3])
        
        # stacked_informative_data
        self.data = [self.mass]
        self.data.extend(self.inertia)

        self.check_print(self.data)

# for rotation invariant
class EdgeData(RobotComponent):
    def __init__(self, joint, joint1, joint2, actuatedJoints = []):
        # JointTypeDict = {'fixed':0, 'revolute':1, 'prismatic':2, 'continuous':3}
        # actuatedJoints = ['coxa', 'femur', 'tibia']

        self.joint_actuated = self.check_actuated(joint, actuatedJoints)
        self.axis = joint.axis
        
        T1 = mymath.get_diff_T(joint.origin, joint1.origin)
        T2 = mymath.get_diff_T(joint.origin, joint2.origin)

        ori1, pos1 = mymath.zyx_p_from_T(T1)
        ori2, pos2 = mymath.zyx_p_from_T(T2)

        # stacked_informative_data
        self.data = [self.joint_actuated]
        self.data.extend(self.axis)
        self.data.extend(pos1)
        self.data.extend(ori1)
        self.data.extend(pos2)
        self.data.extend(ori2)

        self.check_print(joint1.name + '- <' + joint.name + '> -' + joint2.name )
        self.check_print(self.data)

