import os 
import sys
import time
CWD_PATH = os.getcwd()
sys.path.append(CWD_PATH)

import math
import dartpy as dart
import numpy as np
from gn_inverse_dynamics.robot_simulator.dart.magneto_functions import MagnetSimulation
from gn_inverse_dynamics.check_model.magneto.magneto_model_check_interface import MagnetoModelCheckInterface, InterfaceParams


class MagnetoWorldNode(dart.gui.osg.WorldNode): #RealTimeWorldNode
    def __init__(self, world, robot, ground):
        super(MagnetoWorldNode, self).__init__(world)
        self.robot = robot
        self.ground = ground
        self.qinit = robot.getPositions().copy()
        self.kp = 10
        self.kv = 3
        self.adof = [6,7,8,12,13,14,18,19,20,24,25,26]
        self.vdof = [0,1,2,3,4,5,9,10,11,15,16,17,21,22,23,27,28,29]
        self.contact_links = ["BL_foot_link", "AR_foot_link", "AL_foot_link", "BR_foot_link"]
        self._set_magnet_simulator()
        self._set_check_model_ineterface()
        self.first_run = True
        # print()

    def _set_check_model_ineterface(self):
        interface_params = InterfaceParams()
        interface_params.save_dir = "a_result/checkmodel"
        interface_params.base_data_path = "gn_inverse_dynamics/robot_graph_generator/magneto/magneto_simple.urdf"
        interface_params.traj_data_dir = "a_dataset/magneto/rawData_slope_combination/case6"
        interface_params.saved_model_path = "a_result/slope_combination/case5_edge64_np3/saved_model"
        interface_params.num_processing_steps = 3
        interface_params.edge_output_size = 3 
        interface_params.edge_latent_size = 64

        self.interface = MagnetoModelCheckInterface(args=interface_params)
        
        
    def _set_vdof_zero(self, joints):
        for idx in self.vdof:
            joints[idx] = 0.0
        return joints

    def _get_current_states(self):
        q = self.robot.getPositions()
        dq = self.robot.getVelocities()
        return q, dq

    def _get_current_contacts(self):
        current_contacts = list()
        for contact in self.contact_links:            
            dz = self.magnet_simulators[contact]._compute_diff_z()
            if( abs(dz) < 0.05):
                current_contacts.append(contact)
        return current_contacts

    def _set_magnet_simulator(self):
        self.magnet_simulators = dict()
        magnetic_force = 100
        residual_magnetism = 0.05
        gn = self.ground.getBodyNode("ground_link")
        for contact in self.contact_links:
            self.magnet_simulators[contact] = MagnetSimulation(
                self.robot.getBodyNode(contact), 
                gn, magnetic_force, residual_magnetism ) 
    
    def set_magnet_force(self, incontacts):
        for contact in self.magnet_simulators:
            if contact in incontacts:
                self.magnet_simulators[contact].on()
            else:
                self.magnet_simulators[contact].off_residual() #off

    # def customPreStep(self):
    #     time.sleep(0.001)

    #     q, dq = self._get_current_states() 
    #     q_err = np.subtract(q, self.qinit)        
    #     torque = -np.multiply(self.kp, q_err) - np.multiply(self.kv, dq)
    #     torque = self._set_vdof_zero(torque)
        
    #     self.robot.setForces(torque)
    #     contacts = ["AR_foot_link", "AL_foot_link", "BR_foot_link"]
    #     self.set_magnet_force(contacts)

    def customPreStep(self):
        # time.sleep(0.001)

        if(self.first_run):
            q, dq = self.interface.initialize()
            self.robot.setPositions(q)
            self.robot.setVelocities(dq)
            self.first_run = False
            
        q, dq = self._get_current_states() # 30 dof
        contacts = self._get_current_contacts() # links name
        torque, adhesions = self.interface.getCommand(q, dq, contacts)

        self.robot.setForces(torque)
        self.set_magnet_force(adhesions)



