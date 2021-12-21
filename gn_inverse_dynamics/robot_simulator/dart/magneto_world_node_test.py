import os 
import sys
import time
CWD_PATH = os.getcwd()
sys.path.append(CWD_PATH)

import math
import dartpy as dart
import numpy as np
from gn_inverse_dynamics.robot_simulator.dart.magneto_functions import MagnetSimulation


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
        # print()

    def _set_vdof_zero(self, joints):
        for idx in self.vdof:
            joints[idx] = 0.0
        return joints

    def _get_current_states(self):
        q = self.robot.getPositions()
        dq = self.robot.getVelocities()
        return q, dq

    def _set_magnet_simulator(self):
        self.magnet_simulators = dict()
        magnetic_force = 100
        residual_magnetism = 0.05
        gn = self.ground.getBodyNode("ground_link")
        for contact in self.contact_links:
            self.magnet_simulators[contact] = MagnetSimulation(
                self.robot.getBodyNode(contact), 
                gn, magnetic_force, residual_magnetism ) 
    
    def set_magnet_force(self, contacts):
        for link in self.magnet_simulators:
            if link in contacts:
                self.magnet_simulators[link].on()
            else:
                self.magnet_simulators[link].off_residual() #off

    def customPreStep(self):
        time.sleep(0.001)
        q, dq = self._get_current_states()
        q_err = np.subtract(q, self.qinit)        
        torque = -np.multiply(self.kp, q_err) - np.multiply(self.kv, dq)
        torque = self._set_vdof_zero(torque)
        
        self.robot.setForces(torque)
        contacts = ["AR_foot_link", "AL_foot_link", "BR_foot_link"]
        self.set_magnet_force(contacts)



