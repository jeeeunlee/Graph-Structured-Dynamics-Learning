import math
import dartpy as dart
import numpy as np

PI_HALF = math.pi / 2.0

class MagnetSimulation():
    def __init__(self, contact_node, ground_node, magnetic_force, residual_ratio):
        self.cn = contact_node
        self.set_force(magnetic_force, residual_ratio)    
        self.set_ground_config(ground_node)
        # self.gn = ground_node       

    def on(self):
        self.cn.addExtForce([0,0,-self.magnetic_force],[0,0,0], True)

    def off(self):
        self.cn.addExtForce([0,0,0], [0,0,0], True)

    def off_residual(self):
        diff_z = self._compute_diff_z()
        f = self._compute_residual_force(diff_z)
        # print("[{}] off residual force at {} = {}".format(self.cn.getName(),diff_z, f))
        self.cn.addExtForce([0,0,-f], [0,0,0], True)

    def set_force(self, force, res_ratio=None):
        self.magnetic_force = force
        if(res_ratio):
            self.residual_ratio = res_ratio
        self.residual_force = self.residual_ratio * self.magnetic_force 
 
    def set_ground_config(self, ground_node):
        self.R_ground = ground_node.getWorldTransform().matrix()[0:3,0:3]
        self.p_ground = ground_node.getWorldTransform().matrix()[0:3,3]
        self.R_gw = np.transpose(self.R_ground)
        self.p_gw = - np.matmul(self.R_gw, self.p_ground)

    def _compute_residual_force(self, diff_z):
        # f = {(a)/(a+r)}^2 * f_resmag
        a = 0.05
        f = (a)**2 / (a+diff_z)**2 * self.residual_force
        return f

    def _compute_diff_z(self):
        p_foot = self.cn.getWorldTransform().matrix()[0:3,3]
        p_foot = self.p_gw + np.matmul(self.R_gw, p_foot)

        return max(0, p_foot[2])
