import os 
import sys
CWD_PATH = os.getcwd()
sys.path.append(CWD_PATH)

import math
import dartpy as dart
import numpy as np
from gn_inverse_dynamics.robot_simulator.dart.magneto_world_node import MagnetoWorldNode
from gn_inverse_dynamics.robot_simulator.dart.magneto_functions import MagnetSimulation

PI_HALF = math.pi / 2.0

#ground: ground/ground_climbing.urdf
#ground: ground/ground_inclined.urdf
#ground: ground/ground_inclined02.urdf
#ground: ground/ground_inclined03.urdf
#ground: ground/ground_3d.urdf

def _set_friction_coeff(ground, robot, mu):
    ground.getBodyNode("ground_link").setFrictionCoeff(mu)
    robot.getBodyNode("BL_foot_link").setFrictionCoeff(mu)
    robot.getBodyNode("AL_foot_link").setFrictionCoeff(mu)
    robot.getBodyNode("AR_foot_link").setFrictionCoeff(mu)
    robot.getBodyNode("BR_foot_link").setFrictionCoeff(mu)

    robot.getBodyNode("BL_foot_link_3").setFrictionCoeff(mu)
    robot.getBodyNode("AL_foot_link_3").setFrictionCoeff(mu)
    robot.getBodyNode("AR_foot_link_3").setFrictionCoeff(mu)
    robot.getBodyNode("BR_foot_link_3").setFrictionCoeff(mu)
    
def _set_init_robot_config(robot): 
    # dofs = robot.getNumDofs()
    # q = [ robot.getPosition(i) for i in range(dofs) ]    
    # _base_joint =  robot.getDof("_base_joint").getIndexInSkeleton()
    # AL_coxa_joint = robot.getDof("AL_coxa_joint").getIndexInSkeleton()
    # AL_femur_joint = robot.getDof("AL_femur_joint").getIndexInSkeleton()
    # AL_tibia_joint = robot.getDof("AL_tibia_joint").getIndexInSkeleton()
    # AL_foot_joint_1 = robot.getDof("AL_foot_joint_1").getIndexInSkeleton()
    # AL_foot_joint_2 = robot.getDof("AL_foot_joint_2").getIndexInSkeleton()
    # AL_foot_joint_3 = robot.getDof("AL_foot_joint_3").getIndexInSkeleton()
    # print("joint check  (AL) : {}, {}, {}, {}, {}, {}".format(
    #                         AL_coxa_joint, AL_femur_joint, AL_tibia_joint,
    #                         AL_foot_joint_1, AL_foot_joint_2, AL_foot_joint_3))
    
    femur_init = -1.0/10.0*PI_HALF
    tibia_init = -PI_HALF-femur_init
    q_foot_init = [0.0, femur_init, tibia_init, 0.0, 0.0, 0.0 ]

    q = [0.0, 0.0, 0.15 , 0.0, 1.0, 0.0] # base init pose 
    q.extend(q_foot_init*4) #AL,AR,BL, BR
    print(q)
    robot.setPositions(q)


if __name__ == "__main__":

    ## GENERATE WORLD AND ADD SKELETONS
    world = dart.simulation.World()
    urdfParser = dart.utils.DartLoader()
    robot = urdfParser.parseSkeleton( CWD_PATH + 
        "/gn_inverse_dynamics/robot_simulator/config/magneto/MagnetoSim_Dart.urdf")
    ground = urdfParser.parseSkeleton( CWD_PATH + 
        "/gn_inverse_dynamics/robot_simulator/config/ground/magneto/ground_inclined_1rad.urdf")
    world.addSkeleton(robot)
    world.addSkeleton(ground)
    world.setGravity([0, 0, -9.81])
    world.setTimeStep(0.001 )

    ## INITIAL CONFIGURATIONS
    _set_init_robot_config(robot)
    _set_friction_coeff(ground, robot, 0.5)

    ## WRAP A WORLD NODE
    node = MagnetoWorldNode(world, robot, ground)
    node.setNumStepsPerCycle(10)

    # Create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)

    # Viewer settings
    viewer.setUpViewInWindow(1440, 0, 750, 750)
    scale=2
    viewer.setCameraHomePosition([scale*0.6, 5*0.6, scale*0.6],
                                [0.3, -0.0, 0.2],
                                [0.0, 0.0, 2.0])
    
    viewer.run()
 