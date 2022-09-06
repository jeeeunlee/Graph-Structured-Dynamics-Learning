## MAGNETO DEFINITION CREATED BY THIS

import os 
import sys
from urdfpy import URDF

##############################################

def loadRobotURDF(fn):
    return URDF.load(fn)

def hexamagneto_base_graph(fn):
  robot = loadRobotURDF(fn)
  # nodes (v_k : Link Information)
  # mass, inertia, pos, ori
  FootNames = ['base_link','AR','BR','BL','AL']
  links = {}
  link_idx = 0
  
  for link in robot.links:
    for fn in FootNames:
      if(fn in link.name and link.inertial.mass > 1e-3 ):
        links[link.name] = link_idx
        link_idx = link_idx+1
        
  joints = {}
  joint_idx = 0
  srcsj = {}
  destj = {}

  for joint in robot.joints:
    for fn in FootNames:
      if( fn in joint.name and joint.joint_type != 'fixed' ):
        if( joint.parent in links and joint.child in links ):
          joints[joint.name] = joint_idx          
          srcsj[joint.name] = links[joint.parent]
          destj[joint.name] = links[joint.child]
          joint_idx = joint_idx+1

  print('MagnetoGraphNode=')
  print(links)
  print('MagnetoGraphEdge=')
  print(joints)
  print('MagnetoGraphEdgeSender=')
  print(srcsj)
  print('MagnetoGraphEdgeReceiver=')
  print(destj)

##############################################
#                 EXECUTION                  #
##############################################

CURRENT_DIR_PATH = os.getcwd()
# urdf_path = os.path.join( CURRENT_DIR_PATH, 'gn_inverse_dynamics/robot_graph_generator/hexamagneto/hexamagneto_simple.urdf')
urdf_path = os.path.join( CURRENT_DIR_PATH, 'gn_inverse_dynamics/robot_graph_generator/hexamagneto/hexamagneto_2_floatingbase.urdf')

hexamagneto_base_graph(urdf_path)