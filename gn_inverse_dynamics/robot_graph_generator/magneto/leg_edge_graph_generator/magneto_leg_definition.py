# magneto_leg_definition.py

# extract link whose mass > 1e-3
MagnetoGraphNode = {'base_link': 0, 
                    'AL': 1, 'AR': 2, 'BL': 3, 'BR': 4}

# exclude virtual joints
MagnetoGraphEdge = {'AL': 0, 'AR': 1, 'BL': 2, 'BR': 3}

MagnetoGraphEdgeSender = {'AL': 0, 'AR': 0, 'BL': 0, 'BR': 0}

MagnetoGraphEdgeReceiver = {'AL': 1, 'AR': 2, 'BL': 3, 'BR': 4}
