import torch

from vis.analytical_vis import visualize_analytical_planar_robot

"""
This script is used to rapidly debug the visualization of the planar robot with various options. 
"""

# DH Parameter of 2 Link Planar Robot (alpha, a, d, theta)
# In the planar case only a and theta are relevant
DH_EXAMPLES = torch.tensor([
    [[0, 13.5994, 0, -6.],
     [0, 18.755, 0, -3.]
     ],
    [[0, 13.5994, 0, -4.8549],
     [0, 18.755, 0, -2.336]
     ],
[[0, 13.5994, 0, -4],
     [0, 18.755, 0, -2]
     ],[[0, 13.5994, 0, -3],
     [0, 18.755, 0, -4]
     ],[[0, 13.5994, 0, -5],
     [0, 18.755, 0, -6]
     ],[[0, 13.5994, 0, -7],
     [0, 18.755, 0, -8]
     ],[[0, 13.5994, 0, -9],
     [0, 18.755, 0, -1]
     ],[[0, 13.5994, 0, -0],
     [0, 18.755, 0, -1]
     ],[[0, 13.5994, 0, -4.],
     [0, 18.755, 0, -2.1]
     ],[[0, 13.5994, 0, -4.5],
     [0, 18.755, 0, -2.8]
     ],[[0, 13.5994, 0, -4.2],
     [0, 18.755, 0, -2.1]
     ],[[0, 13.5994, 0, -4.4],
     [0, 18.755, 0, -2.9]
     ],[[0, 13.5994, 0, -4.6],
     [0, 18.755, 0, -2.8]
     ],[[0, 13.5994, 0, -4.7],
     [0, 18.755, 0, -2.4]
     ],[[0, 13.5994, 0, -4.8],
     [0, 18.755, 0, -2.3]
     ],[[0, 13.5994, 0, -4.1],
     [0, 18.755, 0, -2.2]
     ],
])

DH_EXAMPLE = torch.tensor(
    [[0, 13.5994, 0, -4.8549],
     [0, 18.755, 0, -2.336]
     ]
)
# between 0 and 1
# accuracy= torch.tensor([0.5, 0.3, 0.9])
robot_goal = torch.tensor([13.4757, -1.3202])

default_line_transparency = 1.
frame_size_scalar = 1.1
default_line_width = 1.5

# visualize_planar_robot(DH_EXAMPLES)
visualize_analytical_planar_robot(DH_EXAMPLES,goal=robot_goal, use_gradual_transparency=True)
# visualize_planar_robot(DH_EXAMPLES, accuracy, robot_goal)
