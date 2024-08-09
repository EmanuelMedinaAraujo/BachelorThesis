import torch
from Visualization.planar_robot_vis import visualize_planar_robot

# DH Parameter of 2 Link Planar Robot (alpha, a, d, theta)
# In the planar case only a and theta are relevant
DH_EXAMPLES = torch.tensor([
    [[0, 13.5994, 0, -6.],
     [0, 18.755, 0, -3.]
     ],
    [[0, 13.5994, 0, -4.8549],
     [0, 18.755, 0, -2.336]
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
visualize_planar_robot(DH_EXAMPLE, default_line_transparency=default_line_transparency,
                       default_line_width=default_line_width, frame_size_scalar=frame_size_scalar,
                       use_color_per_robot=True, goal=robot_goal)
# visualize_planar_robot(DH_EXAMPLES, accuracy, robot_goal)
