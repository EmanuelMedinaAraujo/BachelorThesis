import math
import torch
import matplotlib.pyplot as plt
import numpy as np

FRAME_SIZE_SCALAR = 1.1
DEFAULT_LINE_WIDTH = 1.5
DEFAULT_LINE_TRANSPARENCY = 1.

def visualize_planar_robot(dh_parameter, link_accuracy=None, standard_size=False, show_title=True):
    if link_accuracy is None:
        # Create a tensor of length dh_parameter.shape[0] with DEFAULT_PROBABILITY as values
        link_accuracy = torch.full((dh_parameter.shape[0],), DEFAULT_LINE_TRANSPARENCY)
    else:
        assert dh_parameter.shape[0] == link_accuracy.shape[0], "The amount of robots arm and the amount of probabilities is not equal"

    # General settings
    fig, ax = plt.subplots()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.axhline(0, color='black', lw=1.5)
    ax.axvline(0, color='black', lw=1.5)

    if show_title :
        plt.title('Planar Robot')

    max_length = 0
    for i in range(dh_parameter.shape[0]):
        # Plot the planar robot
        arm_length = plot_planar_robot(ax, dh_parameter[i], link_accuracy[i].item())
        max_length = max(arm_length, max_length)
    # Set the plot size
    set_plot_limits(ax, max_length, standard_size)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def plot_planar_robot(ax, dh_parameter, link_accuracy):
    start_coordinates = np.array([0, 0])
    max_length = 0.0
    total_angle = 0.0
    for i in range(dh_parameter.shape[0]):
        link_length = dh_parameter[i, 1]
        max_length += link_length

        link_angle = dh_parameter[i, 3]
        total_angle += link_angle

        end_coordinates = start_coordinates + np.array(
            [link_length * math.cos(total_angle), link_length * math.sin(total_angle)])

        # Plot link
        ax.plot([start_coordinates[0], end_coordinates[0]], [start_coordinates[1], end_coordinates[1]], 'r-', lw=DEFAULT_LINE_WIDTH, alpha = link_accuracy)
        joint_label = f'$\\theta$={dh_parameter[i, 3]:.1f}, L={dh_parameter[i, 1]:.2f}'
        ax.plot(start_coordinates[0], start_coordinates[1], '-o', label=joint_label)
        start_coordinates = end_coordinates
    return max_length

def set_plot_limits(ax, max_length, standard_size):
    if standard_size:
        # Ensure that the entire robot is visible by setting the limits as the maximum length of the robot
        # Show the same size for every arm position
        max_length *= FRAME_SIZE_SCALAR
        ax.set_xlim(-max_length, max_length)
        ax.set_ylim(-max_length, max_length)
    else:
        # Only show relevant part with a robot arm
        x_max_limit = abs(max(ax.get_xlim(), key=abs)) * FRAME_SIZE_SCALAR
        ax.set_xlim(-x_max_limit, x_max_limit)

# DH Parameter of 2 Link Planar Robot (alpha, a, d, theta)
# In the planar case only a and theta are relevant
DH_EXAMPLES = torch.tensor([
    [[0, 50, 0, np.pi/2],
    [0, 10, 0, np.pi/2],
    [0, 10, 0, 0]
    ],

    [[0, 50, 0, np.pi/4],
    [0, 10, 0, np.pi/4],
    [0, 10, 0, 0]
    ],

    [[0, 50, 0, np.pi/3],
    [0, 10, 0, np.pi/3],
    [0, 10, 0, 0]
    ],
])
# between 0 and 1
accuracy= torch.tensor([0.3, 0.9, 0.5])

#visualize_planar_robot(DH_EXAMPLES)
visualize_planar_robot(DH_EXAMPLES, accuracy)