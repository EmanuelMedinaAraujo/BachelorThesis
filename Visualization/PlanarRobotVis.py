import matplotlib.pyplot as plt
import numpy as np
import torch

# DH Parameter of 2 Link Planar Robot with extended arm (alpha, a, d, theta)
DH_EXAMPLE = torch.tensor([
    [0, 50, 0, np.pi/2],
    [0, 10, 0, 0]
])
DH_EXAMPLE_NUM_LINKS = 2

# Because the robot is planar, we can ignore d and alpha from the DH parameters

def plot_planar_robot(dh_parameter, show_link_info=True):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.axhline(0, color='black', lw=1.5)
    ax.axvline(0, color='black', lw=1.5)
    fig.set_size_inches(10,10)

    start_coordinates = np.array([0, 0])
    max_length = 0.0
    total_angle = 0.0
    for i in range(dh_parameter.shape[0]):
        link_length = dh_parameter[i, 1]
        max_length += link_length

        link_angle = dh_parameter[i, 3]
        total_angle += link_angle

        end_coordinates = start_coordinates + np.array([link_length * np.cos(total_angle), link_length * np.sin(total_angle)])

        # Plot link
        ax.plot([start_coordinates[0], end_coordinates[0]], [start_coordinates[1], end_coordinates[1]], 'r-', lw=2)
        ax.plot(start_coordinates[0], start_coordinates[1], 'bo')

        if show_link_info:
            degree = np.rad2deg(total_angle)
            ax.text(start_coordinates[0].item(), start_coordinates[1].item(), f'Joint {i + 1}\n{link_length:.2f}\n{degree:.2f}',fontsize=12)

        start_coordinates = end_coordinates

    # Ensure that the entire robot is visible by setting the limits as the maximum length of the robot
    max_length *= 1.1
    ax.set_xlim(-max_length, max_length)
    ax.set_ylim(-max_length, max_length)
    plt.show()


plot_planar_robot(DH_EXAMPLE)