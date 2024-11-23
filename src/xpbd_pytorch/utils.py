from typing import Union

import torch
import matplotlib.pyplot as plt

from xpbd_pytorch.quat import rotate_vector

def plot_axis(ax: plt.Axes, x: torch.Tensor, q: torch.Tensor):
    x_axis = torch.tensor([1.0, 0.0, 0.0])
    y_axis = torch.tensor([0.0, 1.0, 0.0])
    z_axis = torch.tensor([0.0, 0.0, 1.0])

    x_rot = rotate_vector(x_axis, q)
    y_rot = rotate_vector(y_axis, q)
    z_rot = rotate_vector(z_axis, q)

    ax.quiver(x[0], x[1], x[2], x_rot[0], x_rot[1], x_rot[2], color='r')
    ax.quiver(x[0], x[1], x[2], y_rot[0], y_rot[1], y_rot[2], color='g')
    ax.quiver(x[0], x[1], x[2], z_rot[0], z_rot[1], z_rot[2], color='b')


def plot_collisions(ax: plt.Axes,
                    x: torch.tensor,
                    q: torch.tensor,
                    collisions: torch.tensor,
                    color: Union[str, tuple] = 'r'):
    if torch.isnan(collisions).all():
        return

    for collision in collisions:
        if torch.isnan(collision).all():
            continue

        # Transform collision point to world coordinates
        c = rotate_vector(collision, q) + x

        # Plot collision point
        ax.scatter(c[0], c[1], c[2], c=color, marker='x')

def plot_box_geometry(ax: plt.Axes,
                  x: torch.Tensor,
                  q: torch.Tensor,
                  dims: torch.Tensor,
                  color: Union[str, tuple] = 'b'):
        hx, hy, hz = dims[0] / 2, dims[1] / 2, dims[2] / 2
        corners = torch.tensor([[-hx, -hy, -hz],
                                [hx, -hy, -hz],
                                [hx, hy, -hz],
                                [-hx, hy, -hz],
                                [-hx, -hy, hz],
                                [hx, -hy, hz],
                                [hx, hy, hz],
                                [-hx, hy, hz]])

        rotated_corners = rotate_vector(corners, q) + x
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for edge in edges:
            start, end = rotated_corners[edge[0]], rotated_corners[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)

def plot_vectors(ax: plt.Axes, x: torch.Tensor, q: torch.Tensor, vectors: torch.Tensor, color: Union[str, tuple] = 'r'):
    for vector in vectors:
        if torch.isnan(vector).all():
            continue

        vector_world = rotate_vector(vector, q)
        ax.quiver(x[0], x[1], x[2], vector_world[0], vector_world[1], vector_world[2], color=color)
