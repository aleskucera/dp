from dataclasses import dataclass
from typing import Union, Tuple

import warp as wp
import numpy as np
import warp.sim.render
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from dp_utils.colors import *


class Trajectory:
    def __init__(self, name: str,
                 time: np.ndarray,
                 data: wp.array = None,
                 requires_grad: bool = False,
                 plot_line_width: int = 2,
                 render_radius: float = 0.01,
                 color: Color = RED):
        
        assert len(time) >= 2, "Time array must have at least two elements."
        if data is not None:
            assert len(data) == len(time), "Data and time arrays must have the same length."
        
        self.name = name
        self.time = time

        self.data = wp.empty(len(time), dtype=wp.vec3, requires_grad=requires_grad) if data is None else data

        self.color = color
        self.render_radius = render_radius
        self.plot_line_width = plot_line_width

    def __len__(self):
        return len(self.time)
    
    @property
    def grad(self):
        if self.data.requires_grad:
            return self.data.grad
    
    @property
    def x(self):
        return self.data.numpy()[:, 0]
    
    @property
    def y(self):
        return self.data.numpy()[:, 1]
    
    @property
    def z(self):
        return self.data.numpy()[:, 2]
    
    def update_position(self, time_step: int, q: wp.array, q_idx: int):
        assert q.dtype == wp.vec3 or q.dtype == wp.transform, "Q must be of type wp.vec3 or wp.transform."

        if q.dtype == wp.vec3:
            wp.launch(kernel=_update_trajectory_kernel_vec3, dim=1, inputs=[self.data, q, time_step, q_idx])
        else:
            wp.launch(kernel=_update_trajectory_kernel_transform, dim=1, inputs=[self.data, q, time_step, q_idx])


@wp.kernel
def _update_trajectory_kernel_vec3(trajectory: wp.array(dtype=wp.vec3), 
                                   q: wp.array(dtype=wp.vec3), 
                                   time_step: wp.int32, 
                                   q_idx: wp.int32):
    """
    Updates the trajectory array at the specified time step using a 3D vector position from `q`.

    Args:
        trajectory (wp.array): Array to store trajectory positions as wp.vec3 elements.
        q (wp.array): Array of 3D vector positions (wp.vec3).
        time_step (wp.int32): Index in `trajectory` to store the updated position.
        q_idx (wp.int32): Index in `q` to get the position.
    """
    trajectory[time_step] = q[q_idx]

@wp.kernel
def _update_trajectory_kernel_transform(trajectory: wp.array(dtype=wp.vec3), 
                                        q: wp.array(dtype=wp.transform), 
                                        time_step: wp.int32, 
                                        q_idx: wp.int32):
    """
    Updates the trajectory array at the specified time step using a translation from a transformation in `q`.

    Args:
        trajectory (wp.array): Array to store trajectory positions as wp.vec3 elements.
        q (wp.array): Array of transformations (wp.transform).
        time_step (wp.int32): Index in `trajectory` to store the updated position.
        q_idx (wp.int32): Index in `q` to get the translation position.
    """
    trajectory[time_step] = wp.transform_get_translation(q[q_idx])

# def update_trajectory(trajectory: wp.array, q: wp.array, time_step: int, q_idx: int):
#     """
#     Updates the trajectory at the specified time step with the position or translation from `q`.
#
#     Args:
#         trajectory (wp.array): Array storing the robot’s trajectory as wp.vec3 positions.
#         q (wp.array): Array containing either 3D positions (wp.vec3) or transformations (wp.transform).
#         time_step (int): Time step in the trajectory to update.
#         q_idx (int): Index in `q` to retrieve the position for updating.
#
#     Raises:
#         AssertionError: If `trajectory` is not wp.vec3 or if `q` is not wp.vec3 or wp.transform.
#     """
#     assert trajectory.dtype == wp.vec3, "Trajectory must be of type wp.vec3."
#     assert q.dtype == wp.vec3 or q.dtype == wp.transform, "Q must be of type wp.vec3 or wp.transform."
#
#     if q.dtype == wp.vec3:
#         wp.launch(kernel=_update_trajectory_kernel_vec3, dim=1, inputs=[trajectory, q, time_step, q_idx])
#     else:
#         wp.launch(kernel=_update_trajectory_kernel_transform, dim=1, inputs=[trajectory, q, time_step, q_idx])
#
# def _compute_segment_xform(p1: np.ndarray, p2: np.ndarray):
#     """
#     Computes the transformation for a capsule segment between two 3D points.
#
#     Args:
#         p1 (np.ndarray): Starting point of the segment.
#         p2 (np.ndarray): Ending point of the segment.
#
#     Returns:
#         tuple: (position (np.ndarray), rotation (np.ndarray), half_height (float)):
#             - position: Midpoint of the segment.
#             - rotation: Quaternion representing the rotation to align with the segment.
#             - half_height: Half the length of the segment.
#     """
#     position = (p1 + p2) / 2
#     height = np.linalg.norm(p2 - p1)
#     direction = (p2 - p1) / height
#     default_axis = np.array([0.0, 1.0, 0.0])
#     rotation, _ = R.align_vectors([direction], [default_axis])
#     return position, rotation.as_quat(), height / 2
#
# def render_trajectory(name: str,
#                       trajectory: Union[list, np.ndarray, wp.array],
#                       renderer: wp.sim.render.SimRenderer,
#                       radius: float = 0.1,
#                       color: tuple = (1.0, 0.0, 0.0)):
#     """
#     Renders a 3D trajectory of capsules connecting consecutive points along a path.
#
#     Args:
#         name (str): Identifier for the trajectory body in the renderer.
#         trajectory (Union[list, np.ndarray, wp.array]): List or array of 3D trajectory points.
#         renderer (wp.sim.render.SimRenderer): Renderer instance for rendering the trajectory.
#         radius (float): Radius of each capsule representing the trajectory segment. Defaults to 0.1.
#         color (tuple): RGB color of the trajectory capsules. Defaults to red (1.0, 0.0, 0.0).
#
#     Raises:
#         ValueError: If the trajectory is not a list, numpy array, or warp array.
#     """
#
#     # Check if the trajectory contains at least two points
#     if len(trajectory) < 2:
#         return
#
#     # Convert the trajectory to a numpy array if it is not already
#     if isinstance(trajectory, list):
#         trajectory = np.array(trajectory)
#     elif isinstance(trajectory, wp.array):
#         trajectory = trajectory.numpy()
#     elif isinstance(trajectory, np.ndarray):
#         pass
#     else:
#         raise ValueError("Trajectory must be a list, numpy array, or warp array.")
#
#     # Register the body with the renderer
#     renderer.register_body(name)
#
#     # Render the trajectory
#     current_pos = trajectory[0]
#     for i in range(1, len(trajectory) - 1):
#         next_pos = trajectory[i]
#
#         # Skip if the current and next positions are the same
#         if np.allclose(current_pos, next_pos, atol=1e-6):
#             continue
#
#         pos, rot, half_height = _compute_segment_xform(current_pos, next_pos)
#
#         renderer.render_capsule(name=f"c{i}",
#                                 pos=pos,
#                                 rot=rot,
#                                 radius=radius,
#                                 half_height=half_height,
#                                 color=color,
#                                 parent_body=name)
#
#         current_pos = next_pos