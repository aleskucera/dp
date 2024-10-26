from typing import Union, Tuple

import warp as wp
from pxr import Gf
import numpy as np
import warp.sim.render
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

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

def update_trajectory(trajectory: wp.array, q: wp.array, time_step: int, q_idx: int):
    """
    Updates the trajectory at the specified time step with the position or translation from `q`.

    Args:
        trajectory (wp.array): Array storing the robot’s trajectory as wp.vec3 positions.
        q (wp.array): Array containing either 3D positions (wp.vec3) or transformations (wp.transform).
        time_step (int): Time step in the trajectory to update.
        q_idx (int): Index in `q` to retrieve the position for updating.
    
    Raises:
        AssertionError: If `trajectory` is not wp.vec3 or if `q` is not wp.vec3 or wp.transform.
    """
    assert trajectory.dtype == wp.vec3, "Trajectory must be of type wp.vec3."
    assert q.dtype == wp.vec3 or q.dtype == wp.transform, "Q must be of type wp.vec3 or wp.transform."

    if q.dtype == wp.vec3:
        wp.launch(kernel=_update_trajectory_kernel_vec3, dim=1, inputs=[trajectory, q, time_step, q_idx])
    else:
        wp.launch(kernel=_update_trajectory_kernel_transform, dim=1, inputs=[trajectory, q, time_step, q_idx])

def _compute_segment_xform(p1: np.ndarray, p2: np.ndarray):
    """
    Computes the transformation for a capsule segment between two 3D points.

    Args:
        p1 (np.ndarray): Starting point of the segment.
        p2 (np.ndarray): Ending point of the segment.
    
    Returns:
        tuple: (position (np.ndarray), rotation (np.ndarray), half_height (float)):
            - position: Midpoint of the segment.
            - rotation: Quaternion representing the rotation to align with the segment.
            - half_height: Half the length of the segment.
    """
    position = (p1 + p2) / 2
    height = np.linalg.norm(p2 - p1)
    direction = (p2 - p1) / height
    default_axis = np.array([0.0, 1.0, 0.0])
    rotation, _ = R.align_vectors([direction], [default_axis])
    return position, rotation.as_quat(), height / 2

def render_trajectory(name: str, 
                      trajectory: Union[list, np.ndarray, wp.array], 
                      renderer: wp.sim.render.SimRenderer, 
                      radius: float = 0.1, 
                      color: tuple = (1.0, 0.0, 0.0)):
    """
    Renders a 3D trajectory of capsules connecting consecutive points along a path.

    Args:
        name (str): Identifier for the trajectory body in the renderer.
        trajectory (Union[list, np.ndarray, wp.array]): List or array of 3D trajectory points.
        renderer (wp.sim.render.SimRenderer): Renderer instance for rendering the trajectory.
        radius (float): Radius of each capsule representing the trajectory segment. Defaults to 0.1.
        color (tuple): RGB color of the trajectory capsules. Defaults to red (1.0, 0.0, 0.0).
    
    Raises:
        ValueError: If the trajectory is not a list, numpy array, or warp array.
    """
    
    # Check if the trajectory contains at least two points
    if len(trajectory) < 2:
        return
    
    # Convert the trajectory to a numpy array if it is not already
    if isinstance(trajectory, list):
        trajectory = np.array(trajectory)        
    elif isinstance(trajectory, wp.array):
        trajectory = trajectory.numpy()
    elif isinstance(trajectory, np.ndarray):
        pass
    else:
        raise ValueError("Trajectory must be a list, numpy array, or warp array.")
    
    # Register the body with the renderer
    renderer.register_body(name)
    
    # Render the trajectory
    current_pos = trajectory[0]
    for i in range(1, len(trajectory) - 1):
        next_pos = trajectory[i]

        # Skip if the current and next positions are the same
        if np.allclose(current_pos, next_pos, atol=1e-6):
            continue
        
        pos, rot, half_height = _compute_segment_xform(current_pos, next_pos)

        renderer.render_capsule(name=f"c{i}", 
                                pos=pos,
                                rot=rot,
                                radius=radius,
                                half_height=half_height,
                                color=color,
                                parent_body=name)
        
        current_pos = next_pos

def create_3d_figure() -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a 3D figure for plotting trajectories using Matplotlib.
    
    Returns:
        plt.Figure: 3D figure for plotting trajectories.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


def update_plot(ax: plt.Axes, 
                trajectory: Union[list, np.ndarray, wp.array], 
                target_trajectory: Union[list, np.ndarray, wp.array],
                limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-1, 1), (-1, 1), (-1, 1))):
    """
    Updates the 3D plot with the robot's trajectory and the target trajectory.

    Args:
        ax (plt.Axes): Matplotlib axes for the 3D plot.
        trajectory (Union[list, np.ndarray, wp.array]): List or array of 3D trajectory points.
        target_trajectory (Union[list, np.ndarray, wp.array]): List or array of 3D target trajectory points.
    """

    if isinstance(trajectory, list):
        trajectory = np.array(trajectory)        
    elif isinstance(trajectory, wp.array):
        trajectory = trajectory.numpy()
    elif isinstance(trajectory, np.ndarray):
        pass
    else:
        raise ValueError("Trajectory must be a list, numpy array, or warp array.")
    
    if isinstance(target_trajectory, list):
        target_trajectory = np.array(target_trajectory)
    elif isinstance(target_trajectory, wp.array):
        target_trajectory = target_trajectory.numpy()
    elif isinstance(target_trajectory, np.ndarray):
        pass
    else:
        raise ValueError("Target trajectory must be a list, numpy array, or warp array.")

    # Determine the limits of the plot
    x = np.concatenate([trajectory[:, 0], target_trajectory[:, 0]])
    y = np.concatenate([trajectory[:, 1], target_trajectory[:, 1]])
    z = np.concatenate([trajectory[:, 2], target_trajectory[:, 2]])

    xlim = (min(x.min(), limits[0][0]), max(x.max(), limits[0][1]))
    ylim = (min(y.min(), limits[1][0]), max(y.max(), limits[1][1]))
    zlim = (min(z.min(), limits[2][0]), max(z.max(), limits[2][1]))

    ax.cla()
    plot_trajectory(ax, trajectory, color='red', linewidth=2)
    plot_trajectory(ax, target_trajectory, color='blue', linewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_title("3D Trajectory Plot")
    plt.draw()
    plt.pause(0.1)  # Pause to allow the plot to refresh


def plot_trajectory(ax: plt.Axes, trajectory: np.ndarray, color: str = 'red', linewidth: int = 2):
    """
    Plots a 3D trajectory using Matplotlib.

    Args:
        trajectory (np.ndarray): List or array of 3D trajectory points.
        color (str): Color for the trajectory path. Defaults to 'red'.
    """
    assert isinstance(trajectory, np.ndarray), "Trajectory must be a numpy array."
    assert trajectory.ndim == 2, "Trajectory must be a 2D array."
    assert trajectory.shape[1] == 3, "Trajectory must be a 3D array."

    # Extract X, Y, Z coordinates from the trajectory
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    ax.plot(x, y, z, color=color, marker=None, linestyle='-', linewidth=linewidth)

@wp.kernel
def _trajectory_loss_kernel(trajectory: wp.array(dtype=wp.vec3f), 
                            target_trajectory: wp.array(dtype=wp.vec3f), 
                            loss: wp.array(dtype=wp.float32)):
    """Compute the L2 loss between the trajectory and the target trajectory 
       and add it to the loss array
    
    Args:
        trajectory (wp.array): The trajectory of the robot
        target_trajectory (wp.array): The target trajectory of the robot
        loss (wp.array): The loss array, should be of size 1
    """

    tid = wp.tid()
    diff = trajectory[tid] - target_trajectory[tid]
    distance_loss = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, distance_loss)

def add_trajectory_loss(trajectory: wp.array, target_trajectory: wp.array, loss: wp.array):
    """
    Compute the L2 loss between the trajectory and the target trajectory and add it to the loss array.

    Args:
        trajectory (wp.array): The trajectory of the robot.
        target_trajectory (wp.array): The target trajectory of the robot.
        loss (wp.array): The loss array, should be of size 1.
    """
    assert trajectory.shape == target_trajectory.shape, "Trajectory and target trajectory must have the same shape."
    assert loss.shape == (1,), "Loss array should be of size 1."

    wp.launch(kernel=_trajectory_loss_kernel, dim=len(trajectory), inputs=[trajectory, target_trajectory, loss])