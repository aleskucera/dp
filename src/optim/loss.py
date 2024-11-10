from typing import Tuple

import warp as wp
from dp_utils import Trajectory


@wp.kernel
def _trajectory_loss_kernel(trajectory: wp.array(dtype=wp.vec3f), 
                            target_trajectory: wp.array(dtype=wp.vec3f), 
                            offset: wp.int32,
                            loss: wp.array(dtype=wp.float32)):
    """Compute the L2 loss between the trajectory and the target trajectory 
       and add it to the loss array
    
    Args:
        trajectory (wp.array): The trajectory of the robot
        target_trajectory (wp.array): The target trajectory of the robot
        loss (wp.array): The loss array, should be of size 1
    """

    tid = wp.tid()
    idx = tid + offset
    diff = trajectory[idx] - target_trajectory[idx]
    distance_loss = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, distance_loss)

def add_trajectory_loss(trajectory: Trajectory, target_trajectory: Trajectory, loss: wp.array, interval: Tuple[int, int] = None):
    """
    Compute the L2 loss between the trajectory and the target trajectory and add it to the loss array.

    Args:
        trajectory (Trajectory): The trajectory of the robot.
        target_trajectory (Trajectory): The target trajectory of the robot.
        loss (wp.array): The loss array, should be of size 1.
    """
    assert loss.shape == (1,), "Loss array should be of size 1."

    if interval is None:
        offset = 0
        kernel_dim = len(trajectory)
    else:
        offset = interval[0]
        kernel_dim = interval[1] - interval[0]

    wp.launch(kernel=_trajectory_loss_kernel, dim=kernel_dim, 
              inputs=[trajectory.data, target_trajectory.data, offset, loss])
