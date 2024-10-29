import warp as wp
from dp_utils import Trajectory


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

# def add_trajectory_loss(trajectory: wp.array, target_trajectory: wp.array, loss: wp.array):
#     """
#     Compute the L2 loss between the trajectory and the target trajectory and add it to the loss array.

#     Args:
#         trajectory (wp.array): The trajectory of the robot.
#         target_trajectory (wp.array): The target trajectory of the robot.
#         loss (wp.array): The loss array, should be of size 1.
#     """
#     assert trajectory.shape == target_trajectory.shape, "Trajectory and target trajectory must have the same shape."
#     assert loss.shape == (1,), "Loss array should be of size 1."

#     wp.launch(kernel=_trajectory_loss_kernel, dim=len(trajectory), inputs=[trajectory, target_trajectory, loss])


def add_trajectory_loss(trajectory: Trajectory, target_trajectory: Trajectory, loss: wp.array):
    """
    Compute the L2 loss between the trajectory and the target trajectory and add it to the loss array.

    Args:
        trajectory (Trajectory): The trajectory of the robot.
        target_trajectory (Trajectory): The target trajectory of the robot.
        loss (wp.array): The loss array, should be of size 1.
    """
    assert loss.shape == (1,), "Loss array should be of size 1."

    wp.launch(kernel=_trajectory_loss_kernel, dim=len(trajectory), 
              inputs=[trajectory.data, target_trajectory.data, loss])
