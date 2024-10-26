import warp as wp

from dp_utils import *

@wp.kernel
def add_kernel(a: wp.array(dtype=wp.vec3f),
                b: wp.array(dtype=wp.vec3f),
                c: wp.array(dtype=wp.vec3f)):
     tid = wp.tid()
     c[tid] = a[tid] + b[tid]

# @wp.kernel
# def add_position_kernel(idx: wp.int32,
#                 trajectory: wp.array(dtype=wp.vec3f),
#                 particle_q: wp.array(dtype=wp.vec3f)):
#     trajectory[idx] = particle_q[0]

@wp.kernel
def loss_kernel(trajectory: wp.array(dtype=wp.vec3f),
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


def main():
    loss = wp.zeros(1, dtype=wp.float32, device='cuda', requires_grad=True)

    # Trajectory is an array of 3D positions
    trajectory = wp.zeros(100, dtype=wp.vec3f, device='cuda', requires_grad=True)
    target_trajectory = wp.ones(100, dtype=wp.vec3f, device='cuda', requires_grad=True)

    # Particle q is an array of 3D positions
    particle_q = wp.full(2, wp.vec3f(0.5, 0.5, 0.5), device='cuda', requires_grad=True)
    t = wp.transform(wp.vec3f(0.5, 0.5, 0.5), wp.quat_identity())
    body_q = wp.full(2, t, device='cuda', requires_grad=True)
    new_particle_q = wp.ones(2, dtype=wp.vec3f, device='cuda', requires_grad=True)
    add_pos = wp.ones(2, dtype=wp.vec3f, device='cuda', requires_grad=True)
    
    tape = wp.Tape()
    with tape:
        # wp.launch(add_kernel, dim=len(particle_q), inputs=[particle_q, add_pos, new_particle_q])
        update_trajectory(trajectory=trajectory, q=particle_q, time_step=0, q_idx=0)
        update_trajectory(trajectory=trajectory, q=particle_q, time_step=1, q_idx=0)
        # if False:
        #     wp.launch(add_position_kernel, dim=1, inputs=[0, trajectory, particle_q])
        #     wp.launch(add_position_kernel, dim=1, inputs=[1, trajectory, particle_q])
        # else:
        #     wp.copy(dest=trajectory, src=particle_q, dest_offset=0, src_offset=0, count=1)
        #     wp.copy(dest=trajectory, src=particle_q, dest_offset=1, src_offset=0, count=1)
        # wp.copy(dest=trajectory, 
        #     src=new_particle_q,
        #     dest_offset=1,
        #     src_offset=0,
        #     count=1)
        wp.launch(loss_kernel, dim=len(trajectory), inputs=[trajectory, target_trajectory, loss])
    tape.backward(loss=loss)

    print(f"Loss: {loss.numpy()[0]}")
    print(f"Trajectory: {trajectory.numpy()[0]}")
    print(f"Trajectory gradient: {trajectory.grad.numpy()[0]}")
    print(f"Body q: {body_q.numpy()[0]}")
    print(f"Body q gradient: {body_q.grad.numpy()[0]}")
    print(f"Particle q: {particle_q.numpy()[0]}")
    print(f"Particle q gradient: {particle_q.grad.numpy()[0]}")
    print(f"New particle q: {new_particle_q.numpy()[0]}")
    print(f"New particle q gradient: {new_particle_q.grad.numpy()[0]}")
    
    # # Copy the first particle position to the trajectory
   
    # print(trajectory.numpy()[0])

if __name__ == "__main__":
    main()



