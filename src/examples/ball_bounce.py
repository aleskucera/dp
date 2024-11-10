import numpy as np
import warp as wp
import warp.sim
import warp.optim
import warp.sim.render
import matplotlib.pyplot as plt

from sim import *
from dp_utils import *

USD_FILE = "data/output/ball_bounce_simple.usd"

def ball_world_model(gravity: bool = True) -> wp.sim.Model:
    if gravity:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))
    else:
        builder = wp.sim.ModelBuilder(gravity=0.0, up_vector=wp.vec3(0, 0, 1))

    b = builder.add_body(origin=wp.transform((0.5, 0.0, 1.0), wp.quat_identity()), name="ball")
    builder.add_shape_sphere(body=b, radius=0.1, density=100.0, ke=2000.0, kd=10.0, kf=200.0, mu=0.2, thickness=0.01)
    builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=0.2)
    model = builder.finalize(requires_grad=True)

    return model

@wp.kernel
def integrate_ball(
    curr_body_q: wp.array(dtype=wp.transform),
    curr_body_qd: wp.array(dtype=wp.spatial_vector),
    curr_body_f: wp.array(dtype=wp.spatial_vector),
    next_body_q: wp.array(dtype=wp.transform),
    next_body_qd: wp.array(dtype=wp.spatial_vector),
    shape_body: wp.array(dtype=wp.int32),
    rigid_contact_count: wp.array(dtype=wp.int32),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_shape0: wp.array(dtype=wp.int32),
    rigid_contact_shape1: wp.array(dtype=wp.int32),
    ball_body: BodyInfo,
    radius: wp.float32,
    restitution: wp.float32,
    dt: wp.float32
):
    # Retrieve position and velocity for the ball
    pos = wp.transform_get_translation(curr_body_q[ball_body.idx])
    vel = wp.spatial_bottom(curr_body_qd[ball_body.idx])
    lin_force = wp.spatial_bottom(curr_body_f[ball_body.idx])
    
    # Apply gravity and other forces
    next_pos = pos + vel * dt
    next_vel = vel + wp.vec3(0.0, 0.0, -9.81) * dt + lin_force * dt

    # Process each contact to determine collision responses
    contact_count = rigid_contact_count[0]  # total number of contacts
    for i in range(contact_count):
        # Check if this contact involves the ball
        if rigid_contact_shape0[i] == ball_body.idx or rigid_contact_shape1[i] == ball_body.idx:
            # Identify contact point and normal based on shape index
            if rigid_contact_shape0[i] == ball_body.idx:
                contact_point = rigid_contact_point0[i]
            else:
                contact_point = rigid_contact_point1[i]
            contact_normal = rigid_contact_normal[i]
            
            # Calculate the penetration depth and adjust the position if necessary
            penetration_depth = wp.dot(contact_normal, next_pos - contact_point) - radius
            if penetration_depth < 0.0:
                next_pos -= contact_normal * penetration_depth  # Project out of collision
                
                # Reflect velocity along the collision normal with restitution
                vel_normal = wp.dot(next_vel, contact_normal)
                if vel_normal < 0.0:
                    next_vel -= contact_normal * vel_normal * (1.0 + restitution)  # Invert with restitution

    # Write back updated position and velocity
    next_body_q[ball_body.idx] = wp.transform(next_pos, wp.transform_get_rotation(curr_body_q[ball_body.idx]))
    next_body_qd[ball_body.idx] = wp.spatial_vector(wp.spatial_top(curr_body_qd[ball_body.idx]), next_vel)


@wp.kernel
def update_trajectory_kernel(trajectory: wp.array(dtype=wp.vec3), 
                                        q: wp.array(dtype=wp.transform), 
                                        time_step: wp.int32, 
                                        q_idx: wp.int32):
    trajectory[time_step] = wp.transform_get_translation(q[q_idx])

@wp.kernel
def trajectory_loss_kernel(trajectory: wp.array(dtype=wp.vec3f), 
                            target_trajectory: wp.array(dtype=wp.vec3f), 
                            loss: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    diff = trajectory[tid] - target_trajectory[tid]
    distance_loss = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, distance_loss)


class BallBounceOptim:
    def __init__(self):

        # Simulation and rendering parameters
        self.fps = 30
        self.num_frames = 60
        self.sim_substeps = 10
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_duration = self.num_frames * self.frame_dt
        self.sim_steps = int(self.sim_duration // self.sim_dt)

        self.model = ball_world_model(gravity=True)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps)

        # self.integrator = wp.sim.SemiImplicitIntegrator()
        # self.integrator = XPBDIntegrator(enable_restitution=True, rigid_contact_relaxation=0.2)
        self.renderer = wp.sim.render.SimRenderer(self.model, USD_FILE, scaling=1.0)

        self.ball_body: BodyInfo = create_body_info("ball", self.model)

        self.loss = wp.array([0], dtype=wp.float32, requires_grad=True)
        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]
        self.target_states = [self.model.state() for _ in range(self.sim_steps + 1)]

        self.target_force = wp.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0], dtype=wp.spatial_vectorf)

        self.trajectory = wp.empty(len(self.time), dtype=wp.vec3, requires_grad=True)
        self.target_trajectory = wp.empty(len(self.time), dtype=wp.vec3)

    def _reset(self):
        self.loss = wp.array([0], dtype=wp.float32, requires_grad=True)

    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            curr_state = self.target_states[i]
            next_state = self.target_states[i + 1]
            curr_state.clear_forces()
            if i == 0:
                wp.copy(curr_state.body_f, self.target_force, dest_offset=0, src_offset=0, count=1)
            wp.sim.collide(self.model, curr_state)
            self.xpbd_simulate(curr_state, next_state, self.sim_dt)
            wp.launch(kernel=update_trajectory_kernel, dim=1, inputs=[self.target_trajectory, curr_state.body_q, i, 0])

    def forward(self, force: wp.array):
        for i in range(self.sim_steps):
            curr_state = self.states[i]
            next_state = self.states[i + 1]
            curr_state.clear_forces()
            if i == 0:
                wp.copy(curr_state.body_f, force, dest_offset=0, src_offset=0, count=1)
            wp.sim.collide(self.model, curr_state)
            self.xpbd_simulate(curr_state, next_state, self.sim_dt)
            wp.launch(kernel=update_trajectory_kernel, dim=1, inputs=[self.trajectory, curr_state.body_q, i, 0])
    
    def xpbd_simulate(self, curr_state: wp.sim.State, next_state: wp.sim.State, dt: float):
        wp.launch(
            integrate_ball,
            dim=1,
            inputs=[
                curr_state.body_q,
                curr_state.body_qd,
                curr_state.body_f,
                next_state.body_q,
                next_state.body_qd,
                self.model.shape_body,
                self.model.rigid_contact_count,
                self.model.rigid_contact_point0,
                self.model.rigid_contact_point1,
                self.model.rigid_contact_normal,
                self.model.rigid_contact_shape0,
                self.model.rigid_contact_shape1,
                self.ball_body,  # Body index for the ball
                0.1,  # Ball radius
                0.6,  # Restitution coefficient
                dt    # Time step
            ]
        )
    def step(self, force: wp.array):
        self.tape = wp.Tape()
        self._reset()
        with self.tape:
            self.forward(force)
            wp.launch(kernel=trajectory_loss_kernel, dim=len(self.trajectory),
                        inputs=[self.trajectory, self.target_trajectory, self.loss])
        self.tape.backward(self.loss)
        force_grad = force.grad.numpy()[0, 3]
        self.tape.zero()
        print(f"Gradient: {force_grad}")

        return self.loss.numpy()[0], force_grad

    def plot_loss(self):
        forces = np.linspace(0, 200, 50)
        losses = np.zeros_like(forces)
        grads = np.zeros_like(forces)

        for i, f_x in enumerate(forces):
            print(f"Iteration {i}")
            force = wp.array([[0.0, 0.0, 0.0, f_x, 0.0, 0.0]], dtype=wp.spatial_vectorf, requires_grad=True)
            losses[i], grads[i] = self.step(force)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot the loss curve
        ax1.plot(forces, losses, label="Loss")
        ax1.set_xlabel("Force")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss vs Force")
        ax1.legend()

        # Make sure that that grads are not too large
        grads = np.clip(grads, -1e4, 1e4)

        # Plot the gradient curve
        ax2.plot(forces, grads, label="Gradient", color="orange")
        ax2.set_xlabel("Force")
        ax2.set_ylabel("Gradient")
        ax2.set_title("Gradient vs Force")
        ax2.legend()

        plt.suptitle("Loss and Gradient vs Force")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def save_usd(self, fps: int = 30):
        frame_interval = 1.0 / fps  # time interval per frame
        last_rendered_time = 0.0    # tracks the time of the last rendered frame

        print("Creating USD render...")
        for t, state in zip(self.time, self.target_states):
            if t >= last_rendered_time:  # render only if enough time has passed
                self.renderer.begin_frame(t)
                self.renderer.render(state)
                self.renderer.end_frame()
                last_rendered_time += frame_interval  # update to next frame time

        self.renderer.save()


def ball_bounce_optimization():
    model = BallBounceOptim()
    model.generate_target_trajectory()
    model.plot_loss()
    model.save_usd()


if __name__ == "__main__":
    ball_bounce_optimization()