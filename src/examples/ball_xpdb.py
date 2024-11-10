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

    b1 = builder.add_body(origin=wp.transform((0.5, 0.0, 1.0), wp.quat_identity()), name="ball_1")
    builder.add_shape_sphere(body=b1, radius=0.1, density=100.0, ke=2000.0, kd=10.0, kf=200.0, mu=0.2, thickness=0.01)

    b2 = builder.add_body(origin=wp.transform((0.5, 0.0, 2.0), wp.quat_identity()), name="ball_2")
    builder.add_shape_sphere(body=b2, radius=0.1, density=100.0, ke=2000.0, kd=10.0, kf=200.0, mu=0.2, thickness=0.01)

    builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=0.2)
    model = builder.finalize(requires_grad=True)

    return model

@wp.kernel
def integrate_body(
    body: BodyInfo,
    dt: wp.float32,
    curr_body_q: wp.array(dtype=wp.transform),
    curr_body_qd: wp.array(dtype=wp.spatial_vector),
    curr_body_f: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=wp.float32),
    body_inertia: wp.array(dtype=wp.mat3),
    next_body_q: wp.array(dtype=wp.transform),
    next_body_qd: wp.array(dtype=wp.spatial_vector)
):
    """Integrate the body state.
    
    v_next = v_curr + (f / m) * dt
    x_next = x_curr + v_next * dt

    omega_next = omega_curr + I^{-1} * (torque - omega_curr x (I * omega_curr)) * dt
    q_next = q_curr + 0.5 * q_curr * [omega_next, 0] * dt
    q_next = q_next / ||q_next||
    """

    # Retrieve position, velocity, force, mass, and inertia for the body
    pos = wp.transform_get_translation(curr_body_q[body.idx])
    vel = wp.spatial_bottom(curr_body_qd[body.idx])
    rot = wp.transform_get_rotation(curr_body_q[body.idx])
    ang_vel = wp.spatial_top(curr_body_qd[body.idx])
    force = wp.spatial_bottom(curr_body_f[body.idx])
    torque = wp.spatial_top(curr_body_f[body.idx])
    mass = body_mass[body.idx]
    inertia = body_inertia[body.idx]

    gravity = wp.vec3(0.0, 0.0, -9.81)

    lin_accel = (force + gravity) / mass
    next_vel = vel + lin_accel * dt
    next_pos = pos + next_vel * dt

    ang_accel = wp.mat3_inv(inertia) @ (torque - wp.cross(ang_vel, inertia @ ang_vel))
    next_ang_vel = ang_vel + ang_accel * dt
    next_rot = rot + 0.5 * rot *  wp.quaternion(next_ang_vel[0], next_ang_vel[1], next_ang_vel[2], 0) * dt
    next_rot = next_rot / wp.norm(next_rot)

    next_body_q[body.idx] = wp.transform(next_pos, next_rot)
    next_body_qd[body.idx] = wp.spatial_vector(next_ang_vel, next_vel)

@wp.kernel
def solve_ground_collisions(
    balls: wp.array(dtype=BodyInfo),
    body_q: wp.array(dtype=wp.transform)):
    ball = balls[wp.tid()]
    pos = wp.transform_get_translation(body_q[ball.idx])
    if pos[2] - 0.1 < 0.0:
        pos[2] = 0.0 + 0.1
    body_q[ball.idx] = wp.transform(pos, wp.transform_get_rotation(body_q[ball.idx]))

@wp.kernel
def solve_distance_constraint(
    balls: wp.array(dtype=BodyInfo),
    body_q: wp.array(dtype=wp.transform),
    distance: wp.float32):
    ball1 = balls[0]
    ball2 = balls[1]

    pos1 = wp.transform_get_translation(body_q[ball1.idx])
    pos2 = wp.transform_get_translation(body_q[ball2.idx])

    

@wp.func
def compute_linear_correction_3d(
    dx: wp.vec3,
    r1: wp.vec3,
    r2: wp.vec3,
    tf1: wp.transform,
    tf2: wp.transform,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float,
) -> float:
    c = wp.length(dx)
    if c == 0.0:
        # print("c == 0.0 in positional correction")
        return 0.0

    n = wp.normalize(dx)

    q1 = wp.transform_get_rotation(tf1)
    q2 = wp.transform_get_rotation(tf2)

    # Eq. 2-3 (make sure to project into the frame of the body)
    r1xn = wp.quat_rotate_inv(q1, wp.cross(r1, n))
    r2xn = wp.quat_rotate_inv(q2, wp.cross(r2, n))

    w1 = m_inv1 + wp.dot(r1xn, I_inv1 * r1xn)
    w2 = m_inv2 + wp.dot(r2xn, I_inv2 * r2xn)
    w = w1 + w2
    if w == 0.0:
        return 0.0
    alpha = compliance
    gamma = compliance * damping

    # Eq. 4-5
    d_lambda = -c - alpha * lambda_in
    # TODO consider damping for velocity correction?
    # delta_lambda = -(err + alpha * lambda_in + gamma * derr)
    if w + alpha > 0.0:
        d_lambda /= w * (dt + gamma) + alpha / dt

    return d_lambda


@wp.func
def compute_angular_correction_3d(
    corr: wp.vec3,
    q1: wp.quat,
    q2: wp.quat,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    alpha_tilde: float,
    # lambda_prev: float,
    relaxation: float,
    dt: float,
):
    # compute and apply the correction impulse for an angular constraint
    theta = wp.length(corr)
    if theta == 0.0:
        return 0.0

    n = wp.normalize(corr)

    # project variables to body rest frame as they are in local matrix
    n1 = wp.quat_rotate_inv(q1, n)
    n2 = wp.quat_rotate_inv(q2, n)

    # Eq. 11-12
    w1 = wp.dot(n1, I_inv1 * n1)
    w2 = wp.dot(n2, I_inv2 * n2)
    w = w1 + w2
    if w == 0.0:
        return 0.0

    # Eq. 13-14
    lambda_prev = 0.0
    d_lambda = (-theta - alpha_tilde * lambda_prev) / (w * dt + alpha_tilde / dt)
    # TODO consider lambda_prev?
    # p = d_lambda * n * relaxation

    # Eq. 15-16
    return d_lambda


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

        self.ball_1_body: BodyInfo = create_body_info("ball_1", self.model)
        self.ball_2_body: BodyInfo = create_body_info("ball_2", self.model)

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
        bodies = [self.ball_1_body, self.ball_2_body]
        wp_bodies = wp.array(bodies, dtype=BodyInfo)
        num_bodies = len(bodies)
        
        num_substeps = 10
        h = dt / num_substeps
        for i in range(num_substeps):
            for b in bodies:
                wp.launch(
                    kernel=integrate_body,
                    dim=1,
                    inputs=[
                        b,
                        h,
                        curr_state.body_q,
                        curr_state.body_qd,
                        curr_state.body_f,
                        self.model.body_mass,
                        self.model.body_inertia,
                        next_state.body_q,
                        next_state.body_qd
                    ]
                )
            
            # wp.sim.collide(self.model, curr_state)
            for i in range(10):
                wp.launch(
                    kernel=solve_ground_collisions,
                    dim=2,
                    inputs=[wp_bodies, next_state.body_q]
                )
                

            for b in bodies:
                pass



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
