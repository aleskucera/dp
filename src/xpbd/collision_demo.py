from typing import Union, List

import hydra
import numpy as np
import warp as wp
import warp.sim
import warp.optim
import warp.sim.render
import matplotlib
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
from sympy.physics.mechanics import inertia

from sim import *
from dp_utils import *
from xpbd_np.position_correction import apply_positional_correction
from xpbd_np.utils import multiply_quaternions
from xpbd.integrator_xpbd import XPBDIntegrator

matplotlib.use('TkAgg')

USD_FILE = "data/output/ball_bounce_simple.usd"


def box_world_model(gravity: bool = True) -> wp.sim.Model:
    if gravity:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))
    else:
        builder = wp.sim.ModelBuilder(gravity=0.0, up_vector=wp.vec3(0, 0, 1))

    b = builder.add_body(origin=wp.transform((2.0, 0.0, 0.0), wp.quat_identity()), name="main_box")
    builder.add_shape_box(body=b, hx=1.0, hy=1.0, hz=1.0, density=1.0, has_ground_collision=False)

    # Create quaternion which is rotated 45 degrees around the z-axis and then 30 degrees around the x-axis
    q = R.from_euler("zx", [80, 30], degrees=True).as_quat()
    b1 = builder.add_body(origin=wp.transform((-1.5, 0.0, 0.0), q), name="box1")
    builder.add_shape_box(body=b1, hx=1.0, hy=1.0, hz=1.0, density=1.0, has_ground_collision=False)

    model = builder.finalize(requires_grad=True)
    return model


class Box:
    def __init__(self, hx: float, hy: float, hz: float, color: Color, trajectory: Trajectory, collisions: np.ndarray):
        """
        Represents a single box.

        Args:
            hx, hy, hz (float): Half dimensions of the box.
            color (Color): Box color.
            trajectory (Trajectory): Trajectory object containing position and rotation data.
            collisions (np.ndarray): Collision points array.
        """
        self.hx, self.hy, self.hz = hx, hy, hz
        self.color = color
        self.trajectory = trajectory
        self.pos = trajectory.pos.numpy()
        self.rot = trajectory.rot.numpy()
        self.collisions = collisions

    def plot(self, ax: plt.Axes, frame: int):
        """
        Plots the box at the specified frame.

        Args:
            ax (matplotlib Axes3D): The 3D axis to plot on.
            frame (int): The current frame index.
        """
        x = self.pos[frame]
        q = self.rot[frame]
        colls = self.collisions[frame]

        self.plot_axis(ax, x, q)
        self.plot_box_geometry(ax, x, q)
        self.plot_collisions(ax, x, q, colls)

    def plot_box_geometry(self, ax: plt.Axes, x: np.ndarray, q: np.ndarray):
        corners = np.array([[-self.hx, -self.hy, -self.hz],
                            [self.hx, -self.hy, -self.hz],
                            [self.hx, self.hy, -self.hz],
                            [-self.hx, self.hy, -self.hz],
                            [-self.hx, -self.hy, self.hz],
                            [self.hx, -self.hy, self.hz],
                            [self.hx, self.hy, self.hz],
                            [-self.hx, self.hy, self.hz]])

        rotation = R.from_quat(q)
        rotated_corners = rotation.apply(corners) + x

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for edge in edges:
            start, end = rotated_corners[edge[0]], rotated_corners[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=self.color.rgb)

    def plot_axis(self, ax: plt.Axes, x: np.ndarray, q: np.ndarray):
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        x_rot = R.from_quat(q).apply(x_axis)
        y_rot = R.from_quat(q).apply(y_axis)
        z_rot = R.from_quat(q).apply(z_axis)

        ax.quiver(x[0], x[1], x[2], x_rot[0], x_rot[1], x_rot[2], color='r')
        ax.quiver(x[0], x[1], x[2], y_rot[0], y_rot[1], y_rot[2], color='g')
        ax.quiver(x[0], x[1], x[2], z_rot[0], z_rot[1], z_rot[2], color='b')

    def plot_collisions(self, ax: plt.Axes, x: np.ndarray, q: np.ndarray, collisions: np.ndarray):
        if np.isnan(collisions).all():
            return

        for collision in collisions:
            if np.isnan(collision).all():
                continue

            c = R.from_quat(q).apply(collision) + x
            ax.scatter(c[0], c[1], c[2], c='r', marker='x')


class BoxAnimationController:
    def __init__(self, boxes: List[Box], step: int = 30):
        """
        Initialize the animation controller for multiple boxes.

        Args:
            boxes (list of Box): List of Box objects.
        """
        self.step = step
        self.boxes = boxes
        self.current_frame = 0
        self.current_box_index = None  # None means show all boxes

        # Set up the plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.update_plot()  # Display the first frame

    def on_key_press(self, event):
        if event.key == 'n':  # Next frame
            self.current_frame = (self.current_frame + self.step) % len(self.boxes[0].trajectory.time)
            self.update_plot()

        elif event.key == 'p':  # Previous frame
            self.current_frame = (self.current_frame - self.step) % len(self.boxes[0].trajectory.time)
            self.update_plot()

        elif event.key in map(str, range(len(self.boxes))):  # Visualize a specific box
            self.current_box_index = int(event.key)
            self.update_plot()

        elif event.key == 'a':  # Visualize all boxes
            self.current_box_index = None
            self.update_plot()

    def update_plot(self):
        self.ax.cla()
        self.ax.set_title(f"Time: {self.boxes[0].trajectory.time[self.current_frame]:.2f}")
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-5, 5)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        if self.current_box_index is None:  # Show all boxes
            for box in self.boxes:
                box.plot(self.ax, self.current_frame)
        else:  # Show only the selected box
            self.boxes[self.current_box_index].plot(self.ax, self.current_frame)

        plt.draw()


def apply_positional_correction(x_a, x_b, q_a, q_b, m_a_inv, m_b_inv, I_a_inv, I_b_inv, r_a, r_b, direction, magnitude,
                                lambda_):
    d_a = R.from_quat(q_a).inv().apply(direction)
    d_b = R.from_quat(q_b).inv().apply(direction)

    rxd_a = np.cross(r_a, d_a)
    rxd_b = np.cross(r_b, d_b)

    d_lambda = compute_d_lambda(m_a_inv, m_b_inv, I_a_inv, I_b_inv, rxd_a, rxd_b, magnitude)
    lambda_ += d_lambda

    # Update positions (x_a, x_b)
    x_a = x_a + lambda_ * m_a_inv * direction
    x_b = x_b - lambda_ * m_b_inv * direction

    # Compute rotations (q_a, q_b)
    omega_a = lambda_ * I_a_inv @ rxd_a
    if np.linalg.norm(omega_a) != 0:
        q_a = multiply_quaternions(q_a, R.from_rotvec(omega_a).as_quat())

    omega_b = - lambda_ * I_b_inv @ rxd_b
    if np.linalg.norm(omega_b) != 0:
        q_b = multiply_quaternions(q_b, R.from_rotvec(omega_b).as_quat())

    return x_a, x_b, q_a, q_b, lambda_

def apply_velocity_correction(q_a, q_b, v_a, v_b, w_a, w_b, m_a_inv, m_b_inv, I_a_inv, I_b_inv, r_a, r_b, direction, magnitude, lambda_):
    d_a = R.from_quat(q_a).inv().apply(direction)
    d_b = R.from_quat(q_b).inv().apply(direction)

    rxd_a = np.cross(r_a, d_a)
    rxd_b = np.cross(r_b, d_b)

    weight_a = m_a_inv + rxd_a @ I_a_inv @ rxd_a
    weight_b = m_b_inv + rxd_b @ I_b_inv @ rxd_b
    d_lambda = magnitude / (weight_a + weight_b)

    lambda_ += d_lambda

    # Update velocities (v_a, v_b)
    v_a = v_a + lambda_ * m_a_inv * direction
    v_b = v_b - lambda_ * m_b_inv * direction

    # Compute angular velocities (w_a, w_b)
    w_a = w_a + lambda_ * I_a_inv @ rxd_a
    w_b = w_b - lambda_ * I_b_inv @ rxd_b

    return v_a, v_b, w_a, w_b, lambda_


def compute_d_lambda(m_a_inv, m_b_inv, I_a_inv, I_b_inv, rxd_a, rxd_b, c):
    w_b = m_a_inv + rxd_a @ I_a_inv @ rxd_a
    w_a = m_b_inv + rxd_b @ I_b_inv @ rxd_b
    return -(c / (w_a + w_b))

def velocity(x: np.ndarray, x_prev: np.ndarray, dt: float):
    return (x - x_prev) / dt

def angular_velocity(q: np.ndarray, q_prev: np.ndarray, dt: float):
    q_rel = multiply_quaternions(q, R.from_quat(q_prev).inv().as_quat())
    omega = 2 * np.array([q_rel[0], q_rel[1], q_rel[2]]) / dt
    if q_rel[3] < 0:
        omega = -omega
    return omega

class CollisionDemo:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Simulation and rendering parameters
        self.fps = cfg.sim.fps
        self.sim_dt = cfg.sim.sim_dt
        self.frame_dt = cfg.sim.frame_dt
        self.sim_steps = cfg.sim.sim_steps
        self.num_frames = cfg.sim.num_frames
        self.sim_substeps = cfg.sim.sim_substeps
        self.sim_duration = cfg.sim.sim_duration

        self.model = box_world_model(gravity=False)
        self.integrator = XPBDIntegrator(angular_damping=0.0)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps)
        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]

        self.main_box: BodyInfo = create_body_info("main_box", self.model)
        self.main_trajectory = Trajectory("box", self.time)
        self.main_colls = np.full((self.sim_steps, self.model.rigid_contact_max, 3), np.nan)

        self.box1: BodyInfo = create_body_info("box1", self.model)
        self.box1_trajectory = Trajectory("box1", self.time)
        self.box1_colls = np.full((self.sim_steps, self.model.rigid_contact_max, 3), np.nan)

    def simulate(self):
        self.main_trajectory.update_data(0, self.states[0].body_q, int(self.main_box.idx))
        self.box1_trajectory.update_data(0, self.states[0].body_q, int(self.box1.idx))

        body_qd = self.states[0].body_qd.numpy()
        body_qd[self.main_box.idx] = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0])
        body_qd[self.box1.idx] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.states[0].body_qd = wp.array(body_qd, dtype=wp.spatial_vector)

        for i in range(self.sim_steps - 1):
            prev_state = self.states[i]
            next_state = self.states[i + 1]
            prev_state.clear_forces()
            wp.sim.collide(self.model, prev_state, edge_sdf_iter=10)
            self.integrator.simulate(self.model, prev_state, next_state, self.sim_dt)

            if self.model.rigid_contact_count.numpy() > 0:
                # Get the indices where the rigid_contact_shape0 is the box (0)
                main_colls = self.get_collisions(self.main_box.coll_shapes.numpy()[0])
                self.main_colls[i, :main_colls.shape[0]] = main_colls

                box1_colls = self.get_collisions(self.box1.coll_shapes.numpy()[0])
                self.box1_colls[i, :box1_colls.shape[0]] = box1_colls

                # self.resolve_collisions(prev_state, next_state)

            self.update_velocities(prev_state, next_state)
            self.solve_velocities(prev_state, next_state)

            self.main_trajectory.update_data(i, next_state.body_q, int(self.main_box.idx))
            self.box1_trajectory.update_data(i, next_state.body_q, int(self.box1.idx))

    def update_velocities(self, prev_state: wp.sim.State, next_state: wp.sim.State):
        for b in [self.main_box, self.box1]:
            x_prev = prev_state.body_q.numpy()[b.idx][:3]
            x_next = next_state.body_q.numpy()[b.idx][:3]
            q_prev = prev_state.body_q.numpy()[b.idx][3:]
            q_next = next_state.body_q.numpy()[b.idx][3:]
            v = velocity(x_next, x_prev, self.sim_dt)
            w = angular_velocity(q_next, q_prev, self.sim_dt)
            body_qd = next_state.body_qd.numpy()
            body_qd[b.idx] = np.concatenate((w, v))
            next_state.body_qd = wp.array(body_qd, dtype=wp.spatial_vector)

    def solve_velocities(self, prev_state: wp.sim.State, next_state: wp.sim.State):
        contact_count = self.model.rigid_contact_count.numpy()[0]
        contact_shape0 = self.model.rigid_contact_shape0.numpy()
        contact_shape1 = self.model.rigid_contact_shape1.numpy()
        contact_normal = self.model.rigid_contact_normal.numpy()
        contact_point0 = self.model.rigid_contact_point0.numpy()
        contact_point1 = self.model.rigid_contact_point1.numpy()

        for i in range(contact_count):
            shape0 = contact_shape0[i]
            shape1 = contact_shape1[i]
            normal = contact_normal[i] * -1
            r_a = contact_point0[i]
            r_b = contact_point1[i]

            if shape0 == self.main_box.idx:
                body_a = self.main_box
                body_b = self.box1
            else:
                body_a = self.box1
                body_b = self.main_box

            q_a = next_state.body_q.numpy()[body_a.idx][3:]
            q_b = next_state.body_q.numpy()[body_b.idx][3:]
            v_a = next_state.body_qd.numpy()[body_a.idx][3:]
            v_b = next_state.body_qd.numpy()[body_b.idx][3:]
            w_a = next_state.body_qd.numpy()[body_a.idx][:3]
            w_b = next_state.body_qd.numpy()[body_b.idx][:3]

            v_a_prev = prev_state.body_qd.numpy()[body_a.idx][3:]
            v_b_prev = prev_state.body_qd.numpy()[body_b.idx][3:]
            w_a_prev = prev_state.body_qd.numpy()[body_a.idx][:3]
            w_b_prev = prev_state.body_qd.numpy()[body_b.idx][:3]

            m_a_inv = self.model.body_inv_mass.numpy()[body_a.idx]
            m_b_inv = self.model.body_inv_mass.numpy()[body_b.idx]
            I_a_inv = self.model.body_inv_inertia.numpy()[body_a.idx]
            I_b_inv = self.model.body_inv_inertia.numpy()[body_b.idx]

            v = (v_a + np.cross(w_a, r_a)) - (v_b + np.cross(w_b, r_b))
            v_n = v @ normal
            tangent = v - v_n * normal
            v_t = v @ tangent

            v_prev = (v_a_prev + np.cross(w_a_prev, r_a)) - (v_b_prev + np.cross(w_b_prev, r_b))
            v_n_prev = v_prev @ normal

            # # TODO: Compute the tangent impulse
            d_v = - tangent * 0.01
            c = np.linalg.norm(d_v)
            if c > 0:
                n = d_v / c

                v_a, v_b, w_a, w_b, _ = apply_velocity_correction(q_a, q_b, v_a, v_b, w_a, w_b,
                                                                  m_a_inv, m_b_inv, I_a_inv, I_b_inv,
                                                                  r_a, r_b, n, c, 0.0)

            # TODO: Compute the normal impulse
            d_v = normal * (-v_n - 0.9 * v_n_prev)
            c = np.linalg.norm(d_v)
            if c > 0:
                n = d_v / c
                # print(f"Normal Impulse: {c} ({contact_count}")
                v_a_new, v_b_new, w_a_new, w_b_new, _ = apply_velocity_correction(q_a, q_b, v_a, v_b, w_a, w_b,
                                                                    m_a_inv, m_b_inv, I_a_inv, I_b_inv,
                                                                    r_a, r_b, n, c, 0.0)

            # print(f"v_a_new: {v_a_new}")
            # print(f"v_b_new: {v_b_new}")
            body_qd = next_state.body_qd.numpy()
            body_qd[body_a.idx] = np.concatenate((w_a_new, v_a_new))
            body_qd[body_b.idx] = np.concatenate((w_b_new, v_b_new))
            next_state.body_qd = wp.array(body_qd, dtype=wp.spatial_vector)

    def get_collisions(self, shape_idx: int):
        indices0 = np.where(self.model.rigid_contact_shape0.numpy() == shape_idx)
        indices1 = np.where(self.model.rigid_contact_shape1.numpy() == shape_idx)
        collisions0 = self.model.rigid_contact_point0.numpy()[indices0]
        collisions1 = self.model.rigid_contact_point1.numpy()[indices1]
        collisions = np.concatenate((collisions0, collisions1), axis=0)
        return collisions

    def resolve_collisions(self, prev_state: wp.sim.State, next_state: wp.sim.State):
        contact_count = self.model.rigid_contact_count.numpy()[0]
        contact_shape0 = self.model.rigid_contact_shape0.numpy()
        contact_shape1 = self.model.rigid_contact_shape1.numpy()
        contact_normal = self.model.rigid_contact_normal.numpy()
        contact_point0 = self.model.rigid_contact_point0.numpy()
        contact_point1 = self.model.rigid_contact_point1.numpy()
        contact_offset0 = self.model.rigid_contact_offset0.numpy()
        contact_offset1 = self.model.rigid_contact_offset1.numpy()
        print(f"Contact offset0: {contact_offset0}")
        print(f"Contact offset1: {contact_offset1}")
        lambda_n = 0.0
        lambda_t = 0.0

        for _ in range(5):
            for i in range(contact_count):
                shape0 = contact_shape0[i]
                shape1 = contact_shape1[i]
                normal = contact_normal[i] * -1
                r_a = contact_point0[i] + contact_offset0[i]
                r_b = contact_point1[i] + contact_offset1[i]

                if shape0 == self.main_box.idx:
                    body_a = self.main_box
                    body_b = self.box1
                else:
                    body_a = self.box1
                    body_b = self.main_box

                # Get the transformations of the main box and box1
                x_a0 = prev_state.body_q.numpy()[body_a.idx][:3]
                q_a0 = prev_state.body_q.numpy()[body_a.idx][3:]
                x_b0 = prev_state.body_q.numpy()[body_b.idx][:3]
                q_b0 = prev_state.body_q.numpy()[body_b.idx][3:]

                x_a1 = next_state.body_q.numpy()[body_a.idx][:3]
                q_a1 = next_state.body_q.numpy()[body_a.idx][3:]
                x_b1 = next_state.body_q.numpy()[body_b.idx][:3]
                q_b1 = next_state.body_q.numpy()[body_b.idx][3:]

                m_a_inv = self.model.body_inv_mass.numpy()[body_a.idx]
                m_b_inv = self.model.body_inv_mass.numpy()[body_b.idx]
                I_a_inv = self.model.body_inv_inertia.numpy()[body_a.idx]
                I_b_inv = self.model.body_inv_inertia.numpy()[body_b.idx]

                p_a0 = x_a0 + R.from_quat(q_a0).apply(r_a)
                p_b0 = x_b0 + R.from_quat(q_b0).apply(r_b)
                p_a1 = x_a1 + R.from_quat(q_a1).apply(r_a)
                p_b1 = x_b1 + R.from_quat(q_b1).apply(r_b)

                # Compute the penetration depth
                depth = (p_a1 - p_b1) @ normal
                if depth <= 0:
                    continue

                # Apply positional correction
                x_a1, x_b1, q_a1, q_b1, lambda_n = apply_positional_correction(x_a1,
                                                                              x_b1,
                                                                              q_a1,
                                                                              q_b1,
                                                                              m_a_inv,
                                                                              m_b_inv,
                                                                              I_a_inv,
                                                                              I_b_inv,
                                                                              r_a,
                                                                              r_b,
                                                                              normal,
                                                                              depth,
                                                                              lambda_n)

                # Compute the relative motion
                d_p = (p_a1 - p_a0) - (p_b1 - p_b0)
                tangent = d_p - (d_p @ normal) * normal  # Tangential component
                tangent_norm = np.linalg.norm(tangent)

                mu = 0.1
                if lambda_t < mu * lambda_n and tangent_norm > 1e-6:
                    x_a1, x_b1, q_a1, q_b1, lambda_t = apply_positional_correction(x_a1,
                                                                                  x_b1,
                                                                                  q_a1,
                                                                                  q_b1,
                                                                                  m_a_inv,
                                                                                  m_b_inv,
                                                                                  I_a_inv,
                                                                                  I_b_inv,
                                                                                  r_a,
                                                                                  r_b,
                                                                                  tangent,
                                                                                  tangent_norm,
                                                                                  lambda_t)

                body_q = next_state.body_q.numpy()
                body_q[body_a.idx] = np.concatenate((x_a1, q_a1))
                body_q[body_b.idx] = np.concatenate((x_b1, q_b1))

                next_state.body_q = wp.array(body_q, dtype=wp.transform)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def collision_demo(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)
    model = CollisionDemo(cfg)
    model.simulate()
    main_box = Box(1.0, 1.0, 1.0, BLUE, model.main_trajectory, model.main_colls)
    box1 = Box(1.0, 1.0, 1.0, ORANGE, model.box1_trajectory, model.box1_colls)
    controller = BoxAnimationController([main_box, box1], step=10)
    plt.show()


if __name__ == "__main__":
    collision_demo()
