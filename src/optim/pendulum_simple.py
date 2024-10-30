from typing import Union

import hydra
import numpy as np
import warp as wp
import warp.sim
import warp.optim
import warp.sim.render
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from sim import *
from dp_utils import *
from sim.integrator_euler import eval_rigid_contacts

from optim.loss import add_trajectory_loss


USD_FILE = "data/output/pendulum_simple_optim.usd"
PLOT2D_FILE = "data/output/pendulum_simple_optim_2d.mp4"

@wp.kernel
def simulate_pendulum_kernel(
    current_joint_q: wp.array(dtype=wp.float32),
    current_joint_qd: wp.array(dtype=wp.float32),
    current_body_f: wp.array(dtype=wp.spatial_vectorf),
    next_joint_q: wp.array(dtype=wp.float32),
    next_joint_qd: wp.array(dtype=wp.float32),
    joint_kd: wp.array(dtype=wp.float32),
    joint: JointInfo,
    body: BodyInfo,
    gravity: wp.vec3f,
    arm_length: wp.float32,
    mass: wp.float32,
    dt: wp.float32,
):
    angle = current_joint_q[joint.axis_idx]
    angular_velocity = current_joint_qd[joint.axis_idx]
    damping_coeff = joint_kd[joint.axis_idx]
    torque = current_body_f[body.idx][0]

    # Compute the angular acceleration
    angular_acceleration = (gravity[2] / arm_length) * wp.sin(angle)
    angular_acceleration -= (
        damping_coeff / (mass * wp.pow(arm_length, 2.0))
    ) * angular_velocity
    angular_acceleration += torque / (mass * wp.pow(arm_length, 2.0))

    # Perform the integration
    next_angular_velocity = angular_velocity + angular_acceleration * dt
    next_angle = angle + next_angular_velocity * dt

    # Update the joint state
    next_joint_q[joint.axis_idx] = next_angle
    next_joint_qd[joint.axis_idx] = next_angular_velocity


class PendulumSimpleOptim:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.epoch = 0

        # Simulation and rendering parameters
        self.fps = cfg.sim.fps
        self.sim_dt = cfg.sim.sim_dt
        self.frame_dt = cfg.sim.frame_dt
        self.num_frames = cfg.sim.num_frames
        self.sim_substeps = cfg.sim.sim_substeps
        self.sim_duration = cfg.sim.sim_duration
        self.sim_steps = cfg.sim.num_frames * cfg.sim.sim_substeps

        self.model = pendulum_world_model(cfg)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps)

        self.renderer = Renderer(self.model, self.time, USD_FILE)

        self.pendulum_joint: JointInfo = create_joint_info("base_to_arm", self.model)
        self.pendulum_end_body: BodyInfo = create_body_info("pendulum_end", self.model)

        # TODO: Get the arm length from the model
        self.mass = 1
        self.arm_length = 1

        self.loss = wp.array([0.0], dtype=wp.float32, requires_grad=True)
        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.states[0])

        self.kd = wp.array([np.random.uniform(0, 1)], dtype=wp.float32, requires_grad=True)
        self.trajectory = Trajectory("trajectory", self.time, color=BLUE, requires_grad=True)

        self.target_kd = wp.array([1.0], dtype=wp.float32)
        self.target_trajectory = Trajectory("target_trajectory", self.time, color=ORANGE)

        self.best_loss = np.inf
        self.best_kd = wp.clone(self.kd)
        self.best_trajectory = self.trajectory.clone

        self.epoch = 0
        self.optimizer = wp.optim.Adam([self.kd], lr=1)
        self.optim_segments = generate_segments(self.sim_steps, 4)
        self.frames_per_segment = 10
        self.epochs_per_segment = 100
        self.num_plot_frames = len(self.optim_segments) * self.frames_per_segment

        self.plot2d = Plot2D(("x", "y", "z"))

    def _reset(self):
        self.loss = wp.array([0], dtype=wp.float32, requires_grad=True)

    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            curr_state = self.states[i]
            next_state = self.states[i + 1]
            self.simulate(curr_state, next_state, self.target_kd, self.sim_dt)
            self.target_trajectory.update_position(i, curr_state.body_q, self.pendulum_end_body.idx)
            
        self.plot2d.add_trajectory(self.target_trajectory)
        self.renderer.add_trajectory(self.target_trajectory)

    def forward(self, segment: dict):
        for i in range(segment["start"], segment["end"]):
            curr_state = self.states[i]
            next_state = self.states[i + 1]
            self.simulate(curr_state, next_state, self.kd, self.sim_dt)
            self.trajectory.update_position(i, curr_state.body_q, self.pendulum_end_body.idx)

    def step(self, segment: dict):
        self.tape = wp.Tape()
        self._reset()
        with self.tape:
            self.forward(segment)
            add_trajectory_loss(self.trajectory, self.target_trajectory, self.loss)
        self.tape.backward(self.loss)
        self.optimizer.step(grad=[self.kd.grad])
        self.tape.zero()

        return self.loss.numpy()[0]

    def train(self):
        epoch = 0
        frame = 0
        vis_interval = self.epochs_per_segment // self.frames_per_segment

        for segment in self.optim_segments:
            for _ in range(self.epochs_per_segment):
                loss = self.step(segment=segment)

                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_kd = wp.clone(self.kd)
                    self.best_trajectory = self.trajectory.clone
                
                if epoch % vis_interval == 0:
                    print(f"Epoch {epoch}, Loss: {loss}")
                    self.plot2d.add_trajectory(self.trajectory, frame)
                    frame += 1

                epoch += 1

        print(f"Best Loss: {self.best_loss}")
        print(f"Best KD: {self.best_kd}")
        self.renderer.add_trajectory(self.best_trajectory)

    def save_usd(self):
        self.renderer.save(self.states)

    def animate_2d_plot(self, save_path: str = None):
        self.plot2d.animate(num_frames=self.num_plot_frames, save_path=save_path)

    def simulate(self, curr_state: wp.sim.State, next_state: wp.sim.State, joint_kd: wp.array, dt: float):
        curr_state.clear_forces()
        wp.sim.collide(self.model, curr_state)

        wp.launch(
            kernel=eval_rigid_contacts,
            dim=self.model.rigid_contact_max,
            inputs=[
                curr_state.body_q,
                curr_state.body_qd,
                self.model.body_com,
                self.model.shape_materials,
                self.model.shape_geo,
                self.model.shape_body,
                self.model.rigid_contact_count,
                self.model.rigid_contact_point0,
                self.model.rigid_contact_point1,
                self.model.rigid_contact_normal,
                self.model.rigid_contact_shape0,
                self.model.rigid_contact_shape1,
                True,
            ],
            outputs=[curr_state.body_f]
        )

        wp.launch(
            kernel=simulate_pendulum_kernel,
            dim=1,
            inputs=[
                curr_state.joint_q,
                curr_state.joint_qd,
                curr_state.body_f,
                next_state.joint_q,
                next_state.joint_qd,
                joint_kd,
                self.pendulum_joint,
                self.pendulum_end_body,
                self.model.gravity,
                self.arm_length,
                self.mass,
                dt,
            ],
        )

        wp.sim.eval_fk(
            self.model, next_state.joint_q, next_state.joint_qd, None, next_state
        )
    

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def pendulum_simple_optimization(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    model = PendulumSimpleOptim(cfg)
    model.generate_target_trajectory()
    model.train()
    model.animate_2d_plot(save_path=PLOT2D_FILE)
    model.save_usd()


if __name__ == "__main__":
    pendulum_simple_optimization()
