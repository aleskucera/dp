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


USD_FILE = "data/output/pendulum_wall.usd"
PLOT2D_FILE = "data/output/pendulum_wall_2d.mp4"

@wp.kernel
def integrate_pendulum(
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
    pendulum_end_body: BodyInfo,
    radius: wp.float32,
    restitution: wp.float32,
    dt: wp.float32
):
    pass

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

class PendulumWallSim:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Simulation and rendering parameters
        self.fps = cfg.sim.fps
        self.sim_dt = cfg.sim.sim_dt
        self.frame_dt = cfg.sim.frame_dt
        self.num_frames = cfg.sim.num_frames
        self.sim_substeps = cfg.sim.sim_substeps
        self.sim_duration = cfg.sim.sim_duration
        self.sim_steps = cfg.sim.num_frames * cfg.sim.sim_substeps

        self.model = pendulum_world_model(cfg, wall=True)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps)

        self.renderer = Renderer(self.model, self.time, USD_FILE)

        self.pendulum_joint: JointInfo = create_joint_info("base_to_arm", self.model)
        self.pendulum_end_body: BodyInfo = create_body_info("pendulum_end", self.model)
        self.wall_body: BodyInfo = create_body_info("wall", self.model)

        # TODO: Get the arm length from the model
        self.mass = 1
        self.arm_length = 1

        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]
        wp.sim.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, None, self.states[0]
        )

        self.trajectory = Trajectory("trajectory", self.time, render_color=BLUE)

        self.plot2d = Plot2D(("x", "y", "z"), 
                             val_lims=[(-1, 1), (-1.5, 1.5), (0, 2)],
                             time_lims=[(0, 2), (0, 2), (0, 2)])

    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            curr_state = self.states[i]
            next_state = self.states[i + 1]
            self.simulate(curr_state, next_state, self.sim_dt)

            self.trajectory.update_data(i, curr_state.body_q, self.pendulum_end_body.idx)
            
            if i > 0:
                self.plot2d.add_trajectory(self.trajectory, i, i)
        
        self.renderer.add_trajectory(self.trajectory)

    def simulate(self, curr_state: wp.sim.State, next_state: wp.sim.State, dt: float):
        curr_state.clear_forces()
        wp.sim.collide(self.model, curr_state)

        contact_count = self.model.rigid_contact_count.numpy()[0]
        if contact_count > 0:
            print(f"Rigid contact count: {contact_count}")

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
                self.model.joint_target_kd,
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
        
    def save_usd(self):
        self.renderer.save(self.states)
    
    def animate_2d_plot(self, save_path: str = None):
        self.plot2d.animate(self.sim_steps, interval=10, save_path=save_path)
    

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def pendulum_simple_simulation(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    model = PendulumWallSim(cfg)
    model.generate_target_trajectory()
    # model.animate_2d_plot(save_path=PLOT2D_FILE)
    model.save_usd()


if __name__ == "__main__":
    pendulum_simple_simulation()
