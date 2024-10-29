import math
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

from .loss import add_trajectory_loss


OUTPUT_FILE = "data/output/pendulum.usd"


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


def pendulum_world_model(cfg: DictConfig) -> wp.sim.Model:
    builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

    parse_urdf_args = dict(cfg.robot.parse_urdf_args)
    parse_urdf_args["xform"] = get_robot_transform(cfg)
    parse_urdf_args["builder"] = builder
    wp.sim.parse_urdf(**parse_urdf_args)
    set_joint_config(cfg, builder)

    builder.add_shape_box(body=-1, pos=wp.vec3(0.0, -0.5, 0.0), 
                          hx=1.0, hy=0.25, hz=1.0, 
                          ke=1e4, kf=0.0, kd=1e2, mu=0.2)

    model = builder.finalize(requires_grad=True)
    model.ground = True

    return model


class PendulumOptim:
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

        self.sim_time = 0.0

        self.use_cuda_graph = wp.get_device().is_cuda

        self.model = pendulum_world_model(cfg)
        self.init_state = self.model.state()

        # Create the integrator and renderer
        self.renderer = wp.sim.render.SimRenderer(self.model, OUTPUT_FILE, scaling=1.0)

        self.pendulum_joint: JointInfo = create_joint_info("base_to_arm", self.model)
        self.pendulum_end_body: BodyInfo = create_body_info("pendulum_end", self.model)
        self.arm_body: BodyInfo = create_body_info("pendulum_arm", self.model)

        # TODO: Get the arm length from the model
        self.arm_length = 1
        self.mass = 1

        self.states = [
            self.model.state(requires_grad=True) for _ in range(self.sim_steps + 1)
        ]
        wp.sim.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, None, self.states[0]
        )

        self.loss = wp.array([0.0], dtype=wp.float32, requires_grad=True)

        self.kd = wp.array(
            [np.random.uniform(0, 1)], dtype=wp.float32, requires_grad=True
        )
        self.trajectory = wp.empty(self.sim_steps, dtype=wp.vec3f, requires_grad=True)

        self.target_kd = wp.array([1.0], dtype=wp.float32)
        self.target_trajectory = wp.empty(
            self.sim_steps, dtype=wp.vec3f, requires_grad=True
        )

        self.optimizer = wp.optim.Adam([self.kd], lr=0.1)
        self.optim_segments = generate_segments(self.sim_steps, 10)

        self.fig, self.ax = plt.subplots()

    def _reset(self):
        self.sim_time = 0.0
        self.loss = wp.array([0.0], dtype=wp.float32, requires_grad=True)

    def simulate(
        self,
        current_state: wp.sim.State,
        next_state: wp.sim.State,
        joint_kd: wp.array,
        dt: float,
    ):
        current_state.clear_forces()
        wp.sim.collide(self.model, current_state)
        if (
            self.model.rigid_contact_max
            and (self.model.ground
            and self.model.shape_ground_contact_pair_count
            or self.model.shape_contact_pair_count)
        ):
            wp.launch(
                kernel=eval_rigid_contacts,
                dim=self.model.rigid_contact_max,
                inputs=[
                    current_state.body_q,
                    current_state.body_qd,
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
                outputs=[current_state.body_f]
            )

        wp.launch(
            kernel=simulate_pendulum_kernel,
            dim=1,
            inputs=[
                current_state.joint_q,
                current_state.joint_qd,
                current_state.body_f,
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

    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            self.simulate(
                current_state=self.states[i],
                next_state=self.states[i + 1],
                joint_kd=self.target_kd,
                dt=self.frame_dt,
            )

            update_trajectory(
                trajectory=self.target_trajectory,
                q=self.states[i].body_q,
                time_step=i,
                q_idx=self.pendulum_end_body.idx,
            )

            if i % self.cfg.sim.sim_substeps == 0:
                self.sim_time += self.frame_dt
                self.render(i)

    def forward(self):
        for i in range(self.sim_steps):
            self.simulate(
                current_state=self.states[i],
                next_state=self.states[i + 1],
                joint_kd=self.kd,
                dt=self.frame_dt,
            )

            update_trajectory(
                trajectory=self.trajectory,
                q=self.states[i].body_q,
                time_step=i,
                q_idx=self.pendulum_end_body.idx,
            )

    def train(self):
        for i in range(100):
            loss, kd = self.step()
            if i % 5 == 0:
                print(f"Iteration {i}, Loss: {loss}", f"KD: {kd}")

    def step(self):
        self.tape = wp.Tape()
        self._reset()
        with self.tape:
            self.forward()
            add_trajectory_loss(self.trajectory, self.target_trajectory, self.loss)
        self.tape.backward(self.loss)
        self.optimizer.step(grad=[self.kd.grad])
        self.tape.zero()

        return self.loss.numpy()[0], self.kd.numpy()

    def render(self, step: int):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.states[step])

            self.renderer.end_frame()


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def carter_demo(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    model = PendulumOptim(cfg)
    model.generate_target_trajectory()
    model.renderer.save()
    model.train()


if __name__ == "__main__":
    carter_demo()
