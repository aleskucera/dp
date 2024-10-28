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


OUTPUT_FILE = "data/output/pendulum_simple.usd"

def set_joint_config(cfg: DictConfig, model: Union[wp.sim.Model, wp.sim.ModelBuilder]):
    q_list = []
    qd_list = []
    target_ke_list = []
    target_kd_list = []
    axis_mode_list = []
    joint_info_list = []

    for joint in cfg.robot.joints:
        q_list.append(joint.q)
        qd_list.append(joint.qd)
        target_ke_list.append(joint.target_ke)
        target_kd_list.append(joint.target_kd)
        axis_mode_list.append(joint.axis_mode)
        joint_info_list.append(create_joint_info(joint.name, model))

    set_joint_q(joints=joint_info_list, values=q_list, model=model)
    set_joint_qd(joints=joint_info_list, values=qd_list, model=model)
    set_joint_target_ke(joints=joint_info_list, values=target_ke_list, model=model)
    set_joint_target_kd(joints=joint_info_list, values=target_kd_list, model=model)
    set_joint_axis_mode(joints=joint_info_list, values=axis_mode_list, model=model)

def get_robot_transform(cfg: DictConfig) -> wp.transformf:
    position = wp.vec3(
        cfg.robot.position.x, cfg.robot.position.y, cfg.robot.position.z
    )
    rotation = wp.quat_rpy(
        math.radians(cfg.robot.rotation.roll),
        math.radians(cfg.robot.rotation.pitch),
        math.radians(cfg.robot.rotation.yaw),
    )
    return wp.transform(position, rotation)

def pendulum_world_model(cfg: DictConfig) -> wp.sim.Model:
    builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

    parse_urdf_args = dict(cfg.robot.parse_urdf_args)
    parse_urdf_args["xform"] = get_robot_transform(cfg)
    parse_urdf_args["builder"] = builder
    wp.sim.parse_urdf(**parse_urdf_args)
    set_joint_config(cfg, builder)
    model = builder.finalize()
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

        self.iter = 0
        self.profiler = {}
        self.sim_time = 0.0

        self.use_cuda_graph = wp.get_device().is_cuda

        self.model = pendulum_world_model(cfg)
        self.init_state = self.model.state()

        # Create the integrator and renderer
        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
        self.renderer = wp.sim.render.SimRenderer(
            self.model, OUTPUT_FILE, scaling=1.0
        )

        self.pendulum_joint: JointInfo = create_joint_info("base_to_arm", self.model)
        self.pendulum_end_body: BodyInfo = create_body_info("pendulum_end", self.model)

        # TODO: Get the arm length from the model
        self.arm_length = 1

        self.states = [
            self.model.state(requires_grad=True) for _ in range(self.sim_steps + 1)
        ]
        wp.sim.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, None, self.states[0]
        )

        self.trajectory = wp.empty(self.sim_steps, dtype=wp.vec3f, requires_grad=True)

        self.fig, self.ax = plt.subplots()

    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            self.states[i].clear_forces()
            wp.sim.collide(self.model, self.states[i])

            self.simulate(
                self.states[i], self.states[i + 1], self.frame_dt
            )

            update_trajectory(
                trajectory=self.trajectory,
                q=self.states[i].body_q,
                time_step=i,
                q_idx=self.pendulum_end_body.idx,
            )

            if i % self.cfg.sim.sim_substeps == 0:
                self.sim_time += self.frame_dt
                self.render(i)

    def simulate(self, current_state: wp.sim.State, next_state: wp.sim.State, dt: float):
        angle = current_state.joint_q.numpy()[self.pendulum_joint.axis_idx]
        angular_velocity = current_state.joint_qd.numpy()[self.pendulum_joint.axis_idx]

        angular_acceleration = (self.model.gravity[2] / self.arm_length) * np.sin(angle) # g/l * sin(angle)
        next_angular_velocity = angular_velocity + angular_acceleration * dt
        next_angle = angle + next_angular_velocity * dt

        set_joint_q(joints=[self.pendulum_joint], values=[next_angle], model=next_state)
        set_joint_qd(joints=[self.pendulum_joint], values=[next_angular_velocity], model=next_state)

        wp.sim.eval_fk(
                self.model, next_state.joint_q, next_state.joint_qd, None, next_state
            )       


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
    plot_time_series(ax=model.ax, trajectory=model.trajectory, axis='z')
    plt.show()


if __name__ == "__main__":
    carter_demo()
