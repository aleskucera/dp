import os
import math
from typing import Tuple, Dict, Any

import nvtx
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import warp as wp
import warp.sim
import warp.sim.render

from dp_utils import *


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

def carter_world_model(cfg: DictConfig) -> wp.sim.Model:
    builder = wp.sim.ModelBuilder()

    parse_urdf_args = dict(cfg.robot.parse_urdf_args)
    parse_urdf_args["xform"] = get_robot_transform(cfg)
    parse_urdf_args["builder"] = builder
    wp.sim.parse_urdf(**parse_urdf_args)
    set_joint_config(cfg, builder)
    model = builder.finalize()
    model.ground = True

    return model


class CarterDemo:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Simulation and rendering parameters
        self.fps = cfg.sim.fps
        self.sim_substeps = cfg.sim.sim_substeps
        self.frame_dt = cfg.sim.frame_dt
        self.sim_dt = cfg.sim.sim_dt

        # Simulation state
        self.sim_time = 0.0
        self.profiler = {}

        self.model = carter_world_model(cfg)

        # Create the integrator
        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
        self.renderer = wp.sim.render.SimRenderer(
            self.model, cfg.stage_path, scaling=40.0
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.left_wheel_joint: JointInfo = create_joint_info("left_wheel", self.model)
        self.right_wheel_joint: JointInfo = create_joint_info("right_wheel", self.model)

        wp.sim.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0
        )

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                control = self.get_control()
                self.simulate(control)
            self.graph = capture.graph

    def simulate(self, control=None):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(
                self.model, self.state_0, self.state_1, self.sim_dt, control=control
            )
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def get_control(self):
        control = self.model.control()
        control.joint_act = get_joint_act(
            joints=[self.left_wheel_joint, self.right_wheel_joint],
            values=[0.0, 0.0],
            model=self.model,
        )
        return control

    def step(self):
        with wp.ScopedTimer("step", dict=self.profiler, print=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                control = self.get_control()
                self.simulate(control)
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def carter_demo(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    demo = CarterDemo(cfg)
    for _ in range(cfg.sim.num_frames):
        demo.step()
        demo.render()

    if demo.renderer:
        demo.renderer.save()


if __name__ == "__main__":
    carter_demo()
