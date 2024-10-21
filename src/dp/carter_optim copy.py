import os
import math
from typing import Tuple, Dict, Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import warp as wp
import warp.sim
import warp.sim.render

from dp_utils import *


@wp.kernel
def loss_kernel(traj: wp.array(dtype=wp.transform), 
                target_traj: wp.array(dtype=wp.transform), 
                loss: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    t = wp.transform_get_translation(traj[tid])
    target_t = wp.transform_get_translation(target_traj[tid])
    difference = t - target_t
    l = wp.dot(difference, difference)
    if l != l:
        wp.atomic_add(loss, 0, 0.0)
    wp.atomic_add(loss, 0, l)

@wp.kernel
def update_ke_kernel(joint: JointInfo, 
                     joint_ke: wp.array(dtype=wp.float32), 
                     joint_ke_grad: wp.array(dtype=wp.float32), 
                     learning_rate: wp.float32):
    idx = joint.axis_idx
    joint_ke[idx] = joint_ke[idx] - learning_rate * joint_ke_grad[idx]


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
    model = builder.finalize(requires_grad=True)
    model.ground = True

    return model


class CarterOptim:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Simulation and rendering parameters
        self.fps = cfg.sim.fps
        self.sim_substeps = cfg.sim.sim_substeps
        self.frame_dt = cfg.sim.frame_dt
        self.sim_dt = cfg.sim.sim_dt
        self.sim_duration = cfg.sim.sim_duration

        self.frame_steps = int(self.sim_duration / self.frame_dt)
        self.sim_steps = self.frame_steps * self.sim_substeps

        # Simulation state
        self.sim_time = 0.0
        self.profiler = {}

        self.model = carter_world_model(cfg)

        # Create the integrator
        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
        self.renderer = wp.sim.render.SimRenderer(
            self.model, cfg.stage_path, scaling=40.0
        )

        self.init_state = self.model.state()
        self.current_state = self.model.state()
        self.next_state = self.model.state()
        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]

        self.left_wheel_joint: JointInfo = create_joint_info("left_wheel", self.model)
        self.right_wheel_joint: JointInfo = create_joint_info("right_wheel", self.model)

        self.imu_body: BodyInfo = create_body_info("imu", self.model)

        wp.sim.eval_fk(
            self.model, 
            self.model.joint_q, 
            self.model.joint_qd, 
            None, 
            self.current_state
        )

        self.target_ke = 1e1

        self.loss = None
        self.learning_rate = 1e-3

        self.iter = 0

        self.trajectory = None
        self.target_trajectory = wp.empty(self.sim_steps, dtype=wp.transformf, requires_grad=True)

        self.control = None
        self.target_control = None

    def forward(self):
        for i in range(self.sim_steps):
            self.states[i].clear_forces()
            wp.sim.collide(self.model, self.states[i])
            self.integrator.simulate(self.model, 
                                     self.states[i], 
                                     self.states[i + 1], 
                                     self.sim_dt, 
                                     control=self.get_control())
            
            wp.launch(update_trajectory, dim=1, inputs=[self.imu_body, self.states[i + 1].body_q, i, self.trajectory])
            self.tape.backward(grads={self.states[i + 1].body_q: wp.ones((len(self.states[i + 1].body_q), 7), dtype=wp.float32)})
            print(self.model.joint_target_ke.grad.numpy())

        wp.launch(loss_kernel, dim=len(self.trajectory), inputs=[self.trajectory, self.target_trajectory, self.loss])
    
    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            self.current_state.clear_forces()
            wp.sim.collide(self.model, self.current_state)
            self.integrator.simulate(self.model, self.current_state, self.next_state, self.sim_dt, control=self.get_control())
            
            wp.launch(update_trajectory, dim=1, inputs=[self.imu_body, self.current_state.body_q, i, self.target_trajectory])
        
            (self.current_state, self.next_state) = (self.next_state, self.current_state)
        
        set_joint_target_ke(joints=[self.left_wheel_joint], values=[2e1], model=self.model)

    def reset(self):
        self.current_state = self.init_state
        self.loss = wp.array([0], dtype=wp.float32, requires_grad=True)
        self.trajectory = wp.empty(self.sim_steps, dtype=wp.transformf, requires_grad=True)
    
    def step(self):
        with wp.ScopedTimer("step"):
            self.tape = wp.Tape()
            self.tape.reset()
            self.reset()

            with self.tape:
                self.forward()
            
            self.tape.backward(self.loss)

            wp.launch(update_ke_kernel, dim=1, inputs=[self.left_wheel_joint, 
                                                       self.model.joint_target_ke, 
                                                       self.model.joint_target_ke.grad, 
                                                       self.learning_rate])
            self.tape.zero()

            self.iter += 1

    def get_control(self):
        control = self.model.control()
        control.joint_act = get_joint_act(
            joints=[self.left_wheel_joint, self.right_wheel_joint],
            values=[1.0, 1.0],
            model=self.model,
        )
        return control

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.current_state)
            self.renderer.end_frame()


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def carter_demo(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    train_iters = 10

    demo = CarterOptim(cfg)
    demo.generate_target_trajectory()

    for i in range(train_iters):
        demo.step()
    #     if i % 10 == 0:
    #         demo.render()
    
    # if demo.renderer:
    #     demo.renderer.save()

if __name__ == "__main__":
    carter_demo()
