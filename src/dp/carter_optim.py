#!/usr/bin/env python

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
import matplotlib.pyplot as plt

from dp_utils import *
from sim import *

@wp.kernel
def loss_kernel(traj: wp.array(dtype=wp.transform), 
                target_traj: wp.array(dtype=wp.transform), 
                loss: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    t = wp.transform_get_translation(traj[tid])
    target_t = wp.transform_get_translation(target_traj[tid])
    difference = t - target_t
    l = wp.dot(difference, difference)
    wp.atomic_add(loss, 0, l)

@wp.kernel
def update_joint_act(step: wp.int32,
                   joints: wp.array(dtype=JointInfo),
                   control: wp.array(dtype=wp.float32), 
                   joint_act: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    joint = joints[tid]
    joint_act[joint.axis_idx] = control[0]

@wp.kernel
def update_control(control_grad: wp.array(dtype=wp.float32),
                   learning_rate: wp.float32,
                   control: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    # Check if the control grad is not NaN
    if control_grad[tid] == control_grad[tid]:
        # Clamp the control grad to avoid large changes
        control_grad[tid] = wp.clamp(control_grad[tid], -0.2, 0.2)
        control[tid] = control[tid] - learning_rate * control_grad[tid]


@wp.kernel
def compute_sin(delta_t: float, out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    t = 2.0 * float(tid) * delta_t
    out[tid] = 5.0 * wp.sin(t)

def set_joint_config(cfg: DictConfig, model: Union[wp.sim.Model, wp.sim.ModelBuilder]):
    q_list = []
    qd_list = []
    act_list = []
    target_ke_list = []
    target_kd_list = []
    axis_mode_list = []
    joint_info_list = []

    for joint in cfg.robot.joints:
        q_list.append(joint.q)
        qd_list.append(joint.qd)
        act_list.append(joint.act)
        target_ke_list.append(joint.target_ke)
        target_kd_list.append(joint.target_kd)
        axis_mode_list.append(joint.axis_mode)
        joint_info_list.append(create_joint_info(joint.name, model))

    set_joint_q(joints=joint_info_list, values=q_list, model=model)
    set_joint_qd(joints=joint_info_list, values=qd_list, model=model)
    set_joint_act(joints=joint_info_list, values=act_list, entity=model)
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
        self.fps          = cfg.sim.fps
        self.sim_dt       = cfg.sim.sim_dt
        self.frame_dt     = cfg.sim.frame_dt
        self.sim_substeps = cfg.sim.sim_substeps
        self.sim_duration = cfg.sim.sim_duration

        # self.frame_steps = int(self.sim_duration / self.frame_dt)
        self.sim_steps   = cfg.sim.num_frames * self.sim_substeps

        self.iter          = 0
        self.sim_time      = 0.0
        self.profiler      = {}
        self.learning_rate = 1

        self.use_cuda_graph = wp.get_device().is_cuda
        self.forward_graph  = None

        self.model = carter_world_model(cfg)
        self.init_state = self.model.state()

        with wp.ScopedTimer("eval_fk", use_nvtx=True, color="yellow", print=True):
            eval_fk(
                self.model, 
                self.model.joint_q, 
                self.model.joint_qd, 
                None, 
                self.init_state
            )

        # Create the integrator
        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
        self.renderer   = wp.sim.render.SimRenderer(self.model, 
                                                    cfg.stage_path, 
                                                    scaling=40.0)

        # ======================== JOINTS ========================

        self.left_wheel_joint:  JointInfo = create_joint_info("left_wheel", self.model)
        self.right_wheel_joint: JointInfo = create_joint_info("right_wheel", self.model)
        # self.joints = wp.array([self.left_wheel_joint, self.right_wheel_joint], dtype=JointInfo)
        self.joints = wp.array([self.right_wheel_joint], dtype=JointInfo)
        # self.joints = wp.array([self.left_wheel_joint], dtype=JointInfo)

        # ======================== BODIES ========================

        self.imu_body: BodyInfo = create_body_info("imu", self.model)

        # ======================== OPTIMIZATION ========================

        self.loss       = None
        self.states     = None
        self.controls   = None
        self.trajectory = None

        # Initialize control that will be optimized
        rand_control    = np.random.rand(self.sim_steps).astype(np.float32)
        self.control    = wp.array(rand_control, dtype=wp.float32, requires_grad=True)

        # Initialize target trajectory and control
        self.target_control    = wp.array([2.0], dtype=wp.float32, requires_grad=True)
        self.target_trajectory = wp.empty(self.sim_steps, dtype=wp.transformf, requires_grad=True)

        # Set the target control as a sinusoidal function
        # wp.launch(compute_sin, dim=self.sim_steps, inputs=[self.sim_dt, self.target_control])

        self._reset()
    
    def _reset(self):
        with wp.ScopedTimer("reset", use_nvtx=True, color="yellow", print=True):
            # Reset the attributes
            self.loss       = wp.array([0], dtype=wp.float32, requires_grad=True)
            self.states     = [self.model.state(requires_grad=True) for _ in range(self.sim_steps + 1)]
            self.controls   = [self.model.control(clone_variables=True, requires_grad=True) for _ in range(self.sim_steps)]
            self.trajectory = wp.empty(self.sim_steps, dtype=wp.transformf, requires_grad=True)

            # Set the state body_q and body_qd by the initial state which was computed by forward kinematics
            for state in self.states:
                state.body_q = wp.clone(self.init_state.body_q, requires_grad=True)
                state.body_qd = wp.clone(self.init_state.body_qd, requires_grad=True)
    
    def generate_target_trajectory(self):
        with wp.ScopedTimer("generate_target_trajectory", use_nvtx=True, color="yellow", print=True):
            for i in range(self.sim_steps):
                with wp.ScopedTimer("clear_forces", use_nvtx=True, color="yellow", print=False):
                    self.states[i].clear_forces()
                with wp.ScopedTimer("collide", use_nvtx=True, color="yellow", print=False):
                    wp.sim.collide(self.model, self.states[i])
                with wp.ScopedTimer("launch_update_joint_act", use_nvtx=True, color="yellow", print=False):
                    wp.launch(update_joint_act,
                            dim=len(self.joints),
                            inputs=[i, 
                                    self.joints, 
                                    self.target_control, 
                                    self.controls[i].joint_act]
                    )
                with wp.ScopedTimer("simulate", use_nvtx=True, color="yellow", print=False):
                    self.integrator.simulate(self.model, 
                                            self.states[i], 
                                            self.states[i + 1], 
                                            self.sim_dt, 
                                            control=self.controls[i])
                with wp.ScopedTimer("launch_update_trajectory", use_nvtx=True, color="yellow", print=False):
                    wp.launch(update_trajectory, 
                            dim=1, 
                            inputs=[self.imu_body, 
                                    self.states[i + 1].body_q, 
                                    i, 
                                    self.target_trajectory]
                            )
                    
                if i % self.cfg.sim.sim_substeps == 0:
                    with wp.ScopedTimer("render"):
                        self.renderer.begin_frame(self.sim_time)
                        self.renderer.render(self.states[i])
                        self.renderer.end_frame()

                    self.sim_time += self.frame_dt

    def forward(self):
        for i in range(self.sim_steps):
            self.states[i].clear_forces()
            wp.sim.collide(self.model, self.states[i])
            wp.launch(kernel=update_joint_act, 
                      dim=len(self.joints), 
                      inputs=[i, 
                              self.joints, 
                              self.control, 
                              self.controls[i].joint_act])
            self.integrator.simulate(self.model, 
                                     self.states[i], 
                                     self.states[i + 1], 
                                     self.sim_dt, 
                                     control=self.controls[i])
            
            wp.launch(kernel=update_trajectory, 
                      dim=1, 
                      inputs=[self.imu_body, 
                              self.states[i + 1].body_q, i, 
                              self.trajectory])
            
            wp.tape.backward(grads={self.states[i + 1].body_q: wp.ones((len(self.states[i + 1].body_q), 7), dtype=wp.float32)})

        wp.launch(kernel=loss_kernel, 
                  dim=len(self.trajectory), 
                  inputs=[self.trajectory, self.target_trajectory, self.loss])
        
    def optimize_control(self):
        self.tape = wp.Tape()
        with wp.ScopedTimer("reset"):
            self._reset()
            
        with self.tape:
            self.forward()
        self.tape.backward(self.loss)

        wp.launch(update_control, 
                  dim=len(self.control), 
                  inputs=[self.control.grad, self.learning_rate, self.control])

        self.tape.zero()
    
    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                if self.forward_graph is None:
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        self._reset()
                        with self.tape:
                            self.forward()
                        self.tape.backward(self.loss)
                    self.forward_graph = capture.graph
                else:
                    wp.capture_launch(self.forward_graph)
            else:
                self.tape = wp.Tape()
                self._reset()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)

            wp.launch(update_control, 
                      dim=len(self.control), 
                      inputs=[self.control.grad, self.learning_rate, self.control])

            self.tape.zero()

            self.iter += 1

        return self.control.numpy()

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.current_state)
            self.renderer.end_frame()

    def plot_target_trajectory(self):
        time = np.linspace(0, self.sim_duration, self.sim_steps)
        plt.plot(time, self.target_trajectory.numpy()[:, 0], label="x-axis")
        plt.plot(time, self.target_trajectory.numpy()[:, 1], label="y-axis")
        plt.show()


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def carter_demo(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    train_iters = 100
    controls = []

    demo = CarterOptim(cfg)
    demo.generate_target_trajectory()
    demo.renderer.save()
    exit()

    target_control = demo.target_control.numpy()

    for i in range(train_iters):
        control = demo.step()
        controls.append(control)

    # Plot the controls
    time = np.linspace(0, demo.sim_duration, demo.sim_steps)
    for i, control in enumerate(controls):
        if i % 10 == 0:
            plt.plot(time, control, label=f"Iteration {i}")
    plt.plot(time, target_control, label="Target control")
    plt.legend()
    # Save the plot as png
    plt.savefig("controls.png")
    plt.show()
    
if __name__ == "__main__":
    carter_demo()
