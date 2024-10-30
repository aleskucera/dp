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


USD_FILE = "data/output/pendulum_simple.usd"
PLOT2D_FILE = "data/output/pendulum_simple_2d.mp4"


class PendulumSimpleSim:
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

        self.model = pendulum_world_model(cfg)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps)

        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
        self.renderer = Renderer(self.model, self.time, USD_FILE)

        self.pendulum_joint: JointInfo = create_joint_info("base_to_arm", self.model)
        self.pendulum_end_body: BodyInfo = create_body_info("pendulum_end", self.model)

        # TODO: Get the arm length from the model
        self.arm_length = 1

        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]
        wp.sim.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, None, self.states[0]
        )

        self.trajectory = Trajectory("trajectory", self.time, color=BLUE)

        self.plot2d = Plot2D(("x", "y", "z"), 
                             val_lims=[(-1, 1), (-1.5, 1.5), (0, 2)],
                             time_lims=[(0, 2), (0, 2), (0, 2)])

    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            curr_state = self.states[i]
            next_state = self.states[i + 1]
            self.simulate(curr_state, next_state, self.sim_dt)

            self.trajectory.update_position(i, curr_state.body_q, self.pendulum_end_body.idx)
            
            if i > 0:
                self.plot2d.add_trajectory(self.trajectory, i, i)
        
        self.renderer.add_trajectory(self.trajectory)

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
        
    def save_usd(self):
        self.renderer.save(self.states)
    
    def animate_2d_plot(self, save_path: str = None):
        self.plot2d.animate(self.sim_steps, interval=10, save_path=save_path)
    

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def pendulum_simple_simulation(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    model = PendulumSimpleSim(cfg)
    model.generate_target_trajectory()
    model.animate_2d_plot(save_path=PLOT2D_FILE)
    model.save_usd()


if __name__ == "__main__":
    pendulum_simple_simulation()
