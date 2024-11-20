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


USD_FILE = "data/output/pendulum_warp.usd"
PLOT2D_FILE = "data/output/pendulum_warp_2d.mp4"


class PendulumWarpSim:
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


        self.sphere_body: BodyInfo = create_body_info("pendulum_end", self.model)
        self.pendulum_end_body: BodyInfo = create_body_info("pendulum_end", self.model)

        # ======================== OPTIMIZATION ========================

        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]

        self.trajectory = Trajectory("trajectory", self.time, render_color=BLUE)

        self.plot2d = Plot2D(("x", "y", "z"), 
                             val_lims=[(-1, 1), (-1.5, 1.5), (0, 2)],
                             time_lims=[(0, 2), (0, 2), (0, 2)])

    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            curr_state = self.states[i]
            next_state = self.states[i + 1]
            curr_state.clear_forces()
            self.integrator.simulate(self.model, curr_state, next_state, self.frame_dt)
            self.trajectory.update_data(i, curr_state.body_q, self.pendulum_end_body.idx)
            
            if i > 0:
                self.plot2d.add_trajectory(self.trajectory, i, i)
        
        self.renderer.add_trajectory(self.trajectory)

    def save_usd(self):
        self.renderer.save(self.states)
    
    def animate_2d_plot(self, save_path: str = None):
        self.plot2d.animate(self.sim_steps, interval=10, save_path=save_path)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def pendulum_warp_simulation(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    model = PendulumWarpSim(cfg)
    model.generate_target_trajectory()
    model.animate_2d_plot(PLOT2D_FILE)
    model.save_usd()


if __name__ == "__main__":
    pendulum_warp_simulation()
