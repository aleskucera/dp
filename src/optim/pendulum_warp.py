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
from optim.loss import add_trajectory_loss


USD_FILE = "data/output/pendulum_warp_optim.usd"
PLOT2D_FILE = "data/output/pendulum_warp_optim_2d.mp4"


class PendulumWarpOptim:
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

        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)   
        self.renderer = Renderer(self.model, self.time, USD_FILE)

        self.sphere_body: BodyInfo = create_body_info("pendulum_end", self.model)
        self.pendulum_end_body: BodyInfo = create_body_info("pendulum_end", self.model)

        # ======================== OPTIMIZATION ========================

        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]
        self.target_states = [self.model.state() for _ in range(self.sim_steps + 1)]
        
        self.target_ke = wp.array([2.0], dtype=wp.float32, requires_grad=True)

        self.trajectory = Trajectory("trajectory", self.time, render_color=BLUE, requires_grad=True)
        self.target_trajectory = Trajectory("target_trajectory", self.time, render_color=BLUE, requires_grad=True)


        self.plot2d = Plot2D(("x", "y", "z"), 
                             val_lims=[(-1, 1), (-1.5, 1.5), (0, 2)],
                             time_lims=[(0, 2), (0, 2), (0, 2)])
        
        self.loss = wp.array([0.0], dtype=wp.float32, requires_grad=True)

    def _reset(self):
        self.loss = wp.array([0.0], dtype=wp.float32, requires_grad=True)
    
    def generate_target_trajectory(self):
        wp.copy(self.model.joint_target_ke, self.target_ke, dest_offset=0, src_offset=0, count=1)
        for i in range(self.sim_steps):
            curr_state = self.target_states[i]
            next_state = self.target_states[i + 1]
            curr_state.clear_forces()
            wp.sim.collide(self.model, curr_state)
            self.integrator.simulate(self.model, curr_state, next_state, self.sim_dt)
            self.target_trajectory.update_data(i, curr_state.body_q, self.pendulum_end_body.idx)
            
            if i > 0:
                self.plot2d.add_trajectory(self.trajectory, i, i)
        
        # self.renderer.add_trajectory(self.trajectory)

    def forward(self, ke: wp.array):
        with wp.ScopedTimer("forward"):
            wp.copy(self.model.joint_target_ke, ke, dest_offset=0, src_offset=0, count=1)
            for i in range(self.sim_steps):
                curr_state = self.states[i]
                next_state = self.states[i + 1]
                curr_state.clear_forces()
                wp.sim.collide(self.model, curr_state)
                self.integrator.simulate(self.model, curr_state, next_state, self.sim_dt)
                self.trajectory.update_data(i, curr_state.body_q, self.pendulum_end_body.idx)
    
    def step(self, ke: wp.array):
        self._reset()
        
        self.tape = wp.Tape()
        with self.tape:
            self.forward(ke)
            add_trajectory_loss(self.trajectory, self.target_trajectory, self.loss)
        self.tape.backward(self.loss)
        
        loss = self.loss.numpy()[0]
        ke_grad = ke.grad.numpy()[0]
        self.tape.zero()
        
        print(f"Loss: {loss}")
        print(f"Gradient: {ke_grad}")

        return loss, ke_grad
    
    def plot_loss(self):
        kes = np.linspace(0, 4, 50)
        losses = np.zeros_like(kes)
        grads = np.zeros_like(kes)

        for i, ke in enumerate(kes):
            print(f"Iteration {i}")
            ke = wp.array([ke], dtype=wp.float32, requires_grad=True)
            losses[i], grads[i] = self.step(ke)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        # Plot the loss curve
        ax1.plot(kes, losses, label="Loss")
        ax1.set_xlabel("Force")
        ax1.set_ylabel("Ke")
        ax1.set_title("Loss vs Ke")
        ax1.legend()

        # Make sure that that grads are not too large
        grads = np.clip(grads, -1e4, 1e4)

        # Plot the gradient curve
        ax2.plot(kes, grads, label="Gradient", color="orange")
        ax2.set_xlabel("Ke")
        ax2.set_ylabel("Gradient")
        ax2.set_title("Gradient vs Ke")
        ax2.legend()

        plt.suptitle("Loss and Gradient vs Ke")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def save_usd(self):
        self.renderer.save(self.target_states, fps=30)
    
    def animate_2d_plot(self, save_path: str = None):
        self.plot2d.animate(self.sim_steps, interval=10, save_path=save_path)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def pendulum_warp_optimization(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    model = PendulumWarpOptim(cfg)
    model.generate_target_trajectory()
    model.plot_loss()
    # model.animate_2d_plot(PLOT2D_FILE)
    model.save_usd()


if __name__ == "__main__":
    pendulum_warp_optimization()
