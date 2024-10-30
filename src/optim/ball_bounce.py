import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import warp as wp
import warp.sim
import warp.optim
import warp.sim.render

from sim import *
from dp_utils import *

from optim.loss import add_trajectory_loss


USD_FILE = "data/output/ball_bounce.usd"
PLOT2D_FILE = "data/output/ball_bounce_2d.mp4"
PLOT3D_FILE = "data/output/ball_bounce_3d.mp4"


@wp.kernel
def apply_force_kernel(
    particle_f: wp.array(dtype=wp.vec3f),
    force: wp.array(dtype=wp.vec3f),
    step: wp.int32,
):
    particle_f[0] = force[step]

def target_force(t: np.ndarray) -> np.ndarray:
    x = np.full_like(t, 0.5)
    y = 5 * np.sin(1.5 * np.pi * t)
    z = np.zeros_like(t)
    return np.vstack([x, y, z]).T

def random_force(t: np.ndarray) -> np.ndarray:
    return np.random.uniform(-1, 1, size=(len(t), 3))


class BallBounceOptim:
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

        self.model = ball_world_model()
        self.time = np.linspace(0, self.sim_duration, self.sim_steps)

        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.renderer = Renderer(self.model, self.time, USD_FILE)

        self.loss = wp.array([0], dtype=wp.float32, requires_grad=True)
        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]
        self.controls = [self.model.control() for _ in range(self.sim_steps)]

        self.force = wp.array(random_force(self.time), dtype=wp.vec3f, requires_grad=True)
        self.target_force = wp.array(target_force(self.time), dtype=wp.vec3f)

        self.trajectory = Trajectory("trajectory", self.time, color=BLUE, requires_grad=True)
        self.target_trajectory = Trajectory("target_trajectory", self.time, color=ORANGE)

        self.best_loss = np.inf
        self.best_force = wp.clone(self.force)
        self.best_trajectory = self.trajectory.clone

        self.epoch = 0
        self.optimizer = wp.optim.Adam([self.force], lr=8e-1)
        self.optim_segments = generate_segments(self.sim_steps, 10)
        self.frames_per_segment = 10
        self.epochs_per_segment = 100
        self.num_plot_frames = len(self.optim_segments) * self.frames_per_segment

        self.plot2d = Plot2D(subplots=("x", "y", "z"))
        self.plot3d = Plot3D(x_lim=(-0.5, 0.18), y_lim=(0.0, 1.0), z_lim=(2.0, 2.57), padding=0.1)

    def _reset(self):
        self.loss = wp.array([0], dtype=wp.float32, requires_grad=True)

    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            curr_state = self.states[i]
            next_state = self.states[i + 1]
            curr_state.clear_forces()
            wp.launch(apply_force_kernel, dim=1,
                inputs=[curr_state.particle_f, self.target_force, i])
            wp.sim.collide(self.model, curr_state)
            self.integrator.simulate(self.model, curr_state, next_state, self.sim_dt)
            self.target_trajectory.update_position(i, curr_state.particle_q, 0)

        self.plot2d.add_trajectory(self.target_trajectory)
        self.plot3d.add_trajectory(self.target_trajectory)
        self.renderer.add_trajectory(self.target_trajectory)

    def simulate(self, segment: dict):
        for i in range(segment["start"], segment["end"]):
            curr_state = self.states[i]
            next_state = self.states[i + 1]
            curr_state.clear_forces()
            wp.launch(apply_force_kernel, dim=1,
                inputs=[curr_state.particle_f, self.force, i])
            wp.sim.collide(self.model, curr_state)
            self.integrator.simulate(self.model, curr_state, next_state, self.sim_dt)
            self.trajectory.update_position(i, curr_state.particle_q, 0)

    def step(self, segment: dict):
        self.tape = wp.Tape()
        self._reset()
        with self.tape:
            self.simulate(segment)
            add_trajectory_loss(self.trajectory, self.target_trajectory, self.loss)
        self.tape.backward(self.loss)
        self.optimizer.step(grad=[self.force.grad])
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
                    self.best_force = wp.clone(self.force)
                    self.best_trajectory = self.trajectory.clone
                
                if epoch % vis_interval == 0:
                    print(f"Epoch {epoch}, Loss: {loss}")
                    self.plot2d.add_trajectory(self.trajectory, frame) 
                    self.plot3d.add_trajectory(self.trajectory, frame)
                    frame += 1

                epoch += 1

        print(f"Best Loss: {self.best_loss}")
        self.renderer.add_trajectory(self.best_trajectory)

    def save_usd(self):
        self.renderer.save(self.states)

    def animate_2d_plot(self, save_path: str = None):
        self.plot2d.animate(num_frames=self.num_plot_frames, save_path=save_path)

    def animate_3d_plot(self, save_path: str = None):
        self.plot3d.animate(num_frames=self.num_plot_frames, save_path=save_path)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def ball_bounce_optimization(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    model = BallBounceOptim(cfg)
    model.generate_target_trajectory()
    model.train()
    model.animate_2d_plot(save_path=PLOT2D_FILE)
    model.animate_3d_plot(save_path=PLOT3D_FILE)
    model.save_usd()


if __name__ == "__main__":
    ball_bounce_optimization()
