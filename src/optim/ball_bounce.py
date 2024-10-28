import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import warp as wp
import warp.sim
import warp.optim
import warp.sim.render
import matplotlib.pyplot as plt

from sim import *
from dp_utils import *


OUTPUT_FILE = "data/output/ball_bounce.usd"


@wp.kernel
def apply_force_kernel(
    particle_f: wp.array(dtype=wp.vec3f),
    force: wp.array(dtype=wp.vec3f),
    step: wp.int32,
):
    particle_f[0] = force[step]


def ball_world_model() -> wp.sim.Model:
    builder = wp.sim.ModelBuilder()

    builder.add_particle(
        pos=wp.vec3(-0.5, 2.0, 0.0), 
        vel=wp.vec3(0.0, 0.0, 0.0), 
        mass=1.0, radius=0.1
    )
    model = builder.finalize(requires_grad=True)
    model.ground = True

    return model

def generate_segments(sim_steps: int, num_segments: int):
    segment_size = sim_steps // num_segments
    segments = []
    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size
        segments.append({"start": start, "end": end})
    return segments


class BallOptim:
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
        self.learning_rate = 8e-1

        self.use_cuda_graph = wp.get_device().is_cuda

        self.model = ball_world_model()
        self.init_state = self.model.state()

        # Create the integrator and renderer
        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.renderer = wp.sim.render.SimRenderer(
            self.model, OUTPUT_FILE, scaling=1.0
        )

        # ======================== OPTIMIZATION ========================

        self.loss = wp.array([0], dtype=wp.float32, requires_grad=True)
        self.states = [
            self.model.state(requires_grad=True) for _ in range(self.sim_steps + 1)
        ]
        self.controls = [
            self.model.control(clone_variables=True, requires_grad=True)
            for _ in range(self.sim_steps)
        ]

        random_force = np.random.uniform(-1, 1, size=(self.sim_steps, 3))
        self.force = wp.array(random_force, dtype=wp.vec3f, requires_grad=True)
        self.trajectory = wp.empty(self.sim_steps, dtype=wp.vec3f, requires_grad=True)

        t = np.linspace(0, self.sim_duration, self.sim_steps)
        x = np.full_like(t, 0.5)
        z = 2 * np.sin(1.5 * np.pi * t)
        y = np.zeros_like(t)
        self.target_force = wp.array(np.vstack([x, y, z]).T, dtype=wp.vec3f)
        self.target_trajectory = wp.empty(
            self.sim_steps, dtype=wp.vec3f, requires_grad=True
        )

        self.best_loss = np.inf
        self.best_force = wp.array(random_force, dtype=wp.vec3f, requires_grad=True)
        self.best_trajectory = wp.empty(
            self.sim_steps, dtype=wp.vec3f, requires_grad=True
        )

        self.optimizer = wp.optim.Adam(
            [self.force],
            lr=self.learning_rate,
        )

        self.optim_segments = generate_segments(self.sim_steps, 10)

        self.fig, self.ax = create_3d_figure()
        padding = 0.1
        self.xlim = (-0.5 - padding, 0.05 + padding)
        self.ylim = (-0.04 - padding, 2.0 + padding)
        self.zlim = (0.0 - padding, 0.36 + padding)
        self.limits = [self.xlim, self.ylim, self.zlim]
        plt.ion()

    def _reset(self):
        self.sim_time = 0.0
        self.loss = wp.array([0], dtype=wp.float32, requires_grad=True)

    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            self.states[i].clear_forces()
            wp.copy(
                dest=self.states[i].particle_f,
                src=self.target_force,
                dest_offset=0,
                src_offset=i,
                count=1,
            )
            wp.sim.collide(self.model, self.states[i])
            self.integrator.simulate(
                self.model, self.states[i], self.states[i + 1], self.frame_dt
            )
            update_trajectory(self.target_trajectory, self.states[i].particle_q, i, 0)

            if i % self.cfg.sim.sim_substeps == 0:
                self.sim_time += self.frame_dt
                self.render(i)

    def simulate(self, segment: dict):
        for i in range(segment["start"], segment["end"]):
            self.states[i].clear_forces()
            wp.launch(
                apply_force_kernel,
                dim=1,
                inputs=[self.states[i].particle_f, self.force, i],
            )
            wp.sim.collide(self.model, self.states[i])
            self.integrator.simulate(
                self.model, self.states[i], self.states[i + 1], self.frame_dt
            )
            update_trajectory(self.trajectory, self.states[i].particle_q, i, 0)

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
        for segment in self.optim_segments:
            for i in range(100):
                loss = self.step(segment=segment)

                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_force = wp.clone(self.force)
                    self.best_trajectory = wp.clone(self.trajectory)

                if i % 10 == 0:
                    print(f"Iteration {i}, Loss: {loss}")
                    update_plot(self.ax, self.trajectory, self.target_trajectory, self.limits)

        print(f"Best Loss: {self.best_loss}")
        self.render_trajectory(
            f"trajectory_{i}",
            self.best_trajectory,
            radius=0.01,
            color=(0.0, 1.0, 0.0),
            time_offset=0,
        )

    def render_trajectory(
        self,
        name: str,
        trajectory: wp.array,
        radius: float = 0.1,
        color: tuple = (1.0, 0.0, 0.0),
        time_offset: int = 0,
    ):
        t = 0.0
        trajectory = trajectory.numpy()
        for i in range(2, self.num_frames):
            traj = trajectory[:i]
            self.renderer.begin_frame(time_offset + t)
            render_trajectory(
                name=name,
                trajectory=traj,
                renderer=self.renderer,
                radius=radius,
                color=color,
            )
            self.renderer.end_frame()
            t += self.frame_dt

    def render(self, step: int):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.states[step])

            if step > 2:
                render_trajectory(
                    name="trajectory",
                    trajectory=self.target_trajectory,
                    renderer=self.renderer,
                    radius=0.01,
                )

            self.renderer.end_frame()


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def ball_optimization(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    model = BallOptim(cfg)
    model.generate_target_trajectory()
    model.train()
    model.renderer.save()


if __name__ == "__main__":
    ball_optimization()
