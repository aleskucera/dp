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

from optim.loss import add_trajectory_loss


OUTPUT_FILE = "data/output/ball.usd"

@wp.kernel
def apply_force_kernel(
    particle_f: wp.array(dtype=wp.vec3f),
    force: wp.array(dtype=wp.vec3f),
    step: wp.int32,
):
    particle_f[0] = force[step]


def ball_world_model() -> wp.sim.Model:
    builder = wp.sim.ModelBuilder(gravity=0.0)

    builder.add_particle(
        pos=wp.vec3(-0.5, 2.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0, radius=0.1
    )
    model = builder.finalize(requires_grad=True)
    model.ground = True

    return model


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

        self.time = np.linspace(0, self.sim_duration, self.sim_steps)
        self.iter = 0
        self.profiler = {}
        self.sim_time = 0.0
        self.learning_rate = 5e-1

        self.use_cuda_graph = wp.get_device().is_cuda

        self.model = ball_world_model()
        self.init_state = self.model.state()

        # Create the integrator and renderer
        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
        self.renderer = Renderer(self.model, self.time, OUTPUT_FILE)

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

        t = np.linspace(0, self.sim_duration, self.sim_steps)
        x = np.full_like(t, 0.5)
        y = 2 * np.sin(1.5 * np.pi * t)
        z = np.zeros_like(t)
        self.target_force = wp.array(np.vstack([x, y, z]).T, dtype=wp.vec3f)

        self.trajectory = Trajectory(name="trajectory",
                                     time=self.time,
                                     requires_grad=True,
                                     color=(1.0, 0.0, 0.0))
        
        self.target_trajectory = Trajectory(name="target_trajectory",
                                            time=self.time,
                                            color=(0.0, 1.0, 0.0))

        self.optimizer = wp.optim.Adam(
            [self.force],
            lr=self.learning_rate,
        )

        self.plot3d = Plot3D(x_lim=(-0.5, 0.18),
                             y_lim=(2.0, 2.57),
                             z_lim=(0.0, 0.0),
                             padding=0.1)
        
        self.plot2d = Plot2D(subplots=("x", "y", "z"))
                             

    def _reset(self):
        self.sim_time = 0.0
        self.loss = wp.array([0.0], dtype=wp.float32, requires_grad=True)

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
            self.target_trajectory.update_position(i, self.states[i].particle_q, 0)

        self.plot2d.add_trajectory(self.target_trajectory)
        self.plot3d.add_trajectory(self.target_trajectory)
        self.renderer.add_trajectory(self.target_trajectory)

    def simulate(self):
        for i in range(self.sim_steps):
            self.states[i].clear_forces()
            wp.launch(
                apply_force_kernel,
                dim=1,
                inputs=[self.states[i].particle_f, self.force, i],
            )
            wp.sim.collide(self.model, self.states[i])
            self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.frame_dt)
            self.trajectory.update_position(i, self.states[i].particle_q, 0)

    def train(self):
        for i in range(300):
            loss = self.step()
            if i % 5 == 0:
                print(f"Iteration {i}, Loss: {loss}")
                self.plot2d.add_trajectory(self.trajectory, i)
                self.plot3d.add_trajectory(self.trajectory, i)

            if i == 249:
                self.renderer.add_trajectory(self.trajectory)

    def step(self):
        self.tape = wp.Tape()
        self._reset()
        with self.tape:
            self.simulate()
            add_trajectory_loss(self.trajectory, self.target_trajectory, self.loss)
        self.tape.backward(self.loss)
        self.optimizer.step(grad=[self.force.grad])
        self.tape.zero()

        return self.loss.numpy()[0]
    
    def render_usd(self):
        self.renderer.render(self.states)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def carter_demo(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    model = BallOptim(cfg)
    model.generate_target_trajectory()
    model.train()
    model.render_usd()
    model.plot3d.animate(num_frames=300, interval=100, save_path="data/output/ball_trajectory.mp4")
    model.plot2d.animate(num_frames=300, interval=100, save_path="data/output/ball_trajectory_2d.mp4")


if __name__ == "__main__":
    carter_demo()
