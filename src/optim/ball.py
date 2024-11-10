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


USD_FILE = "data/output/ball.usd"
PLOT2D_FILE = "data/output/ball_2d.mp4"
PLOT3D_FILE = "data/output/ball_3d.mp4"

@wp.kernel
def apply_force_kernel(
    body_f: wp.array(dtype=wp.spatial_vectorf),
    force: wp.array(dtype=wp.vec3f),
    step: wp.int32,
):  
    
    body_f[0][3] = force[step][0]
    body_f[0][4] = force[step][1]
    body_f[0][5] = force[step][2]


def target_force(t: np.ndarray) -> np.ndarray:
    x = np.full_like(t, 0.01)
    y = 0.05 * np.sin(3 * np.pi * t)
    # x = np.full_like(t, 0.0)
    # y = np.full_like(t, 0.0)    
    z = np.zeros_like(t)
    return np.vstack([x, y, z]).T

def random_force(t: np.ndarray) -> np.ndarray:
    return np.random.uniform(-1, 1, size=(len(t), 3))

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

        self.model = ball_world_model(gravity=True)

        self.time = np.linspace(0, self.sim_duration, self.sim_steps)

        # Create the integrator and renderer
        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.renderer = Renderer(self.model, self.time, USD_FILE)

        self.loss = wp.array([0], dtype=wp.float32, requires_grad=True)
        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]
        self.target_states = [self.model.state() for _ in range(self.sim_steps + 1)]

        self.force = wp.array(random_force(self.time), dtype=wp.vec3f, requires_grad=True)
        self.target_force = wp.array(target_force(self.time), dtype=wp.vec3f)

        self.trajectory = Trajectory("trajectory", self.time, color=BLUE, requires_grad=True)
        self.target_trajectory = Trajectory("target_trajectory", self.time, color=ORANGE)

        self.epoch = 0
        self.num_epochs = 1
        self.optimizer = wp.optim.Adam([self.force], lr=5e-1)

        self.plot3d = Plot3D(x_lim=(-0.5, 0.18),
                             y_lim=(0.0, 0.0),
                             z_lim=(2.0, 2.57),
                             padding=0.1)
        
        self.plot2d = Plot2D(subplots=("x", "y", "z"))
                             

    def _reset(self):
        self.loss = wp.array([0.0], dtype=wp.float32, requires_grad=True)

    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            curr_state = self.target_states[i]
            next_state = self.target_states[i + 1]
            curr_state.clear_forces()
            wp.launch(apply_force_kernel, dim=1, inputs=[curr_state.body_f, self.target_force, i])
            wp.sim.collide(self.model, curr_state)
            self.integrator.simulate(self.model, curr_state, next_state, self.frame_dt)
            print(f"Body q: {curr_state.body_q.numpy()}")
            self.target_trajectory.update_position(i, curr_state.body_q, 0)

        self.plot2d.add_trajectory(self.target_trajectory)
        self.plot3d.add_trajectory(self.target_trajectory)
        self.renderer.add_trajectory(self.target_trajectory)

    def simulate(self):
        for i in range(self.sim_steps):
            curr_state = self.states[i]
            next_state = self.states[i + 1]
            curr_state.clear_forces()
            wp.launch(apply_force_kernel, dim=1, inputs=[curr_state.body_f, self.force, i])
            wp.sim.collide(self.model, curr_state)
            self.integrator.simulate(self.model, curr_state, next_state, self.frame_dt)
            self.trajectory.update_position(i, curr_state.body_q, 0)

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

    def train(self):
        epoch = 0

        for _ in range(self.num_epochs):
            loss = self.step()
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
                self.plot2d.add_trajectory(self.trajectory, epoch)
                self.plot3d.add_trajectory(self.trajectory, epoch)

            epoch += 1
            
        self.renderer.add_trajectory(self.trajectory)
    
    def save_usd(self):
        self.renderer.save(self.target_states)

    def animate_2d_plot(self, save_path: str = None):
        self.plot2d.animate(num_frames=300, interval=100, save_path=save_path)

    def animate_3d_plot(self, save_path: str = None):
        self.plot3d.animate(num_frames=300, interval=100, save_path=save_path)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def ball_optimization(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", custom_eval)
    OmegaConf.resolve(cfg)

    model = BallOptim(cfg)
    model.generate_target_trajectory()
    model.train()
    # model.animate_2d_plot(save_path=PLOT2D_FILE)
    model.save_usd()
    


if __name__ == "__main__":
    ball_optimization()
