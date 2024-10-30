from typing import Tuple, Union
from dataclasses import dataclass

import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from dp_utils.trajectory import Trajectory
from dp_utils.colors import *

@dataclass
class Curve:
    name: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    line_width: int = 2
    color: Color = RED 


class Plot3D:
    def __init__(self,
                 x_lim: Tuple[float, float],
                 y_lim: Tuple[float, float],
                 z_lim: Tuple[float, float],
                 padding: float = 0.1):
        self.fig, self.ax = None, None

        self.base_x_lim = x_lim
        self.base_y_lim = y_lim
        self.base_z_lim = z_lim
        self.padding = padding

        self.x_label = "X"
        self.y_label = "Y"
        self.z_label = "Z"

        self.curves = {-1: []}

    def create_figure(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def _update_limits(self, frame: int):
        x = np.concatenate([curve.x for curve in self.curves[frame] + self.curves[-1]])
        y = np.concatenate([curve.y for curve in self.curves[frame] + self.curves[-1]])
        z = np.concatenate([curve.z for curve in self.curves[frame] + self.curves[-1]])

        x_lim = (min(x.min(), self.base_x_lim[0]) - self.padding,
                 max(x.max(), self.base_x_lim[1]) + self.padding)
        y_lim = (min(y.min(), self.base_y_lim[0]) - self.padding,
                 max(y.max(), self.base_y_lim[1]) + self.padding)
        z_lim = (min(z.min(), self.base_z_lim[0]) - self.padding,
                 max(z.max(), self.base_z_lim[1]) + self.padding)

        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.ax.set_zlim(z_lim)

    def _find_valid_frame(self, frame: int):
        f = frame
        while f not in self.curves and f >= -1:
            f -= 1
        return f

    def add_curve(self,
                  name: str,
                  data: np.ndarray,
                  frame: int = -1,
                  color: Color = RED,
                  line_width: int = 2):
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        if frame in self.curves:
            self.curves[frame].append(Curve(name, x, y, z, line_width, color))
        else:
            self.curves[frame] = [Curve(name, x, y, z, line_width, color)]

    def add_trajectory(self, trajectory: Trajectory, frame: int = -1):
        data = trajectory.data.numpy()
        self.add_curve(trajectory.name, data, frame, trajectory.color, trajectory.plot_line_width)

    def animate(self, num_frames: int, interval: int = 100, save_path: str = None):
        self.create_figure()

        def update(frame):
            self.ax.cla()

            valid_frame = self._find_valid_frame(frame)

            self._update_limits(valid_frame)
            for curve in self.curves[valid_frame] + self.curves[-1]:
                self.ax.plot(curve.x, curve.y, curve.z, color=curve.color.rgb, 
                             linewidth=curve.line_width, label=curve.name)
            self.ax.set_xlabel(self.x_label)
            self.ax.set_ylabel(self.y_label)
            self.ax.set_zlabel(self.z_label)
            return []

        print("Creating 3D animation...")
        anim = FuncAnimation(self.fig, update, frames=num_frames, interval=interval)
        plt.show()

        if save_path is not None:
            if save_path.endswith('.mp4'):
                print(f"Saving animation to {save_path}")
                anim.save(save_path, writer='ffmpeg', fps=1000 // interval, dpi=200)
            else:
                raise ValueError("Save path must be a .mp4 file.")
        
        plt.close(self.fig)
        self.fig, self.ax = None, None



# Demo with Sample Data
if __name__ == "__main__":
    plotter = Plot3D(x_lim=(-5, 5), y_lim=(-5, 5), z_lim=(-5, 5))
    t = np.linspace(0, 2 * np.pi, 100)

    # Adding curves for multiple frames
    for i in range(10):
        x = np.sin(t + 0.1 * i) * (1 + 0.1 * i)
        y = np.cos(t + 0.1 * i) * (1 + 0.1 * i)
        z = t - np.pi

        data = np.vstack([x, y, z]).T
        plotter.add_curve(name=f"Spiral_{i}",
                          data=data,
                          frame=i,
                          color=BLUE,
                          line_width=2)
        if i == 9:
            plotter.add_curve(name="Static",
                              data=data,
                              frame=-1,
                              color=ORANGE,
                              line_width=2)

    # Run the animation
    plotter.animate(num_frames=10, interval=200, save_path="plot3d.mp4")
