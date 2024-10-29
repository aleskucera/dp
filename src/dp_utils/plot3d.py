from typing import Tuple, Union
from dataclasses import dataclass

import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from dp_utils.trajectory import Trajectory

@dataclass
class Curve:
    name: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    color: str = 'red'
    linewidth: int = 2


class Plot3D:
    def __init__(self, 
                 xlim: Tuple[float, float], 
                 ylim: Tuple[float, float], 
                 zlim: Tuple[float, float],
                 padding: float = 0.1):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.base_xlim = xlim
        self.base_ylim = ylim
        self.base_zlim = zlim
        self.padding = padding

        self.x_label = "X"
        self.y_label = "Y"
        self.z_label = "Z"

        self.curves = {-1: []}

    def _update_limits(self, frame: int):
        x = np.concatenate([curve.x for curve in self.curves[frame] + self.curves[-1]])
        y = np.concatenate([curve.y for curve in self.curves[frame] + self.curves[-1]])
        z = np.concatenate([curve.z for curve in self.curves[frame] + self.curves[-1]])

        xlim = (min(x.min(), self.base_xlim[0]) - self.padding, 
                max(x.max(), self.base_xlim[1]) + self.padding)
        ylim = (min(y.min(), self.base_ylim[0]) - self.padding,
                max(y.max(), self.base_ylim[1]) + self.padding)
        zlim = (min(z.min(), self.base_zlim[0]) - self.padding,
                max(z.max(), self.base_zlim[1]) + self.padding)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)

    def _find_valid_frame(self, frame: int):
        f = frame
        while f not in self.curves and f >= -1:
            f -= 1
        return f

    def add_curve(self, name: str, data: np.ndarray, frame: int = -1, color: str = 'red', linewidth: int = 2):
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        if frame in self.curves:
            self.curves[frame].append(Curve(name, x, y, z, color, linewidth))
        else:
            self.curves[frame] = [Curve(name, x, y, z, color, linewidth)]

    def add_trajectory(self, trajectory: Trajectory, frame: int = -1):
        data = trajectory.data.numpy()
        self.add_curve(trajectory.name, data, frame, trajectory.color, trajectory.plot_linewidth)
    
    def animate(self, num_frames: int, interval: int = 100, save_path: str = None):
        def update(frame):
            self.ax.cla()

            valid_frame = self._find_valid_frame(frame)

            self._update_limits(valid_frame)
            for curve in self.curves[valid_frame] + self.curves[-1]:
                self.ax.plot(curve.x, curve.y, curve.z, color=curve.color, linewidth=curve.linewidth, label=curve.name)
            self.ax.set_xlabel(self.x_label)
            self.ax.set_ylabel(self.y_label)
            self.ax.set_zlabel(self.z_label)
            return []

        anim = FuncAnimation(self.fig, update, frames=num_frames, interval=interval)
        plt.show()

        if save_path is not None:
            if save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000 // interval, dpi=300)
            else:
                raise ValueError("Save path must be a .mp4 file.")

# Demo with Sample Data
if __name__ == "__main__":
    plotter = Plot3D(xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5))
    t = np.linspace(0, 2 * np.pi, 100)

    # Adding curves for multiple frames
    for i in range(10):
        x = np.sin(t + 0.1 * i) * (1 + 0.1 * i)
        y = np.cos(t + 0.1 * i) * (1 + 0.1 * i)
        z = t - np.pi

        data = np.vstack([x, y, z]).T
        plotter.add_curve(name=f"Spiral_{i}", data=data, frame=i, color=plt.cm.viridis(i / 10), linewidth=2)
        if i == 9:
            plotter.add_curve(name="Static", data=data, frame=-1, color=plt.cm.viridis(i / 10), linewidth=2)

    # Run the animation
    plotter.animate(num_frames=10, interval=200, save_path="spiral.mp4")


        
            

# def create_3d_figure() -> Tuple[plt.Figure, plt.Axes]:
#     """
#     Creates a 3D figure for plotting trajectories using Matplotlib.
    
#     Returns:
#         plt.Figure: 3D figure for plotting trajectories.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     return fig, ax

# def update_plot(ax: plt.Axes, 
#                 trajectory: Union[list, np.ndarray, wp.array], 
#                 target_trajectory: Union[list, np.ndarray, wp.array],
#                 limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-1, 1), (-1, 1), (-1, 1))):
#     """
#     Updates the 3D plot with the robot's trajectory and the target trajectory.

#     Args:
#         ax (plt.Axes): Matplotlib axes for the 3D plot.
#         trajectory (Union[list, np.ndarray, wp.array]): List or array of 3D trajectory points.
#         target_trajectory (Union[list, np.ndarray, wp.array]): List or array of 3D target trajectory points.
#     """

#     # Convert the data to NumPy arrays
#     trajectory = convert_to_numpy(trajectory, "Trajectory")
#     target_trajectory = convert_to_numpy(target_trajectory, "Target Trajectory")

#     # Determine the limits of the plot
#     x = np.concatenate([trajectory[:, 0], target_trajectory[:, 0]])
#     y = np.concatenate([trajectory[:, 1], target_trajectory[:, 1]])
#     z = np.concatenate([trajectory[:, 2], target_trajectory[:, 2]])

#     xlim = (min(x.min(), limits[0][0]), max(x.max(), limits[0][1]))
#     ylim = (min(y.min(), limits[1][0]), max(y.max(), limits[1][1]))
#     zlim = (min(z.min(), limits[2][0]), max(z.max(), limits[2][1]))

#     # print(f"X: {target_trajectory[:, 0].min()} - {target_trajectory[:, 0].max()}")
#     # print(f"Y: {target_trajectory[:, 1].min()} - {target_trajectory[:, 1].max()}")
#     # print(f"Z: {target_trajectory[:, 2].min()} - {target_trajectory[:, 2].max()}")

#     ax.cla()
#     plot_path(ax, trajectory, color='red', linewidth=2)
#     plot_path(ax, target_trajectory, color='blue', linewidth=2)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     ax.set_zlim(zlim)
#     ax.set_title("3D Plot")
#     plt.draw()
#     plt.pause(0.1)  # Pause to allow the plot to refresh


# def plot_path(ax: plt.Axes, trajectory: np.ndarray, color: str = 'red', linewidth: int = 2):
#     """
#     Plots a 3D trajectory using Matplotlib.

#     Args:
#         trajectory (np.ndarray): List or array of 3D trajectory points.
#         color (str): Color for the trajectory path. Defaults to 'red'.
#     """
#     assert isinstance(trajectory, np.ndarray), "Trajectory must be a numpy array."
#     assert trajectory.ndim == 2, "Trajectory must be a 2D array."
#     assert trajectory.shape[1] == 3, "Trajectory must be a 3D array."

#     # Extract X, Y, Z coordinates from the trajectory
#     x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
#     ax.plot(x, y, z, color=color, marker=None, linestyle='-', linewidth=linewidth)


# def convert_to_numpy(data: Union[list, np.ndarray, wp.array], name: str = "Data") -> np.ndarray:
#     """
#     Converts a list or Warp array to a NumPy array.

#     Args:
#         data (Union[list, np.ndarray, wp.array]): Data to convert.

#     Returns:
#         np.ndarray: NumPy array containing the data.
#     """
#     if isinstance(data, list):
#         return np.array(data)
#     elif isinstance(data, wp.array):
#         return data.numpy()
#     elif isinstance(data, np.ndarray):
#         return data
#     else:
#         raise ValueError(f"{name} must be a list, numpy array, or warp array.")