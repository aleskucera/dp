from typing import Tuple, Union
from dataclasses import dataclass

import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

@dataclass
class TimeSeries:
    name: str
    time: np.ndarray
    values: np.ndarray
    color: str = 'red'
    linewidth: int = 2


class SubPlot2D:
    def __init__(self, 
                 name: str,
                 ax: plt.Axes,
                 xlim: Tuple[float, float] = (0, 1),
                 ylim: Tuple[float, float] = (0, 0),
                 padding: float = 0.1):
        self.ax = ax
        self.name = name

        self.base_xlim = xlim
        self.base_ylim = ylim
        self.padding = padding

        self.x_label = "Time"
        self.y_label = "Value"

        self.time_series = {-1: []}

    def _update_limits(self, frame: int):
        x = np.concatenate([time_series.time for time_series in self.time_series[frame] + self.time_series[-1]])
        y = np.concatenate([time_series.values for time_series in self.time_series[frame] + self.time_series[-1]])

        xlim = (min(x.min(), self.base_xlim[0]) - self.padding, 
                max(x.max(), self.base_xlim[1]) + self.padding)
        ylim = (min(y.min(), self.base_ylim[0]) - self.padding,
                max(y.max(), self.base_ylim[1]) + self.padding)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def _find_valid_frame(self, frame: int):
        f = frame
        while f not in self.time_series and f >= -1:
            f -= 1
        return f

    def add_time_series(self, 
                        name: str, 
                        time: np.ndarray, 
                        values: np.ndarray, 
                        frame: int = -1, 
                        color: str = 'red', 
                        linewidth: int = 2):
        if frame in self.time_series:
            self.time_series[frame].append(TimeSeries(name, time, values, color, linewidth))
        else:
            self.time_series[frame] = [TimeSeries(name, time, values, color, linewidth)]

    def update(self, frame: int):
        self.ax.cla()

        valid_frame = self._find_valid_frame(frame)

        self._update_limits(valid_frame)
        for time_series in self.time_series[valid_frame] + self.time_series[-1]:
            self.ax.plot(time_series.time, 
                         time_series.values, 
                         color=time_series.color, 
                         linewidth=time_series.linewidth, 
                         label=time_series.name)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.name)


class Plot2D:
    def __init__(self,
                 subplots: Tuple[str],
                 xlim: Tuple[float, float] = (0, 1),
                 ylim: Tuple[float, float] = (0, 0),
                 padding: float = 0.1):
        num_subplots = len(subplots)
        if num_subplots < 4:
            self.fig, self.axs = plt.subplots(num_subplots, 1, figsize=(8, num_subplots * 6))
        elif num_subplots == 4:
            self.fig, self.axs = plt.subplots(2, 2, figsize=(16, 12))
        else:
            raise ValueError("Number of subplots must be between 1 and 4.")
        
        self.subplots = {name: SubPlot2D(name, ax, xlim, ylim, padding) for name, ax in zip(subplots, self.axs)}

    def add_time_series(self,
                        name: str,
                        time: np.ndarray,
                        data: np.ndarray,
                        subplots: list,
                        frame: int = -1,
                        color: str = 'red',
                        linewidth: int = 2):
        if len(subplots) == 1 and data.ndim == 1:
            self.subplots[subplots[0]].add_time_series(name, time, data, frame, color, linewidth)
        elif len(subplots) > 1 and data.ndim == 2:
            for i, subplot in enumerate(subplots):
                self.subplots[subplot].add_time_series(name, time, data[i], frame, color, linewidth)


    def animate(self, num_frames: int, interval: int = 100, save_path: str = None):
        def update(frame):
            for subplot in self.subplots.values():
                subplot.update(frame)
            plt.tight_layout()
            return []

        anim = FuncAnimation(self.fig, update, frames=num_frames, interval=interval)
        plt.show()

        if save_path is not None:
            if save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000 // interval, dpi=300)
            else:
                raise ValueError("Save path must be a .mp4 file.")
            
if __name__ == "__main__":
    # Initialize Plot2D with names for subplots
    plotter = Plot2D(subplots=("Plot 1", "Plot 2", "Plot 3"), xlim=(0, 10), ylim=(-1, 1), padding=0.1)

    # Generate sample time-series data
    time = np.linspace(0, 10, 100)

    # Adding time series data for two subplots
    for i in range(10):  # 10 frames
        values_1 = np.sin(time + 0.1 * i)  # Sinusoidal wave, changing over time
        values_2 = np.cos(time + 0.2 * i)  # Cosine wave, changing over time

        # Add data to each subplot for each frame
        plotter.add_time_series(name=f"Sine_{i}", time=time, data=values_1, subplots=["Plot 1"], frame=i, color="blue", linewidth=2)
        plotter.add_time_series(name=f"Cosine_{i}", time=time, data=values_2, subplots=["Plot 2"], frame=i, color="green", linewidth=2)
        plotter.add_time_series(name=f"Sine_{i}", time=time, data=values_1, subplots=["Plot 3"], frame=i, color="red", linewidth=2)

    # Run the animation and save as mp4 (if desired)
    plotter.animate(num_frames=10, interval=200, save_path="timeseries_animation.mp4")
