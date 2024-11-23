from typing import Tuple, List

import torch
import matplotlib.pyplot as plt

from xpbd_pytorch.body import Body
from xpbd_pytorch.quat import *

import matplotlib
matplotlib.use('TkAgg')

MAX_COLLISIONS = 30

class AnimationController:
    def __init__(self,
                 bodies: List[Body],
                 time: torch.Tensor,
                 x_lims: Tuple[float, float] = (-4, 4),
                 y_lims: Tuple[float, float] = (-4, 4),
                 z_lims: Tuple[float, float] = (-4, 4)):
        self.time = time
        self.bodies = bodies

        self.current_frame = 0
        self.current_body_index = None  # None means show all boxes

        self.x_lims = x_lims
        self.y_lims = y_lims
        self.z_lims = z_lims

        # Set up the plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.update_plot()  # Display the first frame

    def on_key_press(self, event):
        if event.key == 'n':  # Next frame
            self.current_frame = (self.current_frame + 1) % len(self.time)
            self.update_plot()
            
        elif event.key == 'N':  # Jump forward by 10 frames
            self.current_frame = (self.current_frame + 10) % len(self.time)
            self.update_plot()

        elif event.key == 'b':  # Previous frame
            self.current_frame = (self.current_frame - 1) % len(self.time)
            self.update_plot()

        elif event.key == 'B':  # Jump backward by 10 frames
            self.current_frame = (self.current_frame - 10) % len(self.time)
            self.update_plot()

        elif event.key in map(str, range(len(self.bodies))):  # Visualize a specific box
            self.current_body_index = int(event.key)
            self.update_plot()

        elif event.key == 'a':  # Visualize all boxes
            self.current_body_index = None
            self.update_plot()

    def update_plot(self):
        self.ax.cla()
        self.ax.set_title(f"Time: {self.time[self.current_frame]:.2f}")
        self.ax.set_xlim(*self.x_lims)
        self.ax.set_ylim(*self.y_lims)
        self.ax.set_zlim(*self.z_lims)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        if self.current_body_index is None:  # Show all boxes
            for box in self.bodies:
                box.plot(self.ax, self.current_frame)
        else:  # Show only the selected box
            self.bodies[self.current_body_index].plot(self.ax, self.current_frame)

        # Make the axis equal
        self.ax.set_box_aspect([1, 1, 1])

        plt.draw()

    @staticmethod
    def start():
        plt.show()
