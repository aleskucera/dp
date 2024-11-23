import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, NamedTuple

from xpbd_pytorch.box import Box, Sphere
from xpbd_pytorch.body import Body
from xpbd_pytorch.quat import *
from xpbd_pytorch.cylinder import Cylinder

import matplotlib
matplotlib.use('TkAgg')

MAX_COLLISIONS = 30

class AnimationController:
    def __init__(self, boxes: List[Body], time: torch.Tensor, step: int = 1):
        """
        Initialize the animation controller for multiple boxes.

        Args:
            boxes (list of Box): List of Box objects.
        """
        self.time = time
        self.step = step
        self.boxes = boxes
        self.current_frame = 0
        self.current_box_index = None  # None means show all boxes

        # Set up the plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.update_plot()  # Display the first frame

    def on_key_press(self, event):
        if event.key == 'n':  # Next frame
            self.current_frame = (self.current_frame + self.step) % len(self.time)
            self.update_plot()

        elif event.key == 'p':  # Previous frame
            self.current_frame = (self.current_frame - self.step) % len(self.time)
            self.update_plot()

        elif event.key in map(str, range(len(self.boxes))):  # Visualize a specific box
            self.current_box_index = int(event.key)
            self.update_plot()

        elif event.key == 'a':  # Visualize all boxes
            self.current_box_index = None
            self.update_plot()

    def update_plot(self):
        self.ax.cla()
        self.ax.set_title(f"Time: {self.time[self.current_frame]:.2f}")
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
        self.ax.set_zlim(-4, 4)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        if self.current_box_index is None:  # Show all boxes
            for box in self.boxes:
                box.plot(self.ax, self.current_frame)
        else:  # Show only the selected box
            self.boxes[self.current_box_index].plot(self.ax, self.current_frame)

        plt.draw()

# Example usage remains the same, but now with more detailed collision detection:
if __name__ == "__main__":
    # Create sample trajectory data
    n_frames = 200
    time_len = 2 * np.pi
    dt = time_len / n_frames
    time = torch.linspace(0, time_len, n_frames)

    box = Box(m=1.0,
              I=torch.eye(3),
              x=torch.tensor([0.0, 0.0, 0.6]),
              q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
              v=torch.tensor([0.0, 0.0, 0.0]),
              w=torch.tensor([0.0, 0.0, 3.0]),
              hx=0.5,
              hy=0.5,
              hz=0.5,
              restitution=0.6,
              static_friction=0.3,
              dynamic_friction=0.02,
              collision_density=3,
              render_color=(0.0, 0.0, 1.0),
              sim_time=time,
              dt=dt)

    sphere = Sphere(m=1.0,
                    I=torch.eye(3),
                    x=torch.tensor([0.0, 0.0, 2.0]),
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    v=torch.tensor([0.0, 1.0, 0.0]),
                    w=torch.tensor([0.0, 1.0, 1.0]),
                    radius=1.0,
                    restitution=0.6,
                    static_friction=0.3,
                    dynamic_friction=0.02,
                    collision_point_density=30,
                    render_color=(0.0, 1.0, 0.0),
                    sim_time=time,
                    dt=dt)

    cylinder = Cylinder(m=1.0,
                        x=torch.tensor([0.0, 0.0, 2.1]),
                        q=torch.tensor([0.7071, 0.7071, 0.0, 0.0]),
                        # q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        v=torch.tensor([0.0, 0.0, 0.0]),
                        w=torch.tensor([0.0, 2.0, 0.0]),
                        radius=2.0,
                        height=0.75,
                        restitution=0.6,
                        static_friction=0.4,
                        dynamic_friction=1.0,
                        collision_point_density=15,
                        render_color=(0.0, 0.0, 1.0),
                        sim_time=time,
                        dt=dt)

    num_pos_iters = 2

    for i in range(n_frames):
        box.integrate()
        box.detect_collisions()
        for j in range(num_pos_iters):
            box.correct_collisions()
        box.update_velocity()
        box.solve_velocity()
        box.save_state(i)
        box.save_collisions(i)
    controller = AnimationController([box], time, step=3)

    # for i in range(n_frames):
    #     sphere.integrate()
    #     sphere.detect_collisions()
    #     for j in range(num_pos_iters):
    #         sphere.correct_collisions()
    #     sphere.update_velocity()
    #     sphere.solve_velocity()
    #     sphere.save_state(i)
    #     sphere.save_collisions(i)
    # controller = AnimationController([sphere], time, step=3)

    # for i in range(n_frames):
    #     cylinder.integrate()
    #     cylinder.detect_collisions()
    #     for j in range(num_pos_iters):
    #         cylinder.correct_collisions()
    #     cylinder.update_velocity()
    #     cylinder.solve_velocity()
    #     cylinder.save_state(i)
    #     cylinder.save_collisions(i)
    # controller = AnimationController([cylinder], time, step=1)

    plt.show()
