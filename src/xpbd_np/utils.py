import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def rotate_vector_with_quaternion(q, v):
    rotation = R.from_quat(q)
    return rotation.apply(v)


def multiply_quaternions(q1, q2):
    rotation1 = R.from_quat(q1)
    rotation2 = R.from_quat(q2)
    result_rotation = rotation1 * rotation2  # Quaternion multiplication
    return result_rotation.as_quat()


class AnimationController3D:
    def __init__(self, x: np.ndarray, q: np.ndarray, dims: np.ndarray, num_iters: int, r: np.ndarray = None, titles: list = None):
        self.x = x                          # 3D positions for each frame
        self.q = q                          # 3D quaternions for each frame
        self.r = r                          # Initial orientation vectors
        self.dims = dims                    # Dimensions of each body
        self.num_iters = num_iters          # Number of iterations
        self.current_frame = 0              # Start at the first frame
        self.titles = titles if titles else [f"Iteration {i + 1}" for i in range(num_iters)]

        # Set up the plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.update_plot()  # Display the first frame

    def on_key_press(self, event):
        if event.key == 'n':  # Next frame
            self.current_frame = (self.current_frame + 1) % self.num_iters
            self.update_plot()

        elif event.key == 'p':  # Previous frame
            self.current_frame = (self.current_frame - 1) % self.num_iters
            self.update_plot()

    def update_plot(self):
        self.ax.cla()
        self.ax.set_title(self.titles[self.current_frame])
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-5, 5)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Plot both bodies with their orientation and dimensions
        colors = ["blue", "orange"]
        for j in range(2):
            self.plot_body(
                self.x[self.current_frame][j],  # Center of the body
                self.q[self.current_frame][j],  # Quaternion for orientation
                self.r[j],  # Initial orientation vector
                self.dims[j],  # Dimensions of the body
                colors[j]
            )

        plt.draw()

    def plot_body(self, center, quaternion, r, dimensions, color):
        # Rotate initial orientation vector r with quaternion
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        z = np.array([0, 0, 1])

        x_rot = rotate_vector_with_quaternion(quaternion, x)
        y_rot = rotate_vector_with_quaternion(quaternion, y)
        z_rot = rotate_vector_with_quaternion(quaternion, z)

        # Plot the center and orientation in 3D space
        self.ax.plot([center[0]], [center[1]], [center[2]], 'o', color=color, label=f'Center {color}')
        self.ax.quiver(center[0], center[1], center[2], x_rot[0], x_rot[1], x_rot[2], color='r', label='X')
        self.ax.quiver(center[0], center[1], center[2], y_rot[0], y_rot[1], y_rot[2], color='g', label='Y')
        self.ax.quiver(center[0], center[1], center[2], z_rot[0], z_rot[1], z_rot[2], color='b', label='Z')

        # Rotate initial orientation vector r with quaternion
        r_rotated = R.from_quat(quaternion).apply(r)

        # Plot the center and orientation in 3D space
        self.ax.quiver(center[0], center[1], center[2], r_rotated[0], r_rotated[1], r_rotated[2],
                       color=color)

        # Create a cube or cuboid to represent the body in 3D
        self.plot_cuboid(center, quaternion, dimensions, color)

    def plot_cuboid(self, center, quaternion, dimensions, color):
        # Define cuboid corners around the origin, then rotate and translate
        l, w, h = dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2
        corners = np.array([
            [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
            [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]
        ])

        # Rotate each corner point by the quaternion and translate
        rotation = R.from_quat(quaternion)
        rotated_corners = rotation.apply(corners) + center

        # Plot edges between cuboid corners
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical edges
        ]

        for edge in edges:
            start, end = rotated_corners[edge[0]], rotated_corners[edge[1]]
            self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)
