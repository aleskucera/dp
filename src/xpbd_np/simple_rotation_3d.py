import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np

matplotlib.use('TkAgg')

# Quaternion rotation
def rotate_vector_with_quaternion(q, v):
    # Initialize the quaternion as a Rotation object
    rotation = R.from_quat(q)  # 'q' is in [x, y, z, w] format
    return rotation.apply(v)   # Rotate vector 'v' by quaternion 'q'

def multiply_quaternions(*quaternions):
    result_rotation = R.from_quat(quaternions[0])  # Start with the first quaternion
    for q in quaternions[1:]:  # Multiply with each subsequent quaternion
        result_rotation *= R.from_quat(q)
    return result_rotation.as_quat()


class AnimationController3D:
    def __init__(self, x, q, a, num_iters):
        self.x = x  # 3D positions for each frame
        self.q = q  # 3D quaternions for each frame
        self.a = a  # Dimensions of each body
        self.num_iters = num_iters
        self.current_frame = 0  # Start at the first frame

        # Set up the plot
        self.fig = plt.figure(figsize=(6, 6))
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
        self.ax.set_title(f"Iteration {self.current_frame + 1}")
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_zlim(-3, 3)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Plot both bodies with their orientation and dimensions
        colors = ["blue", "orange"]
        for j in range(2):
            self.plot_body(
                self.ax,
                self.x[self.current_frame][j],  # Center of the body
                self.q[self.current_frame][j],  # Quaternion for orientation
                self.a[j],  # Dimensions of the body
                colors[j]
            )

        plt.draw()

    def plot_body(self, ax, center, quaternion, dimensions, color):
        # Rotate initial orientation vector r with quaternion
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        z = np.array([0, 0, 1])

        x_rot = rotate_vector_with_quaternion(quaternion, x)
        y_rot = rotate_vector_with_quaternion(quaternion, y)
        z_rot = rotate_vector_with_quaternion(quaternion, z)

        # Plot the center and orientation in 3D space
        ax.plot([center[0]], [center[1]], [center[2]], 'o', color=color, label=f'Center {color}')
        ax.quiver(center[0], center[1], center[2], x_rot[0], x_rot[1], x_rot[2], color='r', label='X')
        ax.quiver(center[0], center[1], center[2], y_rot[0], y_rot[1], y_rot[2], color='g', label='Y')
        ax.quiver(center[0], center[1], center[2], z_rot[0], z_rot[1], z_rot[2], color='b', label='Z')

        # Create a cube or cuboid to represent the body in 3D
        self.plot_cuboid(ax, center, quaternion, dimensions, color)

    def plot_cuboid(self, ax, center, quaternion, dimensions, color):
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
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]

        for edge in edges:
            start, end = rotated_corners[edge[0]], rotated_corners[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)

def main():
    num_frames = 3

    # Define the first rectangular body
    x_a = np.array([0.0, 0.0, 0.0])  # Position of the center of mass
    q_a0 = R.from_euler('z', 45, degrees=True).as_quat()
    dim_a = np.array([2.0, 2.0, 2.0])  # Dimensions of the body

    # Define the second rectangular body
    x_b = np.array([0.1, 0.1, 0.1])  # Position of the center of mass
    q_b0 = R.from_euler('x', 90, degrees=True).as_quat()
    dim_b = np.array([2.0, 2.0, 2.0])  # Dimensions of the body

    q_a0_inv = R.from_quat(q_a0).inv().as_quat()
    q_b0_inv = R.from_quat(q_b0).inv().as_quat()

    # Rotate the bodies by 45 degrees
    q_a1 = R.from_euler('z', 90, degrees=True).as_quat()
    q_b1 = R.from_euler('x', -45, degrees=True).as_quat()

    q_a1_inv = R.from_quat(q_a1).inv().as_quat()
    q_b1_inv = R.from_quat(q_b1).inv().as_quat()

    dims = np.array([dim_a, dim_b])  # Store the dimensions of the bodies
    x = np.zeros((num_frames, 2, 3), dtype=np.float32)  # Array to store the positions of the bodies
    for i in range(num_frames):
        x[i] = np.array([x_a, x_b])

    # Store the quaternions for each body at each frame
    q = np.zeros((num_frames, 2, 4), dtype=np.float32)
    q[0] = np.array([q_a0, q_b0])
    q[1] = np.array([q_a1, q_b1])
    rot_vec1 = R.from_quat(multiply_quaternions(q_a1_inv, q_b1, q_b0_inv, q_a0)).as_rotvec()
    rot_vec2 = R.from_quat(multiply_quaternions(q_b1_inv, q_a1, q_a0_inv, q_b0)).as_rotvec()

    r1 = R.from_quat(R.from_quat(q_a1).inv().as_quat()).apply(rot_vec1)
    r2 = R.from_quat(R.from_quat(q_b1).inv().as_quat()).apply(rot_vec2)

    print(f"r1: {r1}")
    print(f"r2: {r2}")

    theta1 = np.linalg.norm(rot_vec1)
    theta2 = np.linalg.norm(rot_vec2)
    # print(rot_vec)
    n1 = rot_vec1 / theta1
    n2 = rot_vec2 / theta2
    # q[2] = np.array([multiply_quaternions(q_b1, q_b0_inv, q_a0), q_b1])
    q[2][0] = multiply_quaternions(q_a1, R.from_rotvec(0.3 * theta1 * n1).as_quat())
    q[2][1] = multiply_quaternions(q_b1, R.from_rotvec(0.7 * theta2 * n2).as_quat())

    anim_controller = AnimationController3D(x, q, dims, num_frames)
    plt.show()

if __name__ == "__main__":
    main()