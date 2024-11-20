import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

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

def apply_rotational_correction(q_a, q_b, I_a, I_b, n_a, theta_a, n_b, theta_b):
    w1 = np.linalg.norm(np.linalg.inv(I_a) @ n_a)
    w2 = np.linalg.norm(np.linalg.inv(I_b) @ n_b)

    theta_a = w1 / (w1 + w2) * theta_a
    theta_b = w2 / (w1 + w2) * theta_b

    q_a_correction = R.from_rotvec(theta_a * n_a).as_quat()
    q_b_correction = R.from_rotvec(theta_b * n_b).as_quat()

    q_a_new = multiply_quaternions(q_a, q_a_correction)
    q_b_new = multiply_quaternions(q_b, q_b_correction)

    return q_a_new, q_b_new

def compute_correction_vectors(q_a, q_b, q_a0, q_b0):
    q_a_inv = R.from_quat(q_a).inv().as_quat()
    q_b_inv = R.from_quat(q_b).inv().as_quat()
    q_a0_inv = R.from_quat(q_a0).inv().as_quat()
    q_b0_inv = R.from_quat(q_b0).inv().as_quat()

    rot_vec_a = R.from_quat(multiply_quaternions(q_a_inv, q_b, q_b0_inv, q_a0)).as_rotvec()
    rot_vec_b = R.from_quat(multiply_quaternions(q_b_inv, q_a, q_a0_inv, q_b0)).as_rotvec()

    theta_a = np.linalg.norm(rot_vec_a)
    theta_b = np.linalg.norm(rot_vec_b)
    n_a = rot_vec_a / theta_a
    n_b = rot_vec_b / theta_b
    return n_a, theta_a, n_b, theta_b

def main():
    num_frames = 3

    # Define the first cuboid body
    m_a = 1.0                        # Mass
    x_a = np.array([0.0, 0.0, 0.0])  # Position of the center of mass
    q_a0 = R.from_euler('z', 45, degrees=True).as_quat()
    dim_a = np.array([2.0, 2.0, 2.0])  # Dimensions of the body
    I_a = np.array([[(m_a * dim_a[0] ** 2) / 6, 0, 0],
                    [0, (m_a * dim_a[1] ** 2) / 6, 0],
                    [0, 0, (m_a * dim_a[2] ** 2) / 6]])  # Inertia tensor

    # Define the second cuboid body
    m_b = 1.0                        # Mass
    x_b = np.array([0.1, 0.1, 0.1])  # Position of the center of mass
    q_b0 = R.from_euler('x', 90, degrees=True).as_quat()
    dim_b = np.array([2.0, 2.0, 2.0])  # Dimensions of the body
    I_b = np.array([[(m_b * dim_b[0] ** 2) / 6, 0, 0],
                    [0, (m_b * dim_b[1] ** 2) / 6, 0],
                    [0, 0, (m_b * dim_b[2] ** 2) / 6]])  # Inertia tensor

    # Rotate the bodies by 45 degrees
    q_a1 = R.from_euler('z', 90, degrees=True).as_quat()
    q_b1 = R.from_euler('x', -45, degrees=True).as_quat()

    dims = np.array([dim_a, dim_b])  # Store the dimensions of the bodies
    x = np.zeros((num_frames, 2, 3), dtype=np.float32)  # Array to store the positions of the bodies
    for i in range(num_frames):
            x[i] = np.array([x_a, x_b])

    # Store the quaternions for each body at each frame
    q = np.zeros((num_frames, 2, 4), dtype=np.float32)
    q[0] = np.array([q_a0, q_b0])
    q[1] = np.array([q_a1, q_b1])

    n_a, theta_a, n_b, theta_b = compute_correction_vectors(q_a1, q_b1, q_a0, q_b0)
    q[2] = apply_rotational_correction(q_a1, q_b1, I_a, I_b, n_a, theta_a, n_b, theta_b)

    _ = AnimationController3D(x, q, dims, num_frames)
    plt.show()

if __name__ == "__main__":
    main()



