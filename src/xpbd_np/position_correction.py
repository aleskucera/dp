import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use('TkAgg')

# Quaternion rotation
def rotate_vector_with_quaternion(q, v):
    # Initialize the quaternion as a Rotation object
    rotation = R.from_quat(q)  # 'q' is in [x, y, z, w] format
    return rotation.apply(v)   # Rotate vector 'v' by quaternion 'q'

def multiply_quaternions(q1, q2):
    rotation1 = R.from_quat(q1)
    rotation2 = R.from_quat(q2)
    result_rotation = rotation1 * rotation2  # Quaternion multiplication
    return result_rotation.as_quat()

class AnimationController3D:
    def __init__(self, x, q, r, a, num_iters):
        self.x = x  # 3D positions for each frame
        self.q = q  # 3D quaternions for each frame
        self.r = r  # Initial orientation vectors
        self.a = a  # Dimensions of each body
        self.num_iters = num_iters
        self.current_frame = 0  # Start at the first frame

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
        self.ax.set_title(f"Iteration {self.current_frame + 1}")
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
                self.ax,
                self.x[self.current_frame][j],  # Center of the body
                self.q[self.current_frame][j],  # Quaternion for orientation
                self.r[j],  # Initial orientation vector
                self.a[j],  # Dimensions of the body
                colors[j]
            )

        plt.draw()

    def plot_body(self, ax, center, quaternion, r, dimensions, color):

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

        # Rotate initial orientation vector r with quaternion
        r_rotated = self.rotate_vector_with_quaternion(quaternion, r)

        # Plot the center and orientation in 3D space
        ax.quiver(center[0], center[1], center[2], r_rotated[0], r_rotated[1], r_rotated[2],
                  color=color)

        # Create a cube or cuboid to represent the body in 3D
        self.plot_cuboid(ax, center, quaternion, dimensions, color)

    def rotate_vector_with_quaternion(self, quaternion, vector):
        # Rotate a vector using a quaternion
        rotation = R.from_quat(quaternion)
        return rotation.apply(vector)

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

def apply_positional_correction(x_a, x_b, q_a, q_b, m_a, m_b, I_a, I_b, r_a, r_b):
    # Compute the inverse masses and inertia tensors
    m_a_inv = 1 / m_a
    m_b_inv = 1 / m_b
    I_a_inv = np.linalg.inv(I_a)
    I_b_inv = np.linalg.inv(I_b)

    # Compute the correction vector and its magnitude
    d, c = compute_correction_vector(x_a, x_b, q_a, q_b, r_a, r_b)

    d_a = R.from_quat(q_a).inv().apply(d)
    d_b = R.from_quat(q_b).inv().apply(d)

    rxd_a = np.cross(r_a, d_a)
    rxd_b = np.cross(r_b, d_b)

    d_lambda = compute_d_lambda(m_a_inv, m_b_inv, I_a_inv, I_b_inv, rxd_a, rxd_b, c) * 0.8

    # Update positions (x_a, x_b)
    x_a_new = x_a + d_lambda * m_a_inv * d
    x_b_new = x_b - d_lambda * m_b_inv * d

    # Compute rotations (q_a, q_b)
    omega_a = d_lambda * I_a_inv @ rxd_a
    if np.linalg.norm(omega_a) == 0:
        q_a_new = q_a
    else:
        q_a_new = multiply_quaternions(q_a, R.from_rotvec(omega_a).as_quat())

    omega_b = - d_lambda * I_b_inv @ rxd_b
    if np.linalg.norm(omega_b) == 0:
        q_b_new = q_b
    else:
        q_b_new = multiply_quaternions(q_b, R.from_rotvec(omega_b).as_quat())

    return x_a_new, x_b_new, q_a_new, q_b_new

def compute_d_lambda(m_a_inv, m_b_inv, I_a_inv, I_b_inv, rxd_a, rxd_b, c):
    w_b = m_a_inv + rxd_a @ I_a_inv @ rxd_a
    w_a = m_b_inv + rxd_b @ I_b_inv @ rxd_b
    return -(c / (w_a + w_b))

def compute_correction_vector(x_a, x_b, q_a, q_b, r_a, r_b):
    # Rotate relative positions by respective quaternions
    r_a_w = R.from_quat(q_a).apply(r_a) + x_a
    r_b_w = R.from_quat(q_b).apply(r_b) + x_b

    # Compute the difference in the transformed positions
    diff = r_a_w - r_b_w
    magnitude = np.linalg.norm(diff)

    return diff, magnitude

def main():
    num_iters = 15
    
    # Define the first rectangular body
    m_a = 1.0                        # Mass
    x_a = np.array([1.0, 1.0, 3.0])  # Position of the center of mass
    q_a = R.from_euler('z', 90, degrees=True).as_quat()
    r_a = np.array([0.0, 2.0, 0.0]) # Relative position of the constraint from the center of mass
    dims_a = np.array([2.0, 2.0, 2.0])  # Dimensions of the body (2D)
    I_a = np.array([[(m_a * dims_a[0] ** 2) / 6, 0, 0],
                [0, (m_a * dims_a[1] ** 2) / 6, 0],
                [0, 0, (m_a * dims_a[2] ** 2) / 6]])  # Inertia tensor

    # Define the second rectangular body
    m_b = 1.0                        # Mass
    x_b = np.array([-1.0, 2.0, 1.0]) # Position of the center of mass
    q_b = R.from_euler('x', 45, degrees=True).as_quat()
    r_b = np.array([1.0, 1.0, 1.0]) # Relative position of the constraint from the center of mass
    dims_b = np.array([2.0, 2.0, 2.0])  # Dimensions of the body (2D)
    I_b = np.array([[(m_b * dims_b[0] ** 2) / 6, 0, 0],
                [0, (m_b * dims_b[1] ** 2) / 6, 0],
                [0, 0, (m_b * dims_b[2] ** 2) / 6]])  # Inertia tensor

    r = np.array([r_a, r_b])  # Store the relative positions of the constraints
    dims = np.array([dims_a, dims_b])  # Store the dimensions of the bodies
    x = np.zeros((num_iters, 2, 3), dtype=np.float32)  # Array to store the positions of the bodies
    q = np.zeros((num_iters, 2, 4), dtype=np.float32)  # Array to store the quaternions of the bodies

    # Store the initial positions and rotations    
    x[0] = np.array([x_a, x_b])
    q[0] = np.array([q_a, q_b])

    for i in range(num_iters - 1):
        print(f"\nIteration {i + 1}:")
        x_a, x_b = x[i]
        q_a, q_b = q[i]

        x1_new, x2_new, q1_new, q2_new = apply_positional_correction(x_a, x_b, q_a, q_b, m_a, m_b, I_a, I_b, r_a, r_b)

        r1_w = R.from_quat(q1_new).apply(r_a) + x1_new
        r2_w = R.from_quat(q2_new).apply(r_b) + x2_new

        print("\nUpdated relative positions:")
        print(f"Distance between points: {np.linalg.norm(r1_w - r2_w)}")

        # Store the updated positions and rotations
        x[i + 1] = np.array([x1_new, x2_new])
        q[i + 1] = np.array([q1_new, q2_new])

    anim_controller = AnimationController3D(x, q, r, dims, num_iters)
    plt.show()


if __name__ == "__main__":
    main()



