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

class AnimationController:
    def __init__(self, x, q, r, a, num_iters):
        self.x = x
        self.q = q
        self.a = a
        self.r = r
        self.num_iters = num_iters
        self.current_frame = 0  # Start at the first frame

        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.update_plot()  # Display the first frame

    def on_key_press(self, event):

        # Check if the 'n' key was pressed
        if event.key == 'n':
            # Advance to the next frame if possible
            if self.current_frame < self.num_iters - 1:
                self.current_frame += 1
            else:
                self.current_frame = 0  # Loop back to the first frame
            self.update_plot()

        if event.key == 'p':
            # Go back to the previous frame if possible
            if self.current_frame > 0:
                self.current_frame -= 1
            else:
                self.current_frame = self.num_iters - 1
            self.update_plot()


    def update_plot(self):
        self.ax.clear()
        self.ax.set_title(f"Iteration {self.current_frame + 1}")
        self.ax.set_xlim(0, 15)
        self.ax.set_ylim(0, 15)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')

        # Plot both bodies with their orientation and rectangles
        colors = ["blue", "orange"]
        for j in range(2):
            self.plot_body(
                self.ax,
                self.x[self.current_frame][j],  # Center of the body
                self.q[self.current_frame][j],  # Quaternion for orientation
                self.r[j],  # Relative position vector
                self.a[j],  # Dimensions of the body
                colors[j]
            )

        plt.draw()

    def plot_body(self, ax, center, quaternion, r, dimensions, color):
        # Draw center point
        ax.plot(center[0], center[1], 'o', color=color, label=f'Center {color}')
        
        # Draw orientation vector from center point
        r = rotate_vector_with_quaternion(quaternion, r)
        ax.quiver(center[0], center[1], r[0], r[1], color=color, scale=1, scale_units='xy')

        # Calculate rotation angle in degrees from quaternion
        angle = R.from_quat(quaternion).as_euler('zxy', degrees=True)[0]

        # Create the Rectangle patch with rotation about the center
        rect = Rectangle(
            (center[0] - dimensions[0] / 2, center[1] - dimensions[1] / 2),  # Bottom-left corner based on center
            dimensions[0],
            dimensions[1],
            linewidth=1,
            edgecolor=color,
            facecolor='none',
            angle=float(angle),
            rotation_point='center'  # Rotate around the rectangle's center
        )

        # Add the rectangle to the plot
        ax.add_patch(rect)

def apply_positional_correction(x1, x2, q1, q2, m1, m2, r1, r2, d_lambda, n, I1, I2):
    """
    Apply a positional correction based on equations (6)-(9) in the paper using quaternion math.

    Parameters:
    - x1, x2: np.array of shape (3,), positions of body 1 and body 2
    - q1, q2: np.array of shape (4,), quaternions of body 1 and body 2 (in [x, y, z, w] format)
    - m1, m2: float, masses of body 1 and body 2
    - r1, r2: np.array of shape (3,), relative positions of the constraint from the center of mass of each body
    - d_lambda: float, Lagrange multiplier update (correction magnitude)
    - n: np.array of shape (3,), direction of correction (normalized vector)
    - I1, I2: np.array of shape (3,3), inertia tensors of body 1 and body 2

    Returns:
    - Updated positions and quaternions of both bodies.
    """
    # Calculate the positional impulse
    p = d_lambda * n

    # Update positions (x1, x2)
    x1_new = x1 + (p / m1)
    x2_new = x2 - (p / m2)

    # Compute rotational impulse corrections as angular velocities
    omega1 = np.linalg.inv(I1) @ np.cross(r1, p)
    if np.linalg.norm(omega1) == 0:
        q1_new = q1
    else:
        omega_1_norm = np.linalg.norm(omega1)
        q1_correction = np.hstack([np.sin(omega_1_norm / 2) * omega1 / omega_1_norm, np.cos(omega_1_norm / 2)])
        q1_new = multiply_quaternions(q1_correction, q1)

    # If omega2 is non-zero, compute the correction for body 2
    omega2 = np.linalg.inv(I2) @ np.cross(r2, p)
    if np.linalg.norm(omega2) == 0:
        q2_new = q2
    else:
        omega_2_norm = np.linalg.norm(omega2)
        q2_correction = np.hstack([np.sin(-omega_2_norm / 2) * omega2 / omega_2_norm, np.cos(-omega_2_norm / 2)])
        q2_new = multiply_quaternions(q2_correction, q2)

    # Normalize the updated quaternions
    q1_new /= np.linalg.norm(q1_new)
    q2_new /= np.linalg.norm(q2_new)

    return x1_new, x2_new, q1_new, q2_new

def compute_d_lambda(r1, r2, n, c, m1, m2, I1, I2):
    r1xn = np.cross(r1, n)
    r2xn = np.cross(r2, n)

    w1 = 1/m1 + r1xn @ np.linalg.inv(I1) @ r1xn
    w2 = 1/m2 + r2xn @ np.linalg.inv(I2) @ r2xn

    return -(c / (w1 + w2))

def compute_correction_vector(x1, x2, q1, q2, r1, r2):
    # Rotate relative positions by respective quaternions
    r1_w = R.from_quat(q1).apply(r1) + x1
    r2_w = R.from_quat(q2).apply(r2) + x2

    # Compute the difference in the transformed positions
    diff = r1_w - r2_w
    magnitude = np.linalg.norm(diff)
    return diff / magnitude, magnitude  # Return direction (normalized) and magnitude

def main():
    num_iters = 10
    
    # Define the first rectangular body
    m1 = 1.0                        # Mass
    x1 = np.array([10.0, 4.0, 0.0])  # Position of the center of mass
    q1 = np.array([0, 0, 0, 1])     # Quaternion representing no rotation
    r1 = np.array([0.0, -1.0, 0.0]) # Relative position of the constraint from the center of mass
    a1 = np.array([2.0, 2.0, 0.0])  # Dimensions of the body (2D)
    I1 = np.array([[(m1 * a1[0] ** 2) / 6, 0, 0],
                [0, (m1 * a1[1] ** 2) / 6, 0],
                [0, 0, 1]])         # Inertia tensor (the third dimension is ignored)

    # Define the second rectangular body
    m2 = 1.0                        # Mass
    x2 = np.array([11.0, 2.0, 0.0]) # Position of the center of mass
    q2 = np.array([0, 0, 0, 1])     # Quaternion representing no rotation
    r2 = np.array([-1.0, 0.0, 0.0]) # Relative position of the constraint from the center of mass
    a2 = np.array([2.0, 2.0, 2.0])  # Dimensions of the body (2D)
    I2 = np.array([[(m2 * a2[0] ** 2) / 6, 0, 0],
                [0, (m2 * a2[1] ** 2) / 6, 0],
                [0, 0, 1]])         # Inertia tensor (the third dimension is ignored)

    a = np.array([a1, a2])  # Store the dimensions of the bodies
    r = np.array([r1, r2])  # Store the relative positions of the constraints
    x = np.zeros((num_iters, 2, 3), dtype=np.float32)  # Array to store the positions of the bodies
    q = np.zeros((num_iters, 2, 4), dtype=np.float32)  # Array to store the quaternions of the bodies

    # Store the initial positions and rotations    
    x[0] = np.array([x1, x2])
    q[0] = np.array([q1, q2])

    for i in range(num_iters - 1):
        print(f"\nIteration {i + 1}:")
        x1, x2 = x[i]
        q1, q2 = q[i]
        # Compute the normalized direction vector n and the correction magnitude c
        n, c = compute_correction_vector(x1, x2, q1, q2, r1, r2)
        print(f"Normalized direction n: {n}")
        print(f"Correction magnitude c: {c}")

        d_lambda = compute_d_lambda(r1, r2, n, c, m1, m2, I1, I2)
        print(f"Lagrange multiplier update d_lambda: {d_lambda}")

        x1_new, x2_new, q1_new, q2_new = apply_positional_correction(x1, x2, q1, q2, m1, m2, r1, r2, d_lambda, n, I1, I2)

        print("Updated positions:")
        print("Body 1:", x1_new)
        print("Body 2:", x2_new)

        print("\nUpdated quaternions:")
        print("Body 1:", q1_new)
        print("Body 2:", q2_new)

        r1_w = R.from_quat(q1_new).apply(r1) + x1_new
        r2_w = R.from_quat(q2_new).apply(r2) + x2_new

        print("\nUpdated relative positions:")
        print("Body 1 r:", r1_w)
        print("Body 2 r:", r2_w)
        print(f"Distance between bodies: {np.linalg.norm(r1_w - r2_w)}")

        # Store the updated positions and rotations
        x[i + 1] = np.array([x1_new, x2_new])
        q[i + 1] = np.array([q1_new, q2_new])

    anim_controller = AnimationController(x, q, r, a, num_iters)
    plt.show()


if __name__ == "__main__":
    main()



