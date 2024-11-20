import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from xpbd_np.utils import multiply_quaternions, AnimationController3D

matplotlib.use('TkAgg')

def apply_positional_correction(x_a, x_b, q_a, q_b, m_a, m_b, r_a, r_b):
    def positional_multiplier(m_a_inv, m_b_inv,  c):
        w_b = m_a_inv
        w_a = m_b_inv
        return -(c / (w_a + w_b))

    def pos_correction_vector(x_a, x_b, q_a, q_b, r_a, r_b):
        # Rotate relative positions by respective quaternions
        r_a_w = R.from_quat(q_a).apply(r_a) + x_a
        r_b_w = R.from_quat(q_b).apply(r_b) + x_b

        # Compute the difference in the transformed positions
        diff = r_a_w - r_b_w
        magnitude = np.linalg.norm(diff)

        return diff, magnitude

    # Compute the inverse masses and inertia tensors
    m_a_inv = 1 / m_a
    m_b_inv = 1 / m_b

    # Compute the correction vector and its magnitude
    d, c = pos_correction_vector(x_a, x_b, q_a, q_b, r_a, r_b)

    d_lambda = positional_multiplier(m_a_inv, m_b_inv, c) * 0.5

    # Update positions (x_a, x_b)
    x_a_new = x_a + d_lambda * m_a_inv * d
    x_b_new = x_b - d_lambda * m_b_inv * d

    q_a_new = q_a
    q_b_new = q_b

    return x_a_new, x_b_new, q_a_new, q_b_new


def apply_rotational_correction(q_a, q_b, I_a, I_b):
    def rot_correction_vector(q_a, q_b):
        # u's are now the z-axis of the body's initial orientation
        u_a = R.from_quat(q_a).apply(np.array([0, 1, 0]))
        u_b = R.from_quat(q_b).apply(np.array([0, 1, 0]))

        # Compute the correction vector
        n = np.cross(u_b, u_a)
        theta = np.linalg.norm(n)
        return n, theta

    n, theta = rot_correction_vector(q_a, q_b)

    n_a = R.from_quat(q_a).apply(n)
    n_b = R.from_quat(q_b).apply(n)

    w1 = n_a @ np.linalg.inv(I_a) @ n_a
    w2 = n_a @ np.linalg.inv(I_b) @ n_b
    w = w1 + w2

    if np.abs(w) < 1e-6:
        return q_a, q_b

    theta_a = w1 / (w1 + w2) * theta
    theta_b = - w2 / (w1 + w2) * theta

    q_a_correction = R.from_rotvec(theta_a * n_a).as_quat()
    q_b_correction = R.from_rotvec(theta_b * n_b).as_quat()

    q_a_new = multiply_quaternions(q_a, q_a_correction)
    q_b_new = multiply_quaternions(q_b, q_b_correction)

    return q_a_new, q_b_new


def main():
    num_iters = 21

    # Define the first rectangular body
    m_a = 1.0  # Mass
    x_a = np.array([0.0, 1.0, 0.0])  # Position of the center of mass
    q_a = R.from_euler('z', 180, degrees=True).as_quat()
    r_a = np.array([0.1, 1.0, 0.0])  # Relative position of the constraint from the center of mass
    dims_a = np.array([2.0, 2.0, 2.0])  # Dimensions of the body (2D)
    I_a = np.array([[(m_a * dims_a[0] ** 2) / 6, 0, 0],
                    [0, (m_a * dims_a[1] ** 2) / 6, 0],
                    [0, 0, (m_a * dims_a[2] ** 2) / 6]])  # Inertia tensor

    # Define the second rectangular body
    m_b = 1.0  # Mass
    x_b = np.array([0.0, -1.5, -1.0])  # Position of the center of mass
    q_b = R.from_euler('z', 30, degrees=True).as_quat()
    r_b = np.array([1.0, 0.5, 1.0])  # Relative position of the constraint from the center of mass
    dims_b = np.array([2.0, 1.0, 2.0])  # Dimensions of the body (2D)
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

    for i in range(0, num_iters - 1, 2):
        print(f"\nIteration {i + 1}:")
        x_a, x_b = x[i]
        q_a, q_b = q[i]

        q_a, q_b = apply_rotational_correction(q_a, q_b, I_a, I_b)
        x[i + 1] = np.array([x_a, x_b])
        q[i + 1] = np.array([q_a, q_b])

        x_a, x_b, q_a, q_b = apply_positional_correction(x_a, x_b, q_a, q_b, m_a, m_b, r_a, r_b)

        r1_w = R.from_quat(q_a).apply(r_a) + x_a
        r2_w = R.from_quat(q_b).apply(r_b) + x_b

        print("\nUpdated relative positions:")
        print(f"Distance between points: {np.linalg.norm(r1_w - r2_w)}")

        # Store the updated positions and rotations
        x[i + 2] = np.array([x_a, x_b])
        q[i + 2] = np.array([q_a, q_b])

    _ = AnimationController3D(x=x, q=q, r=r,
                              dims=dims, num_iters=num_iters)
    plt.show()


if __name__ == "__main__":
    main()
