import numpy as np
from scipy.spatial.transform import Rotation as R

from xpbd_np.utils import multiply_quaternions

def angular_velocity_1(q: np.ndarray, q_prev: np.ndarray, dt: float):
    # Compute the quaternion derivative
    q_dot = multiply_quaternions(q, R.from_quat(q_prev).inv().as_quat())
    q_dot = q_dot / np.linalg.norm(q_dot)

    # Compute the angular velocity
    w = 2 * np.arccos(q_dot[0]) * q_dot[1:] / np.linalg.norm(q_dot[1:])
    return w / dt

def angular_velocity_2(q: np.ndarray, q_prev: np.ndarray, dt: float):
    # Compute relative quaternion
    q_rel = multiply_quaternions(q, R.from_quat(q_prev).inv().as_quat())
    
    # Extract angle and axis
    angle = 2 * np.arccos(np.clip(q_rel[0], -1, 1))  # Scalar part of quaternion is q_rel[0]
    axis = q_rel[1:4] / np.linalg.norm(q_rel[1:4])  # Vector part is q_rel[1:4]
    
    # Angular velocity
    angular_velocity = (angle / dt) * axis
    return angular_velocity

def angular_velocity_3(q: np.ndarray, q_prev: np.ndarray, dt: float):
    q_rel = multiply_quaternions(q, R.from_quat(q_prev).inv().as_quat())
    omega = 2 * np.array([q_rel[0], q_rel[1], q_rel[2]]) / dt
    if q_rel[3] < 0:
        omega = -omega
    return omega

def angular_velocity_4(q: np.ndarray, q_prev: np.ndarray, dt: float):
    """https://mariogc.com/post/angular-velocity-quaternions/
    """
    q = np.array([q[3], q[0], q[1], q[2]])
    q_prev = np.array([q_prev[3], q_prev[0], q_prev[1], q_prev[2]])
    return (2 / dt) * np.array([
        q_prev[0] * q[1] - q_prev[1] * q[0] - q_prev[2] * q[3] + q_prev[3] * q[2],
        q_prev[0] * q[2] + q_prev[1] * q[3] - q_prev[2] * q[0] - q_prev[3] * q[1],
        q_prev[0] * q[3] - q_prev[1] * q[2] + q_prev[2] * q[1] - q_prev[3] * q[0]])

# Test setup
def generate_quaternions(axis, angle, dt):
    """Generate two quaternions for a rotation of given axis and angle."""
    q_prev = R.from_quat([0, 0, 0, 1]).as_quat()  # Identity quaternion
    q = R.from_rotvec(axis * angle).as_quat()
    return q, q_prev

def main():
    dt = 1  # Time step
    axis = np.array([0, 1, 1])  # Rotation around Z-axis
    angle = 0.01  # Rotate by 45 degrees in one step

    # Compute ground-truth angular velocity
    omega = angle * axis / dt
    print("Ground-truth Angular Velocity:", omega)

    # Generate test quaternions
    q, q_prev = generate_quaternions(axis, angle, dt)

    # Calculate angular velocities
    omega1 = angular_velocity_1(q, q_prev, dt)
    omega2 = angular_velocity_2(q, q_prev, dt)
    omega3 = angular_velocity_3(q, q_prev, dt)
    omega4 = angular_velocity_4(q, q_prev, dt)

    # Print results
    print("Test Results:")
    print(f"Angular Velocity 1: {omega1}")
    print(f"Angular Velocity 2: {omega2}")
    print(f"Angular Velocity 3: {omega3}")
    print(f"Angular Velocity 4: {omega4}")
    print("\nAre results consistent?")
    print("omega1 == omega (Ground-truth):", np.allclose(omega1, omega))
    print("omega2 == omega (Ground-truth):", np.allclose(omega2, omega))
    print("omega3 == omega (Ground-truth):", np.allclose(omega3, omega))
    print("omega4 == omega (Ground-truth):", np.allclose(omega4, omega))

if __name__ == "__main__":
    main()