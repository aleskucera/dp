import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import pytest

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1).type(q1.dtype)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the conjugate of a quaternion.
    q = [w, x, y, z] -> q* = [w, -x, -y, -z]
    """
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1).type(q.dtype)


def rotate_vector(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by quaternion q.
    v_rotated = q * v * q*
    """
    v_quat = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    temp = quat_mul(q, v_quat)
    q_conj = quat_inv(q)
    rotated = quat_mul(temp, q_conj)
    return rotated[..., 1:].type(v.dtype)

def rotate_vector_inverse(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by the inverse of quaternion q.
    v_rotated = q* * v * q
    """
    v_quat = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_inv = quat_inv(q)
    temp = quat_mul(q_inv, v_quat)
    rotated = quat_mul(temp, q)
    return rotated[..., 1:].type(v.dtype)

def rotvec_to_quat(rotation_vector: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation vector to quaternion.
    rotation_vector = angle * axis, where axis is normalized.
    quaternion = [w, x, y, z]

    Args:
        rotation_vector: tensor of shape [..., 3]
    Returns:
        quaternion: tensor of shape [..., 4]
    """
    angle = torch.norm(rotation_vector, dim=-1, keepdim=True)

    # Handle zero rotation
    mask = angle.squeeze(-1) < 1e-8
    if mask.any():
        quat = torch.zeros((*rotation_vector.shape[:-1], 4), device=rotation_vector.device)
        quat[..., 0] = 1.0  # w component = 1 for zero rotation
        if mask.all():
            return quat

    axis = rotation_vector / (angle + 1e-8)  # add epsilon to avoid division by zero

    half_angle = angle / 2
    sin_half = torch.sin(half_angle)

    quat = torch.zeros((*rotation_vector.shape[:-1], 4), device=rotation_vector.device)
    quat[..., 0] = torch.cos(half_angle).squeeze(-1)  # w
    quat[..., 1:] = axis * sin_half

    # Handle zero rotation case
    if mask.any():
        quat[mask, 0] = 1.0
        quat[mask, 1:] = 0.0

    return quat.type(rotation_vector.dtype)


def quat_to_rotvec(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation vector.
    quaternion = [w, x, y, z]
    rotation_vector = angle * axis

    Args:
        quaternion: tensor of shape [..., 4]
    Returns:
        rotation_vector: tensor of shape [..., 3]
    """
    # Normalize quaternion
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    # Extract w component and vector part
    w = quaternion[..., 0]
    xyz = quaternion[..., 1:]

    # Calculate angle
    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))

    # Handle small angles to avoid numerical instability
    small_angle_mask = angle < 1e-8

    # Calculate axis
    sin_half_angle = torch.sqrt(1 - w * w + 1e-8)  # add epsilon to avoid nan
    axis = xyz / (sin_half_angle.unsqueeze(-1) + 1e-8)

    # Calculate rotation vector
    rotation_vector = axis * angle.unsqueeze(-1)

    # Handle small angles
    if small_angle_mask.any():
        rotation_vector[small_angle_mask] = xyz[small_angle_mask] * 2

    return rotation_vector.type(quaternion.dtype)


def numerical_angular_velocity(q: torch.Tensor, q_prev: torch.Tensor, dt: float):
    q_prev_inv = quat_inv(q_prev)
    q_rel = quat_mul(q, q_prev_inv)
    omega = 2 * torch.tensor([q_rel[1], q_rel[2], q_rel[3]]) / dt
    if q_rel[0] < 0:
        omega = -omega
    return omega

def relative_rotation(q: torch.Tensor, q_init: torch.Tensor):
    q_rel = quat_mul(quat_inv(q_init), q)
    return q_rel

def delta_q_fixed(q_a: torch.Tensor, q_b: torch.Tensor):
    dq = quat_mul(q_a, quat_inv(q_b))
    return dq

def normalize_quat(q: torch.Tensor):
    return q / torch.norm(q)