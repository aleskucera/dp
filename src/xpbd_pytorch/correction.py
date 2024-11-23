import torch
from setuptools.command.rotate import rotate

from xpbd_pytorch.quat import *
from xpbd_pytorch.utils import *

import matplotlib
matplotlib.use('TkAgg')

def numerical_angular_velocity(q: torch.Tensor, q_prev: torch.Tensor, dt: float):
    q_prev_inv = quaternion_conjugate(q_prev)
    q_rel = quaternion_multiply(q_prev_inv, q)
    omega = 2 * torch.tensor([q_rel[1], q_rel[2], q_rel[3]]) / dt
    if q_rel[0] < 0:
        omega = -omega
    return omega

def numerical_linear_velocity(x: torch.Tensor, x_prev: torch.Tensor, dt: float):
    return (x - x_prev) / dt

def velocity_update(x: torch.Tensor, x_prev: torch.Tensor, q: torch.Tensor, q_prev: torch.Tensor, dt: float):
    v = numerical_linear_velocity(x, x_prev, dt)
    w = numerical_angular_velocity(q, q_prev, dt)
    return v, w

def restitution_delta(q: torch.Tensor,
                      v: torch.Tensor,
                      w: torch.Tensor,
                      v_prev: torch.Tensor,
                      w_prev: torch.Tensor,
                      r: torch.Tensor,
                      n: torch.Tensor,
                      m_inv: float,
                      I_inv: torch.Tensor,
                      restitution: float):
    # Compute the normal component of the relative velocity
    v_rel = v + rotate_vector(torch.linalg.cross(w, r), q)
    v_n_magnitude = torch.dot(v_rel, n)
    v_n = v_n_magnitude * n

    if torch.abs(v_n_magnitude) < 1e-5:
        restitution = 0.0

    # Compute the normal component of the relative velocity before the velocity update
    v_rel_prev = v_prev + rotate_vector(torch.linalg.cross(w_prev, r), q)
    v_n_magnitude_prev = torch.dot(v_rel_prev, n)
    v_n_prev = v_n_magnitude_prev * n

    # Compute the change of velocity due to the restitution
    delta_v_restitution = - (v_n + restitution * v_n_prev)

    # Compute the impulse due to the restitution in the world frame
    nb = rotate_vector_inverse(n, q)  # Rotate the normal to the body frame
    J_restitution = delta_v_restitution / (
                m_inv + torch.dot(torch.linalg.cross(I_inv @ torch.linalg.cross(r, nb), r), nb))
    dv = J_restitution * m_inv
    dw = I_inv @ torch.linalg.cross(r, rotate_vector_inverse(J_restitution, q))

    return dv, dw

def dynamic_friction_delta(q: torch.Tensor,
                   v: torch.Tensor,
                   w: torch.Tensor,
                   r: torch.Tensor,
                   n: torch.Tensor,
                   m_inv: float,
                   I_inv: torch.Tensor,
                   dt: float,
                   dynamic_friction: float,
                   lambda_n: float):
    # Compute the relative velocity of the contact point in the world frame
    v_rel = v + rotate_vector(torch.linalg.cross(w, r), q)

    # Compute the tangent component of the relative velocity
    v_t = v_rel - torch.dot(v_rel, n) * n
    v_t_magnitude = torch.norm(v_t)
    t = v_t / v_t_magnitude

    # Compute the change of velocity due to the friction (now infinite friction)
    coulomb_friction = torch.abs(torch.tensor(dynamic_friction * lambda_n / dt))
    delta_v_friction = - v_t #* torch.min(coulomb_friction, v_t_magnitude) / v_t_magnitude

    # Compute the impulse due to the friction in the world frame
    tb = rotate_vector_inverse(t, q)  # Rotate the tangent to the body frame
    J_friction = delta_v_friction / (
                m_inv + torch.dot(torch.linalg.cross(I_inv @ torch.linalg.cross(r, tb), r), tb))
    dv = J_friction * m_inv
    dw = I_inv @ torch.linalg.cross(r, rotate_vector_inverse(J_friction, q))

    return dv, dw

def position_delta(q: torch.Tensor,
                   r: torch.Tensor,
                   m_inv: float,
                   I_inv: torch.Tensor,
                   delta_x: torch.Tensor):
    c = torch.norm(delta_x)
    n = delta_x / c

    nb = rotate_vector_inverse(n, q)
    d_lambda = - c / (m_inv + torch.dot(torch.linalg.cross(I_inv @ torch.linalg.cross(r, nb), r), nb))
    
    p = d_lambda * n
    pb = d_lambda * nb

    dx = p * m_inv

    w = I_inv @ torch.linalg.cross(r, pb)
    w_quat = torch.cat([torch.tensor([0.0]), w], dim=0)
    dq = 0.5 * quaternion_multiply(q, w_quat)

    return dx, dq, d_lambda

# def velocity_solve(q: torch.Tensor, v: torch.Tensor, w: torch.Tensor, r: torch.Tensor, n: torch.Tensor, m_inv: float, I_inv: torch.Tensor, restitution: float):
#     dv = torch.zeros(3)
#     dw = torch.zeros(3)
#
#     # Compute the relative velocity of the contact point in the world frame
#     v_rel = v + rotate_vector_inverse(torch.linalg.cross(w, r), q)
#
#     # Compute the normal component of the relative velocity
#     v_n_magnitude = torch.dot(v_rel, n)
#     v_n = v_n_magnitude * n
#
#     # Compute the tangent component of the relative velocity
#     v_t = v_rel - v_n
#     v_t_magnitude = torch.norm(v_t)
#     t = v_t / v_t_magnitude
#
#     nb = rotate_vector_inverse(n, q)  # Rotate the normal to the body frame
#     tb = rotate_vector_inverse(t, q)  # Rotate the tangent to the body frame
#
#     # Compute the change of velocity due to the restitution
#     delta_v_restitution = - (1 + restitution) * v_n
#
#     # Compute the impulse due to the restitution in the world frame
#     J_restitution = delta_v_restitution / (m_inv + torch.dot(torch.linalg.cross(I_inv @ torch.linalg.cross(r, nb), r), nb))
#     dv += J_restitution * m_inv
#     dw += I_inv @ torch.linalg.cross(r, rotate_vector_inverse(J_restitution, q))
#
#     # Compute the change of velocity due to the friction (now infinite friction)
#     delta_v_friction = - 0.1 * v_t
#
#     # Compute the impulse due to the friction in the world frame
#     J_friction = delta_v_friction / (m_inv + torch.dot(torch.linalg.cross(I_inv @ torch.linalg.cross(r, tb), r), tb))
#     dv += J_friction * m_inv
#     dw += I_inv @ torch.linalg.cross(r, rotate_vector_inverse(J_friction, q))
#
#     return dv, dw
