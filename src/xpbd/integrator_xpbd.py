import warp as wp
import warp.sim

from sim import Model, State, Control
from sim.integrator import Integrator
from sim.model import ModelShapeMaterials

@wp.func
def linear_multiplier(
        dx: wp.vec3,
        r1: wp.vec3,
        r2: wp.vec3,
        tf1: wp.transform,
        tf2: wp.transform,
        m_inv1: wp.float32,
        m_inv2: wp.float32,
        I_inv1: wp.mat33,
        I_inv2: wp.mat33,
        lambda_: wp.float32,
        compliance: wp.float32,
        dt: wp.float32,
) -> wp.float32:
    c = wp.length(dx)
    if c == 0.0:
        return 0.0
    n = wp.normalize(dx)

    q1 = wp.transform_get_rotation(tf1)
    q2 = wp.transform_get_rotation(tf2)

    # Eq. 2-3 (make sure to project into the frame of the body)
    r1xn = wp.quat_rotate_inv(q1, wp.cross(r1, n))
    r2xn = wp.quat_rotate_inv(q2, wp.cross(r2, n))

    weight1 = m_inv1 + wp.dot(r1xn, I_inv1 * r1xn)
    weight2 = m_inv2 + wp.dot(r2xn, I_inv2 * r2xn)

    alpha = compliance / wp.pow(dt, 2)
    denominator = weight1 + weight2 + alpha
    if denominator == 0.0:
        return 0.0

    numerator = - (c + lambda_ * alpha)
    d_lambda = numerator / denominator
    return lambda_ + d_lambda

@wp.func
def compute_contact_constraint_delta(
    err: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    m_inv_a: float,
    m_inv_b: float,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    linear_a: wp.vec3,
    linear_b: wp.vec3,
    angular_a: wp.vec3,
    angular_b: wp.vec3,
    relaxation: float,
    dt: float,
) -> float:
    denom = 0.0
    denom += wp.length_sq(linear_a) * m_inv_a
    denom += wp.length_sq(linear_b) * m_inv_b

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    delta_lambda = -err
    if denom > 0.0:
        delta_lambda /= dt * denom

    return delta_lambda * relaxation

# @wp.kernel
# def apply_body_deltas(
#     q_in: wp.array(dtype=wp.transform),
#     qd_in: wp.array(dtype=wp.spatial_vector),
#     body_com: wp.array(dtype=wp.vec3),
#     body_I: wp.array(dtype=wp.mat33),
#     body_inv_m: wp.array(dtype=float),
#     body_inv_I: wp.array(dtype=wp.mat33),
#     deltas: wp.array(dtype=wp.spatial_vector),
#     constraint_inv_weights: wp.array(dtype=float),
#     dt: float,
#     # outputs
#     q_out: wp.array(dtype=wp.transform),
#     qd_out: wp.array(dtype=wp.spatial_vector),
# ):
#     tid = wp.tid()
#     m_inv = body_inv_m[tid]
#     if m_inv == 0.0:
#         q_out[tid] = q_in[tid]
#         qd_out[tid] = qd_in[tid]
#         return
#     I_inv = body_inv_I[tid]
#
#     tf = q_in[tid]
#     delta = deltas[tid]
#
#     x = wp.transform_get_translation(tf)
#     q = wp.transform_get_rotation(tf)
#     dx = m_inv * wp.spatial_bottom(delta)
#     dq = I_inv @ wp.spatial_top(delta)
#
#     # Update the position
#     new_x = x + dx * dt
#
#     # Update the orientation
#     if wp.length(dq) > 1e-4:
#         # Create quaternion from the rotation vector
#         q_delta = wp.quat_from_axis_angle(dq, wp.length(dq) * dt)
#         new_q = wp.normalize(q * q_delta)
#     else:
#         new_q = q
#
#     # Update the transform
#     q_out[tid] = wp.transform(new_x, new_q)

@wp.kernel
def apply_body_deltas(
    q_in: wp.array(dtype=wp.transform),
    qd_in: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_I: wp.array(dtype=wp.mat33),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),
    deltas: wp.array(dtype=wp.spatial_vector),
    constraint_inv_weights: wp.array(dtype=float),
    dt: float,
    # outputs
    q_out: wp.array(dtype=wp.transform),
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    inv_m = body_inv_m[tid]
    if inv_m == 0.0:
        q_out[tid] = q_in[tid]
        qd_out[tid] = qd_in[tid]
        return
    inv_I = body_inv_I[tid]

    tf = q_in[tid]
    delta = deltas[tid]

    p0 = wp.transform_get_translation(tf)
    q0 = wp.transform_get_rotation(tf)

    weight = 1.0
    if constraint_inv_weights:
        inv_weight = constraint_inv_weights[tid]
        if inv_weight > 0.0:
            weight = 1.0 / inv_weight

    dp = wp.spatial_bottom(delta) * (inv_m * weight)
    dq = wp.spatial_top(delta) * weight
    dq = wp.quat_rotate(q0, inv_I * wp.quat_rotate_inv(q0, dq))

    # update orientation
    q1 = q0 + 0.5 * wp.quat(dq * dt, 0.0) * q0
    q1 = wp.normalize(q1)

    # update position
    com = body_com[tid]
    x_com = p0 + wp.quat_rotate(q0, com)
    p1 = x_com + dp * dt
    p1 -= wp.quat_rotate(q1, com)

    q_out[tid] = wp.transform(p1, q1)

    v0 = wp.spatial_bottom(qd_in[tid])
    w0 = wp.spatial_top(qd_in[tid])

    # update linear and angular velocity
    v1 = v0 + dp
    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(q0, w0 + dq)
    tb = -wp.cross(wb, body_I[tid] * wb)  # coriolis forces
    w1 = wp.quat_rotate(q0, wb + inv_I * tb * dt)

    # XXX this improves gradient stability
    if wp.length(v1) < 1e-4:
        v1 = wp.vec3(0.0)
    if wp.length(w1) < 1e-4:
        w1 = wp.vec3(0.0)

    qd_out[tid] = wp.spatial_vector(w1, v1)

@wp.kernel
def solve_body_contact_positions(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_thickness: wp.array(dtype=float),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    relaxation: float,
    dt: float,
    contact_torsional_friction: float,
    contact_rolling_friction: float,
    # outputs
    deltas: wp.array(dtype=wp.spatial_vector),
    contact_inv_weight: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    body_a = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    body_b = -1
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    # Check for the self-collision
    if body_a == body_b:
        return

    # Find the body to world transforms
    if body_a >= 0:
        mu_a    = shape_materials.mu[shape_a]
        com_a   = body_com[body_a]
        X_wb_a  = body_q[body_a]
        m_inv_a = body_m_inv[body_a]
        omega_a = wp.spatial_top(body_qd[body_a])
        I_inv_a = body_I_inv[body_a]
    else:
        mu_a    = 0.0
        com_a   = wp.vec3(0.0)
        X_wb_a  = wp.transform_identity()
        m_inv_a = 0.0
        omega_a = wp.vec3(0.0)
        I_inv_a = wp.mat33(0.0)
    if body_b >= 0:
        mu_b    = shape_materials.mu[shape_b]
        com_b   = body_com[body_b]
        X_wb_b  = body_q[body_b]
        m_inv_b = body_m_inv[body_b]
        omega_b = wp.spatial_top(body_qd[body_b])
        I_inv_b = body_I_inv[body_b]
    else:
        mu_b    = 0.0
        com_b   = wp.vec3(0.0)
        X_wb_b  = wp.transform_identity()
        m_inv_b = 0.0
        omega_b = wp.vec3(0.0)
        I_inv_b = wp.mat33(0.0)

    # r_a = contact_point0[tid] + contact_offset0[tid] - body_com[body_a]
    # r_b = contact_point1[tid] + contact_offset1[tid] - body_com[body_b]

    # Transform the contact points to world space
    bx_a = wp.transform_point(X_wb_a, contact_point0[tid])
    bx_b = wp.transform_point(X_wb_b, contact_point1[tid])

    # Compute the depth and the normal of the contact
    n = -contact_normal[tid]
    d = wp.dot(n, bx_b - bx_a) - contact_thickness[tid]
    # If the contact is separating, exit
    if d >= 0.0:
        return

    r_a = bx_a - wp.transform_point(X_wb_a, com_a)
    r_b = bx_b - wp.transform_point(X_wb_b, com_b)

    angular_a = -wp.cross(r_a, n)
    angular_b = wp.cross(r_b, n)

    lambda_n = compute_contact_constraint_delta(
        d, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, -n, n, angular_a, angular_b, relaxation, dt
    )

    lin_delta_a = -n * lambda_n
    lin_delta_b = n * lambda_n
    ang_delta_a = angular_a * lambda_n
    ang_delta_b = angular_b * lambda_n

    # wp.printf("Contact %d: %f\n", tid, lambda_n)
    if body_a >= 0:
        wp.atomic_add(deltas, body_a, wp.spatial_vector(ang_delta_a, lin_delta_a))
    if body_b >= 0:
        wp.atomic_add(deltas, body_b, wp.spatial_vector(ang_delta_b, lin_delta_b))


class XPBDIntegrator(Integrator):
    def __init__(self, angular_damping: float = 0.0):
        self.angular_damping = angular_damping
        self.iterations = 5
        self.rigid_contact_relaxation = 1.0

    # def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None):
    #     self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)

    def apply_body_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        body_deltas: wp.array,
        dt: float,
        rigid_contact_inv_weight: wp.array = None,
    ):
        with wp.ScopedTimer("apply_body_deltas", False):
            body_q = state_out.body_q
            body_qd = state_out.body_qd
            new_body_q = wp.clone(body_q)
            new_body_qd = wp.clone(body_qd)

            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    body_q,
                    body_qd,
                    model.body_com,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    body_deltas,
                    rigid_contact_inv_weight,
                    dt,
                ],
                outputs=[
                    new_body_q,
                    new_body_qd,
                ],
                device=model.device,
            )

            state_out.body_q = new_body_q
            state_out.body_qd = new_body_qd

        return new_body_q, new_body_qd

    def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None):
        body_q = None
        body_qd = None
        body_deltas = None

        rigid_contact_inv_weight = None

        with wp.ScopedTimer("simulate", False):
            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                body_deltas = wp.empty_like(state_out.body_qd)

                self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)

            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    body_deltas.zero_()

                    wp.launch(
                        kernel=solve_body_contact_positions,
                        dim=model.rigid_contact_max,
                        inputs=[
                            body_q,
                            body_qd,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.shape_body,
                            model.rigid_contact_count,
                            model.rigid_contact_point0,
                            model.rigid_contact_point1,
                            model.rigid_contact_offset0,
                            model.rigid_contact_offset1,
                            model.rigid_contact_normal,
                            model.rigid_contact_thickness,
                            model.rigid_contact_shape0,
                            model.rigid_contact_shape1,
                            model.shape_materials,
                            self.rigid_contact_relaxation,
                            dt,
                            model.rigid_contact_torsional_friction,
                            model.rigid_contact_rolling_friction,
                        ],
                        outputs=[
                            body_deltas,
                            rigid_contact_inv_weight,
                        ],
                        device=model.device,
                    )

                    # print(f"Body deltas: {body_deltas.numpy()}")

                    body_q, body_qd = self.apply_body_deltas(
                        model, state_in, state_out, body_deltas, dt, rigid_contact_inv_weight
                    )

                    state_out.body_q.assign(body_q)
                    state_out.body_qd.assign(body_qd)

                    # # update body velocities from position changes
                    # if self.compute_body_velocity_from_position_delta and model.body_count and not requires_grad:
                    #     # causes gradient issues (probably due to numerical problems
                    #     # when computing velocities from position changes)
                    #     if requires_grad:
                    #         out_body_qd = wp.clone(state_out.body_qd)
                    #     else:
                    #         out_body_qd = state_out.body_qd
                    #
                    #     # update body velocities
                    #     wp.launch(
                    #         kernel=update_body_velocities,
                    #         dim=model.body_count,
                    #         inputs=[state_out.body_q, body_q_init, model.body_com, dt],
                    #         outputs=[out_body_qd],
                    #         device=model.device,
                    #     )
                    #
                    #     if model.body_count:
                    #         body_deltas.zero_()
                    #         wp.launch(
                    #             kernel=apply_rigid_restitution,
                    #             dim=model.rigid_contact_max,
                    #             inputs=[
                    #                 state_out.body_q,
                    #                 state_out.body_qd,
                    #                 body_q_init,
                    #                 body_qd_init,
                    #                 model.body_com,
                    #                 model.body_inv_mass,
                    #                 model.body_inv_inertia,
                    #                 model.shape_body,
                    #                 model.rigid_contact_count,
                    #                 model.rigid_contact_normal,
                    #                 model.rigid_contact_shape0,
                    #                 model.rigid_contact_shape1,
                    #                 model.shape_materials,
                    #                 model.rigid_contact_point0,
                    #                 model.rigid_contact_point1,
                    #                 model.rigid_contact_offset0,
                    #                 model.rigid_contact_offset1,
                    #                 model.rigid_contact_thickness,
                    #                 rigid_contact_inv_weight_init,
                    #                 model.gravity,
                    #                 dt,
                    #             ],
                    #             outputs=[
                    #                 body_deltas,
                    #             ],
                    #             device=model.device,
                    #         )
                    #
                    #         wp.launch(
                    #             kernel=apply_body_delta_velocities,
                    #             dim=model.body_count,
                    #             inputs=[
                    #                 body_deltas,
                    #             ],
                    #             outputs=[state_out.body_qd],
                    #             device=model.device,
                    #         )

                    # return state_out