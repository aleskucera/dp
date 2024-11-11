import warp as wp

@wp.kernel
def solve_ground_collisions(
    balls: wp.array(dtype=BodyInfo),
    body_q: wp.array(dtype=wp.transform)):
    ball = balls[wp.tid()]
    pos = wp.transform_get_translation(body_q[ball.idx])
    if pos[2] - 0.1 < 0.0:
        pos[2] = 0.0 + 0.1
    body_q[ball.idx] = wp.transform(pos, wp.transform_get_rotation(body_q[ball.idx]))

@wp.func
def compute_positional_correction_multiplier(
    dx: wp.vec3,
    r1: wp.vec3,
    r2: wp.vec3,
    tf1: wp.transform,
    tf2: wp.transform,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    lambda_in: float,
    compliance: float,
    dt: float,
) -> float:
    
    # If the distance is zero, no correction is needed
    c = wp.length(dx)
    if c == 0.0:
        return 0.0

    n = wp.normalize(dx)

    # Eq. 2-3 (make sure to project into the frame of the body???)
    r1xn = wp.cross(r1, n)
    r2xn = wp.cross(r2, n)

    w1 = m_inv1 + wp.dot(r1xn, wp.dot(I_inv1, r1xn))
    w2 = m_inv2 + wp.dot(r2xn, wp.dot(I_inv2 * r2xn))

    w = w1 + w2
    if w == 0.0:
        return 0.0

    # Eq. 4    
    alpha = compliance * wp.pow(dt, 2)
    d_lambda = (-c - alpha * lambda_in) / (w + alpha)

    return d_lambda


@wp.kernel
def positional_correction_kernel():
    pass


@wp.func
def compute_angular_correction_3d(
    corr: wp.vec3,
    q1: wp.quat,
    q2: wp.quat,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    alpha_tilde: float,
    # lambda_prev: float,
    relaxation: float,
    dt: float,
):
    # compute and apply the correction impulse for an angular constraint
    theta = wp.length(corr)
    if theta == 0.0:
        return 0.0

    n = wp.normalize(corr)

    # project variables to body rest frame as they are in local matrix
    n1 = wp.quat_rotate_inv(q1, n)
    n2 = wp.quat_rotate_inv(q2, n)

    # Eq. 11-12
    w1 = wp.dot(n1, I_inv1 * n1)
    w2 = wp.dot(n2, I_inv2 * n2)
    w = w1 + w2
    if w == 0.0:
        return 0.0

    # Eq. 13-14
    lambda_prev = 0.0
    d_lambda = (-theta - alpha_tilde * lambda_prev) / (w * dt + alpha_tilde / dt)
    # TODO consider lambda_prev?
    # p = d_lambda * n * relaxation

    # Eq. 15-16
    return d_lambda