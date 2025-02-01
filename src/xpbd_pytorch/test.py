import torch

from xpbd_pytorch.body import *
from xpbd_pytorch.quat import *
from xpbd_pytorch.utils import *
from xpbd_pytorch.cuboid import *
from xpbd_pytorch.constants import *
from xpbd_pytorch.animation import *
from xpbd_pytorch.correction import *



def positional_correction():
    n_steps = 20
    time_len = 3.0
    time = torch.linspace(0, time_len, n_steps)

    box_a = Cuboid(x=torch.tensor([0.0, 0.0, 0.0]),
                   q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                   v=torch.tensor([0.0, 0.0, 0.0]),
                   w=torch.tensor([0.0, 0.0, 0.0]),
                   m=1.0,
                   I=torch.eye(3),
                   restitution=0.5,
                   static_friction=0.2,
                   dynamic_friction=0.4,
                   n_collision_points=36)
    r_a = torch.tensor([0.0, 0.0, 0.0])

    box_b = Cuboid(x=torch.tensor([0.0, 0.0, 2.0]),
                   q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                   v=torch.tensor([0.0, 0.0, 0.0]),
                   w=torch.tensor([0.0, 0.0, 0.0]),
                   m=1.0,
                   I=torch.eye(3),
                   restitution=0.5,
                   static_friction=0.2,
                   dynamic_friction=0.4,
                   n_collision_points=36)
    r_b = torch.tensor([0.0, 0.0, 0.0])

    m_a_inv, m_b_inv = box_a.m_inv, box_b.m_inv
    I_a_inv, I_b_inv = box_a.I_inv, box_b.I_inv

    box_a.trajectory = Trajectory(time)
    box_b.trajectory = Trajectory(time)

    box_a.save_state(0)
    box_b.save_state(0)

    for i in range(1, n_steps):
        x_a, x_b = box_a.x, box_b.x
        q_a, q_b = box_a.q, box_b.q

        dx_a, dx_b, dq_a, dq_b, d_lambda = positional_delta(x_a=x_a,
                                                            x_b=x_b,
                                                            q_a=q_a,
                                                            q_b=q_b,
                                                            r_a=r_a,
                                                            r_b=r_b,
                                                            m_a_inv=m_a_inv,
                                                            m_b_inv=m_b_inv,
                                                            I_a_inv=I_a_inv,
                                                            I_b_inv=I_b_inv)

        box_a.x += dx_a
        box_b.x += dx_b
        box_a.q += dq_a
        box_b.q += dq_b
        box_a.q /= torch.norm(box_a.q)
        box_b.q /= torch.norm(box_b.q)

        box_a.save_state(i)
        box_b.save_state(i)

    controller = AnimationController(bodies=[box_a, box_b],
                                     time=time,
                                     x_lims=(-3, 3),
                                     y_lims=(-3, 3),
                                     z_lims=(-1, 5))
    controller.start()

def positional_correction_2():
    n_steps = 20
    time_len = 3.0
    time = torch.linspace(0, time_len, n_steps)

    box_a = Cuboid(x=torch.tensor([0.0, 0.0, 0.0]),
                   q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                   v=torch.tensor([0.0, 0.0, 0.0]),
                   w=torch.tensor([0.0, 0.0, 0.0]),
                   m=1.0,
                   I=torch.eye(3),
                   restitution=0.5,
                   static_friction=0.2,
                   dynamic_friction=0.4,
                   n_collision_points=4)
    r_a = torch.tensor([1.0, 1.0, 0.0])

    box_b = Cuboid(x=torch.tensor([0.0, 0.0, 2.0]),
                   q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                   v=torch.tensor([0.0, 0.0, 0.0]),
                   w=torch.tensor([0.0, 0.0, 0.0]),
                   m=1.0,
                   I=torch.eye(3),
                   restitution=0.5,
                   static_friction=0.2,
                   dynamic_friction=0.4,
                   n_collision_points=4)
    r_b = torch.tensor([-1.0, -1.0, 0.0])

    m_a_inv, m_b_inv = box_a.m_inv, box_b.m_inv
    I_a_inv, I_b_inv = box_a.I_inv, box_b.I_inv

    box_a.trajectory = Trajectory(time)
    box_b.trajectory = Trajectory(time)

    box_a.vector_history[0] = []
    box_a.vector_history[0].append(LocalVector(r_a, box_a.x.clone(), box_a.q.clone()))

    box_b.vector_history[0] = []
    box_b.vector_history[0].append(LocalVector(r_b, box_b.x.clone(), box_b.q.clone()))

    box_a.save_state(0)
    box_b.save_state(0)

    for i in range(1, n_steps):
        x_a, x_b = box_a.x, box_b.x
        q_a, q_b = box_a.q, box_b.q

        dx_a, dx_b, dq_a, dq_b, d_lambda = positional_delta(x_a=x_a,
                                                            x_b=x_b,
                                                            q_a=q_a,
                                                            q_b=q_b,
                                                            r_a=r_a,
                                                            r_b=r_b,
                                                            m_a_inv=m_a_inv,
                                                            m_b_inv=m_b_inv,
                                                            I_a_inv=I_a_inv,
                                                            I_b_inv=I_b_inv)

        box_a.x += dx_a
        box_b.x += dx_b
        box_a.q += dq_a
        box_b.q += dq_b
        box_a.q /= torch.norm(box_a.q)
        box_b.q /= torch.norm(box_b.q)

        box_a.vector_history[i] = []
        box_a.vector_history[i].append(LocalVector(r_a, box_a.x.clone(), box_a.q.clone()))

        box_b.vector_history[i] = []
        box_b.vector_history[i].append(LocalVector(r_b, box_b.x.clone(), box_b.q.clone()))

        box_a.save_state(i)
        box_b.save_state(i)

    controller = AnimationController(bodies=[box_a, box_b],
                                     time=time,
                                     x_lims=(-3, 3),
                                     y_lims=(-3, 3),
                                     z_lims=(-1, 5))
    controller.start()


if __name__ == "__main__":
    # positional_correction()
    # positional_correction_2()
    demo_angular_correction()