import torch
import matplotlib.pyplot as plt
from torchgen.gen import static_dispatch_extra_headers

from xpbd_pytorch.quat import *
from xpbd_pytorch.correction import *
from xpbd_pytorch.utils import *


class Trajectory:
    def __init__(self, time: torch.Tensor):
        self.time = time
        self.x = torch.zeros((len(time), 3))
        self.q = torch.zeros((len(time), 4))
        self.v = torch.zeros((len(time), 3))
        self.w = torch.zeros((len(time), 3))

    def __len__(self):
        return len(self.time)

    def add_state(self, x: torch.Tensor, q: torch.Tensor, v: torch.Tensor, w: torch.Tensor, step: int):
        self.x[step] = x
        self.q[step] = q
        self.v[step] = v
        self.w[step] = w

    def get_state(self, step: int):
        return self.x[step], self.q[step], self.v[step], self.w[step]


class Body:
    def __init__(self,
                 dt: float,
                 sim_time: torch.Tensor,
                 x: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 q: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0]),
                 v: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 w: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 m: float = 1.0,
                 I: torch.Tensor = None,
                 restitution: float = 0.5,
                 static_friction: float = 0.2,
                 dynamic_friction: float = 0.4,
                 render_color: tuple = (0.0, 1.0, 0.0),
                 ):
        self.m = m
        self.m_inv = 1 / m

        if I is None:
            I = self._default_moment_of_inertia()
        self.I = I
        self.I_inv = torch.inverse(I)

        self.x = x
        self.q = q
        self.v = v
        self.w = w

        self.x_prev = x
        self.q_prev = q
        self.v_prev = v
        self.w_prev = w

        self.dt = dt
        self.lambda_n = torch.tensor([0.0])
        self.lambda_t = torch.tensor([0.0])

        self.restitution = restitution
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction

        self.color = render_color

        self.coll_vertices = self._create_collision_vertices()

        self.num_collisions = 0
        self.collisions = torch.full((len(self.coll_vertices), 3), float('nan'))
        self.coll_normals = torch.full((len(self.coll_vertices), 3), float('nan'))

        self.trajectory = Trajectory(sim_time)
        self.collision_history = torch.full((len(sim_time), len(self.coll_vertices), 3), float('nan'))

        self.custom_vectors = torch.full((len(sim_time), len(self.coll_vertices), 3), float('nan'))
        self.custom_vectors_x = torch.full((len(sim_time), len(self.coll_vertices), 3), float('nan'))
        self.custom_vectors_q = torch.full((len(sim_time), len(self.coll_vertices), 4), float('nan'))

    def _default_moment_of_inertia(self):
        raise NotImplementedError

    def _create_collision_vertices(self):
        raise NotImplementedError

    def plot_geometry(self, ax: plt.Axes, x: torch.Tensor, q: torch.Tensor):
        raise NotImplementedError

    def save_state(self, step: int):
        self.trajectory.add_state(self.x, self.q, self.v, self.w, step)

    def save_collisions(self, step: int):
        self.collision_history[step] = self.collisions

        self.custom_vectors[step] = self.collisions
        self.custom_vectors_x[step] = self.x
        self.custom_vectors_q[step] = self.q

    def plot(self, ax: plt.Axes, frame: int):
        x, q, _, _ = self.trajectory.get_state(frame)
        collisions = self.collision_history[frame]

        v = self.custom_vectors[frame]
        v_x = self.custom_vectors_x[frame]
        v_q = self.custom_vectors_q[frame]

        plot_axis(ax, x, q)
        self.plot_geometry(ax, x, q)
        plot_collisions(ax, x, q, collisions, color='r')
        # plot_vectors(ax, x, q, v, color='b')
        #
        # if self.num_collisions != 0:
        #     print("Collision detected at frame: ", frame)
        #     return

    def detect_collisions(self):
        self.collisions = torch.full((len(self.coll_vertices), 3), float('nan'))
        self.coll_normals = torch.full((len(self.coll_vertices), 3), float('nan'))

        ground_normal = torch.tensor([0., 0., -1.])

        # Rotate vertices to world coordinates
        world_vertices = rotate_vector(self.coll_vertices, self.q) + self.x

        # Check which vertices are below ground
        below_ground = world_vertices[:, 2] < 0

        # Store the number of collisions
        self.num_collisions = int(below_ground.sum())

        # Store collision points in local coordinates
        self.collisions[below_ground] = self.coll_vertices[below_ground]

        # Store ground normal for all collision points
        self.coll_normals[below_ground] = ground_normal

    def integrate(self,
                  lin_force: torch.Tensor = torch.zeros(3),
                  torque: torch.Tensor = torch.zeros(3),
                  gravity: torch.Tensor = torch.tensor([0.0, 0.0, -9.81])):
        # Rotate to body frame
        t = torque - torch.linalg.cross(self.w, torch.matmul(self.I, self.w))  # coriolis forces

        # Linear integration
        # add_v = torch.tensor([0.0, 0.0, 0.0])
        # if self.num_collisions > 0:
        #     add_v = torch.tensor([3.0, 0.0, 0.0])

        v_next = self.v + (lin_force + gravity) * self.m_inv * self.dt
        x_next = self.x + v_next * self.dt

        # Angular integration
        w_next = self.w + torch.matmul(self.I_inv, t) * self.dt

        # Quaternion integration using first-order approximation
        w_quat = torch.cat([torch.tensor([0.0]), w_next], dim=0)
        q_next = self.q + 0.5 * quaternion_multiply(self.q, w_quat) * self.dt
        q_next = q_next / torch.norm(q_next)  # normalize quaternion

        # Save previous state
        self.x_prev = self.x
        self.q_prev = self.q
        self.v_prev = self.v
        self.w_prev = self.w

        # Update state
        self.x = x_next
        self.q = q_next
        self.v = v_next
        self.w = w_next

    def correct_collisions(self):
        delta_x_collision = torch.zeros(3)
        delta_q_collision = torch.zeros(4)

        delta_x_friction = torch.zeros(3)
        delta_q_friction = torch.zeros(4)
        num_corrections = 0

        for r, n in zip(self.collisions, self.coll_normals):
            if torch.isnan(r).all():
                continue

            # Get the depth of the collision (correction magnitude)
            p_a = rotate_vector(r, self.q) + self.x
            p_b = torch.zeros(3)
            p_a_prev = rotate_vector(r, self.q_prev) + self.x_prev
            p_b_prev = torch.zeros(3)

            # Depth of the collision
            d = (p_a - p_b) * n

            # Compute the relative motion of the contact points and their tangential velocity
            dp = (p_a - p_a_prev) - (p_b - p_b_prev)
            dp_t = dp - torch.dot(dp, n) * n

            # Resolve collisions
            dx, dq, d_lambda = position_delta(self.q, r, self.m_inv, self.I_inv, d * n)
            delta_x_collision += dx
            delta_q_collision += dq
            self.lambda_n += d_lambda

            # Resolve friction
            dx, dq, d_lambda = position_delta(self.q, r, self.m_inv, self.I_inv, dp_t)
            delta_x_friction += dx
            delta_q_friction += dq
            self.lambda_t += d_lambda

            num_corrections += 1

        if num_corrections == 0:
            return

        delta_x = delta_x_collision
        delta_q = delta_q_collision
        if torch.abs(self.lambda_t) < torch.abs(self.lambda_n) * self.static_friction:
            delta_x += delta_x_friction
            delta_q += delta_q_friction

        self.x += delta_x / num_corrections
        self.q += delta_q / num_corrections

        self.q = self.q / torch.norm(self.q)  # normalize quaternion

    def update_velocity(self):
        self.v = (self.x - self.x_prev) / self.dt
        self.w = numerical_angular_velocity(self.q, self.q_prev, self.dt)

    def solve_velocity(self):
        delta_v_restitution = torch.zeros(3)
        delta_w_restitution = torch.zeros(3)
        delta_v_friction = torch.zeros(3)
        delta_w_friction = torch.zeros(3)
        num_corrections = 0

        for r, n in zip(self.collisions, self.coll_normals):
            if torch.isnan(r).all():
                continue

            dv, dw = restitution_delta(self.q, self.v, self.w, self.v_prev, self.w_prev, r, n, self.m_inv, self.I_inv,
                                       self.restitution)
            delta_v_restitution += dv
            delta_w_restitution += dw

            dv, dw = dynamic_friction_delta(self.q, self.v, self.w, r, n, self.m_inv, self.I_inv, self.dt,
                                            self.dynamic_friction, self.lambda_n)
            delta_v_friction += dv
            delta_w_friction += dw

            num_corrections += 1

        if num_corrections == 0:
            return

        self.v += (delta_v_restitution + delta_v_friction) / num_corrections
        self.w += (delta_w_restitution + delta_w_friction) / num_corrections

        self.lambda_n = torch.tensor([0.0])
        self.lambda_t = torch.tensor([0.0])