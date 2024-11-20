import warp as wp
import numpy as np

from dp_utils.colors import *


class Trajectory:
    def __init__(self,
                 name: str,
                 time: np.ndarray,
                 pos: wp.array = None,
                 rot: wp.array = None,
                 requires_grad: bool = False,
                 plot_line_width: int = 2,
                 render_radius: float = 0.01,
                 render_color: Color = RED):

        assert len(time) >= 2, "Time array must have at least two elements."

        # Data properties
        self.pos = pos
        self.rot = rot
        self.name = name
        self.time = time

        # Rendering properties
        self.color = render_color
        self.render_radius = render_radius
        self.plot_line_width = plot_line_width

        if pos is None:
            self.pos = wp.empty((len(time),), dtype=wp.vec3, requires_grad=requires_grad)
            self.rot = wp.empty((len(time),), dtype=wp.quat, requires_grad=requires_grad)
        else:
            assert pos.dtype == wp.vec3, "Data must be of type wp.vec3."
            assert len(pos) == len(time), "Data and time arrays must have the same length."
            self.pos.requires_grad = requires_grad
            self.rot.requires_grad = requires_grad

    def __len__(self):
        return len(self.time)

    @property
    def grad(self):
        if self.pos.requires_grad:
            return self.pos.grad

    @property
    def x(self):
        return self.pos.numpy()[:, 0]

    @property
    def y(self):
        return self.pos.numpy()[:, 1]

    @property
    def z(self):
        return self.pos.numpy()[:, 2]

    @property
    def clone(self):
        return Trajectory(name=self.name,
                          time=self.time,
                          pos=wp.clone(self.pos),
                          rot=wp.clone(self.rot),
                          requires_grad=self.pos.requires_grad,
                          plot_line_width=self.plot_line_width,
                          render_color=self.color,
                          render_radius=self.render_radius)

    def update_data(self, time_step: int, q: wp.array, q_idx: int):
        if q.dtype == wp.vec3:
            wp.launch(kernel=_update_position_kernel_vec3, dim=(1,),
                      inputs=[self.pos, q, time_step, q_idx])
        elif q.dtype == wp.transform:
            wp.launch(kernel=_update_position_kernel_transform, dim=(1,),
                      inputs=[self.pos, q, time_step, q_idx])
            wp.launch(kernel=_update_rotation_kernel, dim=(1,),
                      inputs=[self.rot, q, time_step, q_idx])
        else:
            raise ValueError(f"Unsupported data type: {q.dtype}")


@wp.kernel
def _update_position_kernel_vec3(trajectory: wp.array(dtype=wp.vec3),
                                 q: wp.array(dtype=wp.vec3),
                                 time_step: wp.int32,
                                 q_idx: wp.int32):
    trajectory[time_step] = q[q_idx]


@wp.kernel
def _update_position_kernel_transform(trajectory: wp.array(dtype=wp.vec3),
                                      q: wp.array(dtype=wp.transform),
                                      time_step: wp.int32,
                                      q_idx: wp.int32):
    trajectory[time_step] = wp.transform_get_translation(q[q_idx])


@wp.kernel
def _update_rotation_kernel(trajectory: wp.array(dtype=wp.quat),
                            q: wp.array(dtype=wp.transform),
                            time_step: wp.int32,
                            q_idx: wp.int32):
    trajectory[time_step] = wp.transform_get_rotation(q[q_idx])
