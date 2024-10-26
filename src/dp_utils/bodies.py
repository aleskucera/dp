from typing import List, Union

import nvtx
import warp as wp


@wp.struct
class BodyInfo:
    idx: wp.int32


def create_body_info(body_name: str, model: wp.sim.Model) -> BodyInfo:
    body_info = BodyInfo()
    body_info.idx = model.body_name.index(body_name)
    return body_info


@wp.kernel
def update_trajectory_kernel(
    body: BodyInfo,
    body_q: wp.array(dtype=wp.transform),
    trajectory_index: wp.int32,
    trajectory: wp.array(dtype=wp.transform),
):
    trajectory[trajectory_index] = body_q[body.idx]


@wp.kernel
def get_q(body: BodyInfo, body_q: wp.array(dtype=wp.transform), q: wp.transform):
    q = body_q[body.idx]


# @wp.kernel
# def get_body_q(bodies: List[BodyInfo], body_q: wp.array(dtype=wp.float32), q: wp.array(dtype=wp.float32)):
#     tid = wp.tid()
#     body = bodies[tid]


# def get_body_q(bodies: List[BodyInfo], model: Union[wp.sim.Model, wp.sim.ModelBuilder]) -> List[wp.float32]:
#     if isinstance(model, wp.sim.Model):
#         body_array = wp.array(bodies, dtype=BodyInfo)

