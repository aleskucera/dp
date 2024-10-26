from typing import Union, List, Tuple

import nvtx
import warp.sim
import warp as wp
import numpy as np


@wp.struct
class JointInfo:
    joint_idx: wp.int32
    q_idx: wp.int32
    qd_idx: wp.int32
    axis_idx: wp.int32


def create_joint_info(
    joint_name: str, model: Union[wp.sim.Model, wp.sim.ModelBuilder]
) -> JointInfo:
    if isinstance(model, wp.sim.Model):
        joint_q_start = model.joint_q_start.to("cpu").numpy()
        joint_qd_start = model.joint_qd_start.to("cpu").numpy()
        joint_axis_start = model.joint_axis_start.to("cpu").numpy()
    elif isinstance(model, wp.sim.ModelBuilder):
        joint_q_start = model.joint_q_start
        joint_qd_start = model.joint_qd_start
        joint_axis_start = model.joint_axis_start
    else:
        raise ValueError(
            "Model must be of type warp.sim.Model or warp.sim.ModelBuilder."
        )

    joint_info = JointInfo()
    joint_info.joint_idx = model.joint_name.index(joint_name)
    joint_info.q_idx = joint_q_start[joint_info.joint_idx]
    joint_info.qd_idx = joint_qd_start[joint_info.joint_idx]
    joint_info.axis_idx = joint_axis_start[joint_info.joint_idx]
    return joint_info


def set_joint_q(
    joints: List[JointInfo],
    values: List[float],
    model: Union[wp.sim.Model, wp.sim.ModelBuilder],
):
    if isinstance(model, wp.sim.Model):
        joint_array = wp.array(joints, dtype=JointInfo)
        value_array = wp.array(values, dtype=wp.float32)
        wp.launch(
            kernel=set_joint_q_kernel,
            dim=len(joints),
            inputs=[joint_array, value_array, model.joint_q],
            device=model.device,
        )
    elif isinstance(model, wp.sim.ModelBuilder):
        for joint, value in zip(joints, values):
            model.joint_q[joint.q_idx] = value
    else:
        raise ValueError(
            "Model must be of type warp.sim.Model or warp.sim.ModelBuilder."
        )


@wp.kernel
def set_joint_q_kernel(
    joints: wp.array(dtype=JointInfo),
    values: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    joint = joints[tid]
    joint_q[joint.axis_idx] = values[tid]


def set_joint_qd(
    joints: List[JointInfo],
    values: List[float],
    model: Union[wp.sim.Model, wp.sim.ModelBuilder],
):
    if isinstance(model, wp.sim.Model):
        joint_array = wp.array(joints, dtype=JointInfo)
        value_array = wp.array(values, dtype=wp.float32)
        wp.launch(
            kernel=set_joint_qd_kernel,
            dim=len(joints),
            inputs=[joint_array, value_array, model.joint_qd],
            device=model.device,
        )
    elif isinstance(model, wp.sim.ModelBuilder):
        for joint, value in zip(joints, values):
            model.joint_qd[joint.qd_idx] = value
    else:
        raise ValueError(
            "Model must be of type warp.sim.Model or warp.sim.ModelBuilder."
        )


@wp.kernel
def set_joint_qd_kernel(
    joints: wp.array(dtype=JointInfo),
    values: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    joint = joints[tid]
    joint_qd[joint.qd_idx] = values[tid]


def set_joint_axis_mode(
    joints: List[JointInfo],
    values: List[int],
    model: Union[wp.sim.Model, wp.sim.ModelBuilder],
):
    if isinstance(model, wp.sim.Model):
        joint_array = wp.array(joints, dtype=JointInfo)
        value_array = wp.array(values, dtype=wp.int32)
        wp.launch(
            kernel=set_joint_axis_mode_kernel,
            dim=len(joints),
            inputs=[joint_array, value_array, model.joint_axis_mode],
            device=model.device,
        )
    elif isinstance(model, wp.sim.ModelBuilder):
        for joint, value in zip(joints, values):
            model.joint_axis_mode[joint.axis_idx] = value
    else:
        raise ValueError(
            "Model must be of type warp.sim.Model or warp.sim.ModelBuilder."
        )


@wp.kernel
def set_joint_axis_mode_kernel(
    joints: wp.array(dtype=JointInfo),
    values: wp.array(dtype=wp.int32),
    joint_axis_mode: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    joint = joints[tid]
    joint_axis_mode[joint.axis_idx] = values[tid]


def set_linear_compliance(
    joints: List[JointInfo],
    values: List[float],
    model: Union[wp.sim.Model, wp.sim.ModelBuilder],
):
    if isinstance(model, wp.sim.Model):
        joint_array = wp.array(joints, dtype=JointInfo)
        value_array = wp.array(values, dtype=wp.float32)
        wp.launch(
            kernel=set_linear_compliance_kernel,
            dim=len(joints),
            inputs=[joint_array, value_array, model.joint_linear_compliance],
            device=model.device,
        )
    elif isinstance(model, wp.sim.ModelBuilder):
        for joint, value in zip(joints, values):
            model.joint_linear_compliance[joint.joint_idx] = value
    else:
        raise ValueError(
            "Model must be of type warp.sim.Model or warp.sim.ModelBuilder."
        )


@wp.kernel
def set_linear_compliance_kernel(
    joints: wp.array(dtype=JointInfo),
    values: wp.array(dtype=wp.float32),
    joint_linear_compliance: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    joint = joints[tid]
    joint_linear_compliance[joint.joint_idx] = values[tid]


def set_angular_compliance(
    joints: List[JointInfo],
    values: List[float],
    model: Union[wp.sim.Model, wp.sim.ModelBuilder],
):
    if isinstance(model, wp.sim.Model):
        joint_array = wp.array(joints, dtype=JointInfo)
        value_array = wp.array(values, dtype=wp.float32)
        wp.launch(
            kernel=set_angular_compliance_kernel,
            dim=len(joints),
            inputs=[joint_array, value_array, model.joint_angular_compliance],
            device=model.device,
        )
    elif isinstance(model, wp.sim.ModelBuilder):
        for joint, value in zip(joints, values):
            model.joint_angular_compliance[joint.joint_idx] = value
    else:
        raise ValueError(
            "Model must be of type warp.sim.Model or warp.sim.ModelBuilder."
        )


@wp.kernel
def set_angular_compliance_kernel(
    joints: wp.array(dtype=JointInfo),
    values: wp.array(dtype=wp.float32),
    joint_angular_compliance: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    joint = joints[tid]
    joint_angular_compliance[joint.joint_idx] = values[tid]


def set_joint_target_ke(
    joints: List[JointInfo],
    values: List[float],
    model: Union[wp.sim.Model, wp.sim.ModelBuilder, wp.sim.State],
):
    if isinstance(model, wp.sim.Model) or isinstance(model, wp.sim.State):
        joint_array = wp.array(joints, dtype=JointInfo)
        value_array = wp.array(values, dtype=wp.float32)
        wp.launch(
            kernel=set_joint_target_ke_kernel,
            dim=len(joints),
            inputs=[joint_array, value_array, model.joint_target_ke],
            device=model.device,
        )
    elif isinstance(model, wp.sim.ModelBuilder):
        for joint, value in zip(joints, values):
            model.joint_target_ke[joint.axis_idx] = value
    else:
        raise ValueError(
            "Model must be of type warp.sim.Model or warp.sim.ModelBuilder."
        )


@wp.kernel
def set_joint_target_ke_kernel(
    joints: wp.array(dtype=JointInfo),
    values: wp.array(dtype=wp.float32),
    joint_target_ke: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    joint = joints[tid]
    joint_target_ke[joint.axis_idx] = values[tid]


def set_joint_target_kd(
    joints: List[JointInfo],
    values: List[float],
    model: Union[wp.sim.Model, wp.sim.ModelBuilder],
):
    if isinstance(model, wp.sim.Model):
        joint_array = wp.array(joints, dtype=JointInfo)
        value_array = wp.array(values, dtype=wp.float32)
        wp.launch(
            kernel=set_joint_target_kd_kernel,
            dim=len(joints),
            inputs=[joint_array, value_array, model.joint_target_kd],
            device=model.device,
        )
    elif isinstance(model, wp.sim.ModelBuilder):
        for joint, value in zip(joints, values):
            model.joint_target_kd[joint.axis_idx] = value
    else:
        raise ValueError(
            "Model must be of type warp.sim.Model or warp.sim.ModelBuilder."
        )


@wp.kernel
def set_joint_target_kd_kernel(
    joints: wp.array(dtype=JointInfo),
    values: wp.array(dtype=wp.float32),
    joint_target_kd: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    joint = joints[tid]
    joint_target_kd[joint.axis_idx] = values[tid]


def get_joint_act(
    joints: List[JointInfo],
    values: List[float],
    model: Union[wp.sim.Model, wp.sim.ModelBuilder],
) -> List[float]:
    joint_act = np.zeros(model.joint_axis_count, dtype=np.float32)
    for joint, value in zip(joints, values):
        joint_act[joint.axis_idx] = value
    joint_act = wp.from_numpy(joint_act, device=model.device)
    return joint_act


def set_joint_act(
    joints: List[JointInfo],
    values: List[float],
    entity: Union[wp.sim.Model, wp.sim.ModelBuilder, wp.sim.Control],
):
    if isinstance(entity, wp.sim.Model) or isinstance(entity, wp.sim.Control):
        device = wp.get_device()
        joints_arr = wp.array(joints, dtype=JointInfo, device=device)
        values_arr = wp.array(values, dtype=wp.float32, device=device)
        if isinstance(entity, wp.sim.Control) and entity.joint_act is None:
            entity.joint_act = wp.zeros(entity.model.joint_axis_count, dtype=wp.float32)
        wp.launch(
            kernel=set_joint_act_kernel,
            dim=len(joints),
            inputs=[joints_arr, values_arr],
            outputs=[entity.joint_act],
            device=device,
        )
    elif isinstance(entity, wp.sim.ModelBuilder):
        for joint, value in zip(joints, values):
            entity.joint_act[joint.axis_idx] = value
    else:
        raise ValueError(
            "Model must be of type warp.sim.Model or warp.sim.ModelBuilder."
        )


@wp.kernel
@nvtx.annotate()
def set_joint_act_kernel(
    joints: wp.array(dtype=JointInfo),
    values: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    joint = joints[tid]
    joint_act[joint.axis_idx] = values[tid]
