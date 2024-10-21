from typing import Any, List, Optional

import warp as wp
import numpy as np


def custom_eval(input: str):
    import os
    import sys
    import math

    import numpy as np    
    import warp as wp
    import warp.sim
    import warp.sim.render

    import dp
    import dp_utils
    return eval(input)

def get_index_by_value(
    elements: List[Any], 
    target_value: Any, 
    ignore_values: List[Any] = []
) -> Optional[int]:
    """
    Returns the index of the first occurrence of `target_value` in the `elements` list,
    while ignoring any elements specified in `ignore_values`.
    
    Args:
        elements (List[Any]): The list of elements to search through.
        target_value (Any): The value for which to find the index.
        ignore_values (List[Any], optional): A list of values to ignore during the search. 
                                             Defaults to an empty list.
    
    Returns:
        Optional[int]: The index of the first occurrence of `target_value` that is not
                       in `ignore_values`. If no match is found, returns `None`.
    
    Example:
        elements = ['a', 'b', 'a', 'c']
        get_index_by_value(elements, 'b', ignore_values=['a'])  # Returns: 0
    """
    filtered_elements = [v for v in elements if v not in ignore_values]
    return filtered_elements.index(target_value) if target_value in filtered_elements else None

@wp.kernel
def compute_sin(delta_t: float, out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    t = 10.0 * float(tid) * delta_t
    out[tid] = 5.0 * wp.sin(t)

# Taken from env/environment.py
def compute_env_offsets(num_envs, env_offset=(5.0, 0.0, 5.0), up_axis="Y"):
    
    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]

    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
    else:
        env_offsets = np.zeros((num_envs, 3))

    if num_dim == 1:
        for i in range(num_envs):
            env_offsets.append(i * env_offset)
    elif num_dim == 2:
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)
    elif num_dim == 3:
        for i in range(num_envs):
            d0 = i // (side_length * side_length)
            d1 = (i // side_length) % side_length
            d2 = i % side_length
            offset = np.zeros(3)
            offset[0] = d0 * env_offset[0]
            offset[1] = d1 * env_offset[1]
            offset[2] = d2 * env_offset[2]
            env_offsets.append(offset)
    env_offsets = np.array(env_offsets)
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    correction[up_axis] = 0.0  # ensure the envs are not shifted below the ground plane
    env_offsets -= correction
    return env_offsets