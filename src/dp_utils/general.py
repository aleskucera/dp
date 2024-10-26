from typing import Any, List, Optional, Union

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
    elements: List[Any], target_value: Any, ignore_values: List[Any] = []
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
    return (
        filtered_elements.index(target_value)
        if target_value in filtered_elements
        else None
    )

