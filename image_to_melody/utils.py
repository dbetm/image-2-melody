from typing import Any, List, Optional, Union

import numpy as np


def map_value_to_dest(
    val_: Union[int, float],
    dest_vals: List[Any],
    max_value: Optional[Union[int, float]] = None,
    thresholds: Optional[List[Union[int, float]]] = None,
) -> Any:
    """Given a value, a destination values list, and an upper bound inclusive for that value or the list of thresholds, then perform a linear map."""

    if max_value:
        thresholds = np.linspace(start=0, stop=max_value, num=len(dest_vals))

    dest_value = dest_vals[-1]

    for i in range(0, len(thresholds)):
        if val_ <= thresholds[i]:
            dest_value = dest_vals[i]
            break

    return dest_value