from typing import Callable

import numpy as np


def _kuramoto_transform(x):
    return np.mod(x, 2 * np.pi)


_STATE_TRANSFORMS: dict[str, Callable] = {
    "kuramoto": _kuramoto_transform,
    "kuramoto_all_to_all": _kuramoto_transform,
}


def get_state_transform(name: str):
    return _STATE_TRANSFORMS.get(name)
