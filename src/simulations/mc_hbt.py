import numpy as np
from matplotlib import pyplot as plt
from numba import float64, int64, njit, void
from numba.types import Tuple  # type: ignore
from numpy.typing import NDArray


@njit(void(int64))
def seed_env(seed: int) -> None:
    np.random.seed(seed)


@njit(float64[:](int64, float64, float64, float64))
def f_1(n: int, T_ns: float, eff: float, lifetime_ns: float) -> NDArray[np.float64]:
    dur = np.log(np.random.random(n)) * -lifetime_ns
    if eff == 1:
        return np.arange(1, n + 1) * T_ns + dur
    else:
        return (
            np.cumsum(np.floor(np.log(np.random.random(n)) / np.log(1 - eff)) + 1)
            * T_ns
            + dur
        )


@njit(float64[:](float64[:], float64[:]))
def f_2(t_1: NDArray[np.float64], t_2: NDArray[np.float64]) -> NDArray[np.float64]:
    t = np.empty(len(t_1) + len(t_2), dtype=np.float64)
    i = 0
    j = 0

    for k in range(len(t)):
        if i == len(t_1):
            t[k:] = t_2[j:]
            break

        if j == len(t_2):
            t[k:] = t_1[i:]
            break

        if t_1[i] <= t_2[j]:
            t[k] = t_1[i]
            i += 1
        else:
            t[k] = t_2[j]
            j += 1

    return t


@njit(Tuple((float64[:], float64[:]))(float64[:]))
def f_3(t: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mask = np.random.random(len(t)) <= 0.5
    return t[mask], t[~mask]


@njit(float64[:](float64[:], float64[:], float64))
def f_4(
    t_1: NDArray[np.float64], t_2: NDArray[np.float64], half_int_ns: float
) -> np.ndarray:
    starts = np.empty(len(t_1), dtype=np.int64)
    ends = np.empty(len(t_1), dtype=np.int64)

    size = 0
    ptr_1 = 0
    ptr_2 = 0
    for i in range(len(t_1)):
        while ptr_1 < len(t_2) and t_2[ptr_1] < t_1[i] - half_int_ns:
            ptr_1 += 1

        if ptr_2 < ptr_1:
            ptr_2 = ptr_1

        while ptr_2 < len(t_2) and t_2[ptr_2] < t_1[i] + half_int_ns:
            ptr_2 += 1

        starts[i] = ptr_1
        ends[i] = ptr_2
        size += ptr_2 - ptr_1

    taus = np.empty(size, dtype=np.float64)
    i = 0
    for j in range(len(t_1)):
        for k in range(starts[j], ends[j]):
            taus[i] = t_1[j] - t_2[k]
            i += 1

    return taus
