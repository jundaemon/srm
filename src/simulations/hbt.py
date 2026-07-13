import numpy as np
from numba import float64, int64, njit, prange, void
from numba.types import Tuple  # type: ignore
from numpy.typing import NDArray


@njit(void(int64))
def seed_env(seed: int) -> None:
    np.random.seed(seed)


@njit(int64[:](int64))
def seed_gen(size: int) -> NDArray[np.int64]:
    seeds = np.empty(size, dtype=np.int64)

    i = 0
    while i < size:
        candidate = np.random.randint(10, 9_999)
        if candidate not in seeds:
            seeds[i] = candidate
            i += 1

    return seeds


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
) -> NDArray[np.float64]:
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


@njit(Tuple((int64[:], float64))(float64[:], int64, float64))
def f_5(
    taus: NDArray[np.float64], bins: int, T_ns: float
) -> tuple[NDArray[np.int64], float]:
    hist, edges = np.histogram(taus, bins=bins)
    return hist, np.floor(T_ns / (edges[1] - edges[0]))


@njit(Tuple((int64[:], int64))(int64, float64))
def f_6(bins: int, bpp: float) -> tuple[NDArray[np.int64], int]:
    peak_i = np.arange(bpp, bins, bpp, dtype=np.int64)
    peak_i = peak_i[peak_i != bins // 2]

    return peak_i, bins // 2


@njit(float64(int64[:], float64, int64[:], int64))
def f_7(
    hist: NDArray[np.int64], bpp: float, peak_i: NDArray[np.int64], tau_zero_i: int
) -> float:
    areas = np.empty(len(peak_i), dtype=np.float64)
    for i in range(len(peak_i)):
        areas[i] = hist[peak_i[i] - bpp // 2 : peak_i[i] + bpp // 2].sum()

    return hist[tau_zero_i - bpp // 2 : tau_zero_i + bpp // 2].sum() / areas.mean()


EFF_1S = np.repeat(np.linspace(0.1, 1.0, 91), 91)
EFF_2S = EFF_1S.reshape(91, 91).T.ravel()

LABELS_N = 500_000
T_NS = 50
LIFETIME_NS = 3
HALF_INT_NS = 250
BINS = 500


@njit(float64[:](int64), parallel=True)
def label_gen(seed: int) -> NDArray[np.float64]:
    g2_zeros = np.empty(len(EFF_1S), dtype=np.float64)
    seed_env(seed)

    for i in prange(len(EFF_1S)):  # type: ignore
        t_1 = f_1(LABELS_N, T_NS, EFF_1S[i], LIFETIME_NS)
        t_2 = f_1(LABELS_N, T_NS, EFF_2S[i], LIFETIME_NS)
        t = f_2(t_1, t_2)
        t_1, t_2 = f_3(t)
        taus = f_4(t_1, t_2, HALF_INT_NS)
        hist, bpp = f_5(taus, BINS, T_NS)
        peak_i, tau_zero_i = f_6(BINS, bpp)
        g2_zeros[i] = f_7(hist, bpp, peak_i, tau_zero_i)

    return g2_zeros


INPUT_N = 50


@njit(int64[:, :](int64), parallel=True)
def input_gen(seed: int) -> NDArray[np.int64]:
    histograms = np.empty((len(EFF_1S), BINS), dtype=np.int64)
    seed_env(seed)

    for i in prange(len(EFF_1S)):  # type: ignore
        t_1 = f_1(INPUT_N, T_NS, EFF_1S[i], LIFETIME_NS)
        t_2 = f_1(INPUT_N, T_NS, EFF_2S[i], LIFETIME_NS)
        t = f_2(t_1, t_2)
        t_1, t_2 = f_3(t)
        taus = f_4(t_1, t_2, HALF_INT_NS)
        hist, _ = f_5(taus, BINS, T_NS)
        histograms[i,] = hist

    return histograms
