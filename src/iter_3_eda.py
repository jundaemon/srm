import numpy as np
from matplotlib import pyplot as plt
from numba import float64, int64, njit, prange
from numpy.typing import NDArray

from simulations.hbt import (
    BINS,
    HALF_INT_NS,
    INPUT_N,
    INPUT_N_2,
    LIFETIME_NS,
    T_NS,
    f_1,
    f_2,
    f_3,
    f_4,
    input_gen,
    seed_env,
)

EFF_1S = np.repeat(0.1, 10)
EFF_2S = np.linspace(0.1, 1.0, 10)

fig, axes = plt.subplots(4, 5, figsize=(16, 10))

SEED = 10
seed_env(SEED)
i = 0
for eff_1, eff_2 in zip(EFF_1S, EFF_2S):
    set_1 = f_1(INPUT_N, T_NS, eff_1, LIFETIME_NS)
    set_2 = f_1(INPUT_N, T_NS, eff_2, LIFETIME_NS)
    set_t = f_2(set_1, set_2)
    set_1, set_2 = f_3(set_t)
    taus = f_4(set_1, set_2, HALF_INT_NS)

    ax = axes.flat[i]
    ax.hist(taus, BINS, color="blue")
    ax.set_title(f"n = {INPUT_N}, eff 1 = {eff_1:.1f}, eff 2 = {eff_2:.1f}")
    i += 1

for eff_1, eff_2 in zip(EFF_1S, EFF_2S):
    set_1 = f_1(INPUT_N_2, T_NS, eff_1, LIFETIME_NS)
    set_2 = f_1(INPUT_N_2, T_NS, eff_2, LIFETIME_NS)
    set_t = f_2(set_1, set_2)
    set_1, set_2 = f_3(set_t)
    taus = f_4(set_1, set_2, HALF_INT_NS)

    ax = axes.flat[i]
    ax.hist(taus, BINS, color="red")
    ax.set_title(f"n = {INPUT_N_2}, eff 1 = {eff_1:.1f}, eff 2 = {eff_2:.1f}")
    i += 1

plt.tight_layout()
plt.show()


BPP = T_NS
BPHP = T_NS // 2
PEAK_I = np.arange(BPP, BINS, BPP, dtype=np.int64)
PEAK_I = PEAK_I[PEAK_I != BINS // 2]


@njit(float64(int64[:]))
def calculate_std_dev(histogram: NDArray[np.int64]) -> float:
    side_peak_areas = np.empty(len(PEAK_I), np.int64)
    for i, peak_i in enumerate(PEAK_I):
        side_peak_areas[i] = histogram[peak_i - BPHP : peak_i + BPHP].sum()

    return float(np.std(side_peak_areas))


# the average standard deviation of side peak areas in histograms is calculated here
# instead of the standard deviation of average side peak area in histograms because
# the concern here is the number of signals within histograms for the model to train on
# instead of variance within the entire dataset
@njit(float64(int64[:, :]), parallel=True)
def calculate_avg_std_dev(histograms: NDArray[np.int64]) -> float:
    std_devs = np.empty(len(histograms), np.float64)
    for i in prange(len(histograms)):  # type: ignore
        std_devs[i] = calculate_std_dev(histograms[i])

    return std_devs.mean()


old_histograms = input_gen(INPUT_N, SEED)
new_histograms = input_gen(INPUT_N_2, SEED)
print(calculate_avg_std_dev(old_histograms))
print(calculate_avg_std_dev(new_histograms))
# average standard deviation of side peak area increased from ~3 to ~13
# by increasing n from 50 to 1000
