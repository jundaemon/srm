## Introduction
This repository serves as a preliminary to my final year project, where I try to predict the second order correlation at time delay of 0s, $g^{2}(0)$ from sparse histograms of $\tau$.

## Problem statement
The $g^{2}(0)$ value can only be reliably calculated when there is a large number of detection events at both detectors, in which then $g^{2}(0)$ = area of peak at $\tau$ = 0s / average area of side peaks (in the context of a pulsed laser, which this simulation is using).

However, collecting a large number of detection events is time consuming. This project thus aims to train a regression model to predict the $g^{2}(0)$ from sparse histograms of $\tau$ generated using only a few detection events, providing massive time saves for each $g^{2}(0)$ realization.

## Synthetic data generation
Code for data generation can be found in this [subdirectory](https://github.com/jundaemon/srm/blob/main/src/simulations/hbt.py).

### f_1
This first function creates an array of photon arrival times from a single emitter. `n` is the number of arrival times to match and `T_ns` is the period of laser pulses.

When a nitrogen vacancy center is pulsed with a laser, a photon isn't guaranteed to be emitted. `eff` is the probability a center emits a photon when pulsed with a laser. When `eff` is 1, `np.arange` is used to instantiate the photon numbers, these photon numbers are then multiplied by `T_ns` to get the time at which the center is pulsed with a laser. When `eff` is less than 1, then the cumulative distribution function of the [geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution) derives how many pulses are needed for each photon to be emitted.

Photons are not immediately emitted when a center is pulsed with a laser, the time taken depends on `lifetime_ns`, the lifetime of an electron. The duration for each photon to be emitted is derived from the cumulative distribution function of the [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution), then added to the time at which the center was pulsed with a laser to get the arrival time for each photon.

### f_2
This function sorts and merges the 2 passed in arrays by ascending arrival times.

### f_3
This function simulates a 50/50 beam splitter. After photons are emitted, they have to pass through a beam splitter in which they then end up at either detector.

### f_4
The fourth function calculates all $\tau$ between a detection event at the first detector and all other detection events at the second detector within a window, in a sliding window fashion. `t_1` represents the arrival times at detector 1, `t_2` represents the arrival times at detector 2 and `half_int_ns` represents half the interval (window) in nanoseconds.

### f_5
The fifth function returns the histogram after binning $\tau$ as well as the number of bins per pulse, calculated using the period of each pulse, `T_ns`.

### f_6
This function returns an array containing the indices of the centre of side peaks as well as the index of $\tau$ = 0s in the histogram. The first and last indices of the side peaks are omitted because they aren't fully formed for the `half_int_ns` that I am using. The index of $\tau$ = 0s is always at the centre of the histogram.

### f_7
This function calculates the $g^{2}(0)$ using `hist`, the histogram of $\tau$, `bpp`, the number of bins per pulse, `peak_i` the indices of centre of side peaks and `tau_zero_i`, the index of $\tau$ = 0s. $g^{2}(0)$ is calculated by dividing the area of peak at $\tau$ = 0s by the average area of side peaks.

### Putting everything together
All helper functions were eventually used in `label_gen` and `input_gen`. `label_gen` generates 1_000_000 total arrival times, sorts, merges then splits the arrival times, calculates the $\tau$, bins the $\tau$ and calculates the $g^{2}(0)$. It does this repeatedly over 8_281 permutations of efficiency pairs in parallel for each seed over 121 seeds to obtain the labels. `input_gen` does the same thing but only generates 100 total arrival times for each efficiency pair and stops at `f_5` to get the sparse histogram as training input.

The training data will likely change depending on if it is too noisy to train on, if so, then the entropy of training data will be reduced by increasing `INPUT_N` in `input_gen`.

## Training
Model architectures and training loops can be found in the `src` directory, in files prefixed with `iter_`.

## Progress
The progress of this project can be tracked through this [tldraw whiteboard](https://www.tldraw.com/p/u_fYcaZ1v9dJCUSF2kiNZ?d=v-142.-598.3320.1866.page), where I present model performance and write down thoughts on next actions.
