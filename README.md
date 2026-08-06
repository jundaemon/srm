## Introduction
This repository serves as a preliminary to my final year project, where I try to predict the second order correlation at time delay of 0s, $g^{(2)}(0)$ from sparse histograms of $\tau$. Instead of histograms generated from a simulated continuous laser, this repository uses a Monte Carlo simulation of a nitrogen vacancy center pulsed with a laser to generate histograms of $\tau$.

Updates on my progress can be tracked through this [tldraw whiteboard](https://www.tldraw.com/p/u_fYcaZ1v9dJCUSF2kiNZ?d=v-142.-70.3320.1866.page).

## Problem
In the context of simulated pulses to a nitrogen vacancy center, the $g^{(2)}(0)$ can only be reliably calculated when there is a significant number of detection events at both detectors, then the $g^{(2)}(0)$ can be calculated by dividing the area of peak at $\tau$ = 0s by the average area of side peaks. In real life, you can imagine the process of collecting all these detection events would take a very long time, this project thus aims to predict the $g^{(2)}(0)$ from histograms of $\tau$ when there only a few detection events.

## Process
1) Generate histograms of $\tau$ using a million detection events total between 2 detectors, for every possible pair of efficiencies over 121 seeds.
2) Use these histograms to calculate $g^{(2)}(0)$, which will be the labels.
3) Using the same set of efficiencies and seeds, generate histograms of $\tau$ using only 100 (or more) detection events total between 2 detectors, these will be the model inputs.
4) Train different ML/DL models to predict the $g^{(2)}(0)$ from sparse histograms of $\tau$, pick the one that's most accurate.
