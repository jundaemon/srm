import os

import numpy as np

from simulations.hbt import input_gen, label_gen, seed_env, seed_gen
from utils.cache_utils import DB_NAME, cache_samples, create_cache

seed_env(10)
SEEDS = seed_gen(121)

if not os.path.isfile(DB_NAME):
    create_cache()

    for seed in SEEDS:
        print(f"generating data for seed: {seed}")
        inputs = input_gen(seed).astype(np.float32)
        labels = label_gen(seed).astype(np.float32)

        print(f"caching samples from seed: {seed}")
        cache_samples(seed, inputs, labels)
