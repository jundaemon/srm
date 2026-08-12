import sqlite3

import numpy as np
from numpy.typing import NDArray


def create_cache(name: str) -> None:
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    cursor.execute("""CREATE TABLE samples (
    seed INTEGER NOT NULL,
    eff_1 REAL NOT NULL,
    eff_2 REAL NOT NULL,
    input BLOB NOT NULL,
    label REAL NOT NULL,
    PRIMARY KEY (seed, eff_1, eff_2)
);""")
    conn.commit()

    conn.close()


def cache_samples(
    name: str,
    seeds: NDArray[np.int64],
    eff_1s: NDArray[np.float64],
    eff_2s: NDArray[np.float64],
    X: NDArray[np.float32],
    y: NDArray[np.float32],
) -> None:
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    values = []
    for seed, eff_1, eff_2, input, label in zip(seeds, eff_1s, eff_2s, X, y):
        values.append((seed, eff_1, eff_2, input.tobytes(), label))

    cursor.executemany(
        "INSERT INTO samples (seed, eff_1, eff_2, input, label) VALUES (?, ?, ?, ?, ?)",
        values,
    )
    conn.commit()

    conn.close()


def hit_cache(
    name: str, rows: int, cols: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    inputs = np.empty((rows, cols), dtype=np.float32)
    labels = np.empty(rows, dtype=np.float32)
    for i, row in enumerate(cursor.execute("SELECT input, label FROM samples;")):
        inputs[i] = np.frombuffer(row[0], dtype=np.float32)
        labels[i] = np.frombuffer(row[1], dtype=np.float32)[0]

    conn.close()
    return inputs, labels
