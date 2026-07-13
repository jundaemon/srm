import sqlite3

import numpy as np
from numpy.typing import NDArray

DB_NAME = "cache.db"


def create_cache() -> None:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "CREATE TABLE samples (seed INTEGER NOT NULL, input BLOB NOT NULL, label REAL NOT NULL);"
    )
    conn.commit()

    conn.close()


def cache_samples(
    seed: int, inputs: NDArray[np.float32], labels: NDArray[np.float32]
) -> None:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    values = []
    for input, label in zip(inputs, labels):
        values.append((seed, input.tobytes(), label))

    cursor.executemany(
        "INSERT INTO samples (seed, input, label) VALUES (?, ?, ?)", values
    )
    conn.commit()

    conn.close()


def hit_cache(
    seed: int, rows: int, cols: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    inputs = np.empty((rows, cols), dtype=np.float32)
    labels = np.empty(rows, dtype=np.float32)
    for i, row in enumerate(
        cursor.execute(f"SELECT input, label FROM samples WHERE seed = {seed};")
    ):
        inputs[i] = np.frombuffer(row[0], dtype=np.float32)
        labels[i] = row[1]

    conn.close()
    return inputs, labels
