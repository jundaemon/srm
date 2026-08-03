import sqlite3

import numpy as np
from numpy.typing import NDArray


def create_cache(name: str) -> None:
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE samples (input BLOB NOT NULL, label REAL NOT NULL);")
    conn.commit()

    conn.close()


def cache_samples(name: str, X: NDArray[np.float32], y: NDArray[np.float32]) -> None:
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    values = []
    for input, label in zip(X, y):
        values.append((input.tobytes(), label))

    cursor.executemany("INSERT INTO samples (input, label) VALUES (?, ?)", values)
    conn.commit()

    conn.close()


def hit_cache(
    name: str, rows: int, cols: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    inputs = np.empty((rows, cols), dtype=np.float32)
    labels = np.empty(rows, dtype=np.float32)
    for i, row in enumerate(cursor.execute("SELECT * FROM samples;")):
        inputs[i] = np.frombuffer(row[0], dtype=np.float32)
        labels[i] = np.frombuffer(row[1], dtype=np.float32)[0]

    conn.close()
    return inputs, labels
