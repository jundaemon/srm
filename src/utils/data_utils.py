import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def train_validation_test_split(
    X: NDArray[np.float32],
    y: NDArray[np.float32],
    test_ratio: float,
    valid_ratio: float,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=int(len(X) * test_ratio),
        shuffle=True,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp,
        y_temp,
        test_size=int(len(X) * valid_ratio),
        shuffle=True,
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_loader(X: NDArray[np.float32], y: NDArray[np.float32]) -> DataLoader:
    X_tensor = torch.from_numpy(X)
    X_tensor = X_tensor.unsqueeze(1)

    y_tensor = torch.from_numpy(y)

    return DataLoader(TensorDataset(X_tensor, y_tensor), 64, True)
