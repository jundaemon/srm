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
        random_state=1
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp,
        y_temp,
        test_size=int(len(X) * valid_ratio),
        shuffle=True,
        random_state=1
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_loaders(
    X: NDArray[np.float32], y: NDArray[np.float32]
) -> tuple[DataLoader, DataLoader, Tensor, Tensor]:
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_validation_test_split(
        X, y, 0.15, 0.15
    )
    X_train = torch.from_numpy(X_train)
    X_train = X_train.unsqueeze(1)

    X_valid = torch.from_numpy(X_valid)
    X_valid = X_valid.unsqueeze(1)

    X_test = torch.from_numpy(X_test)
    X_test = X_test.unsqueeze(1)

    y_train = torch.from_numpy(y_train)
    y_valid = torch.from_numpy(y_valid)
    y_test = torch.from_numpy(y_test)

    return (
        DataLoader(TensorDataset(X_train, y_train), 64, True),
        DataLoader(TensorDataset(X_valid, y_valid), 64, True),
        X_test,
        y_test,
    )
