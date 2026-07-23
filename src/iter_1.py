import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score

from simulations.hbt import (BINS, EFF_1S, input_gen, label_gen, seed_env,
                             seed_gen)
from utils.cache_utils import cache_samples, create_cache, hit_cache
from utils.data_utils import create_loaders
from utils.plot_utils import plot_results

DATA_GENERATED = True
TRAINED = False
WEIGHTS_DIR = "weights"
ASSETS_DIR = "assets"

seed_env(10)
SEEDS = seed_gen(121)
TOTAL = len(EFF_1S) * len(SEEDS)

if not DATA_GENERATED:
    create_cache()

    for seed in SEEDS:
        inputs = input_gen(seed).astype(np.float32)
        labels = label_gen(seed).astype(np.float32)
        print(f"{seed} {inputs.shape} {labels.shape}")

        cache_samples(inputs, labels)

model = nn.Sequential()
model.add_module("conv1d_1", nn.Conv1d(1, 64, 25, 1, 12))
model.add_module("relu_1", nn.ReLU())
model.add_module("maxpool_1", nn.MaxPool1d(2, 2))
model.add_module("conv1d_2", nn.Conv1d(64, 128, 13, 1, 6))
model.add_module("relu_2", nn.ReLU())
model.add_module("maxpool_2", nn.MaxPool1d(2, 2))
model.add_module("flatten", nn.Flatten())
model.add_module("fc_1", nn.Linear(16_000, 4_096))
model.add_module("relu_3", nn.ReLU())
model.add_module("dropout_1", nn.Dropout(0.5))
model.add_module("fc_2", nn.Linear(4_096, 1_028))
model.add_module("relu_4", nn.ReLU())
model.add_module("dropout_2", nn.Dropout(0.25))
model.add_module("fc_3", nn.Linear(1_028, 64))
model.add_module("relu_4", nn.ReLU())
model.add_module("fc_4", nn.Linear(64, 1))
model.add_module("softplus", nn.Softplus())

inputs, labels = hit_cache(TOTAL, BINS)
train_loader, valid_loader, X_test, y_test = create_loaders(inputs, labels)

if not TRAINED:
    device = torch.device("cuda")
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 30

    train_loss_graph = np.zeros(epochs)
    valid_loss_graph = np.zeros(epochs)
    train_mae_graph = np.zeros(epochs)
    train_r2_graph = np.zeros(epochs)
    valid_mae_graph = np.zeros(epochs)
    valid_r2_graph = np.zeros(epochs)

    for epoch in range(epochs):
        train_actual = []
        train_pred = []
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch).squeeze(-1)
            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_graph[epoch] += loss.item() * y_batch.size(0)
            train_actual.append(y_batch.detach().cpu().numpy())
            train_pred.append(pred.detach().cpu().numpy())

        train_loss_graph[epoch] /= len(train_loader.dataset)  # type: ignore
        train_actual = np.concat(train_actual)
        train_pred = np.concat(train_pred)
        train_mae_graph[epoch] = mean_absolute_error(train_actual, train_pred)
        train_r2_graph[epoch] = r2_score(train_actual, train_pred)

        print(epoch + 1)
        print(f"training loss: {train_loss_graph[epoch]}")
        print(f"training mae: {train_mae_graph[epoch]}")
        print(f"training r2: {train_r2_graph[epoch]}")

        valid_actual = []
        valid_pred = []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                pred = model(X_batch).squeeze(-1)
                loss = loss_fn(pred, y_batch)

                valid_loss_graph[epoch] += loss.item() * y_batch.size(0)
                valid_actual.append(y_batch.detach().cpu().numpy())
                valid_pred.append(pred.detach().cpu().numpy())

            valid_loss_graph[epoch] /= len(valid_loader.dataset)  # type: ignore
            valid_actual = np.concat(valid_actual)
            valid_pred = np.concat(valid_pred)
            valid_mae_graph[epoch] = mean_absolute_error(valid_actual, valid_pred)
            valid_r2_graph[epoch] = r2_score(valid_actual, valid_pred)

            print(f"validation loss: {valid_loss_graph[epoch]}")
            print(f"validation mae: {valid_mae_graph[epoch]}")
            print(f"validation r2: {valid_r2_graph[epoch]}\n")

    torch.save(model.state_dict(), f"{WEIGHTS_DIR}/iter_1_weights.pth")

    plot_results(
        train_loss_graph,
        valid_loss_graph,
        "loss",
        f"{ASSETS_DIR}/iter_1_loss.png",
    )
    plot_results(
        train_mae_graph,
        valid_mae_graph,
        "mean absolute error",
        f"{ASSETS_DIR}/iter_1_mae.png",
    )
    plot_results(
        train_r2_graph,
        valid_r2_graph,
        "r2 score",
        f"{ASSETS_DIR}/iter_1_r2.png",
    )
else:
    model.load_state_dict(
        torch.load(f"{WEIGHTS_DIR}/iter_1_weights.pth", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        pred = model(X_test).squeeze(-1).numpy()
        actual = y_test.numpy()

        print(f"test mae: {mean_absolute_error(actual, pred)}")
        print(f"test r2: {r2_score(actual, pred)}")
