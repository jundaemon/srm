import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score

from simulations.hbt import (EFF_1S, INPUT_N, input_gen, label_gen, seed_env,
                             seed_gen)
from utils.cache_utils import (TEST_DB, TRAIN_DB, VALID_DB, cache_samples,
                               create_cache, hit_cache)
from utils.data_utils import create_loader, train_validation_test_split
from utils.plot_utils import plot_results

DATA_GENERATED = False
TRAINED = False
WEIGHTS_PATH = "weights/iter_1_weights.pth"

seed_env(10)
SEEDS = seed_gen(121)
TOTAL = len(SEEDS) * len(EFF_1S)

if not DATA_GENERATED:
    create_cache(TRAIN_DB)
    create_cache(TEST_DB)
    create_cache(VALID_DB)

    for seed in SEEDS:
        inputs = input_gen(seed).astype(np.float32)
        labels = label_gen(seed).astype(np.float32)

        X_train, X_valid, X_test, y_train, y_valid, y_test = (
            train_validation_test_split(inputs, labels, 0.15, 0.15)
        )

        print(seed)
        print(
            f"X train: {(X_train.shape, X_train.dtype)}, X valid shape: {(X_valid.shape, X_valid.dtype)}, X test shape: {(X_test.shape, X_test.dtype)}"
        )
        print(
            f"y train: {(y_train.shape, y_train.dtype)}, y valid shape: {(y_valid.shape, y_valid.dtype)}, y test shape: {(y_test.shape, y_test.dtype)}\n"
        )

        cache_samples(TRAIN_DB, seed, X_train, y_train)
        cache_samples(VALID_DB, seed, X_valid, y_valid)
        cache_samples(TEST_DB, seed, X_test, y_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}\n")

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
model.add_module("fc_3", nn.Linear(64, 1))
model.add_module("softplus", nn.Softplus())
model.to(device)

if not TRAINED:
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
        for seed in SEEDS:
            inputs, labels = hit_cache(TRAIN_DB, seed, len(EFF_1S), INPUT_N)
            train_loader = create_loader(inputs, labels)

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

        train_loss_graph[epoch] /= TOTAL
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
            for seed in SEEDS:
                inputs, labels = hit_cache(VALID_DB, seed, len(EFF_1S), INPUT_N)
                valid_loader = create_loader(inputs, labels)

                for X_batch, y_batch in valid_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    pred = model(X_batch).squeeze(-1)
                    loss = loss_fn(pred, y_batch)

                    valid_loss_graph[epoch] += loss.item() * y_batch.size(0)
                    valid_actual.append(y_batch.detach().cpu().numpy())
                    valid_pred.append(pred.detach().cpu().numpy())

            valid_loss_graph[epoch] /= TOTAL
            valid_actual = np.concat(valid_actual)
            valid_pred = np.concat(valid_pred)
            valid_mae_graph[epoch] = mean_absolute_error(valid_actual, valid_pred)
            valid_r2_graph[epoch] = r2_score(valid_actual, valid_pred)

            print(f"validation loss: {valid_loss_graph[epoch]}")
            print(f"validation mae: {valid_mae_graph[epoch]}")
            print(f"validation r2: {valid_r2_graph[epoch]}")

    torch.save(model.state_dict(), WEIGHTS_PATH)

    plot_results(
        train_loss_graph,
        valid_loss_graph,
        "training loss",
        "validation loss",
        "epochs - loss",
        "loss",
        "iter_1_loss",
    )
    plot_results(
        train_mae_graph,
        valid_mae_graph,
        "training mean absolute error",
        "validation mean absolute error",
        "epochs - mean absolute error",
        "mean absoute error",
        "iter_1_mae",
    )
    plot_results(
        train_r2_graph,
        valid_r2_graph,
        "training r2 score",
        "validation r2 score",
        "epochs - r2 score",
        "r2 score",
        "iter_1_r2",
    )
