import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def plot_ce(taus: NDArray[np.float64], bins: int, path: str) -> None:
    plt.hist(taus, bins)
    plt.title("tau - coincidence events")
    plt.xlabel("tau")
    plt.ylabel("coincidence events")
    plt.savefig(path)
    plt.close("all")


def plot_ce_via_hist(
    hist: NDArray[np.int64], edges: NDArray[np.float64], path: str
) -> None:
    plt.stairs(hist, edges, fill=True, color="y")
    plt.title("tau - coincidence events")
    plt.xlabel("tau")
    plt.ylabel("coincidence events")
    plt.savefig(path)
    plt.close("all")


def plot_results(
    train_arr: NDArray[np.float64],
    valid_arr: NDArray[np.float64],
    metric: str,
    path: str,
) -> None:
    epochs = np.arange(1, len(train_arr) + 1)
    plt.plot(epochs, train_arr, label=f"training {metric}")
    plt.plot(epochs, valid_arr, label=f"validation {metric}", color="y")

    plt.title(f"epochs - {metric}")
    plt.xlabel("epochs")
    plt.ylabel(metric)

    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close("all")
