import ssl
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

from fvhd import FVHD
from knn import Graph, NeighborConfig, NeighborGenerator
from sklearn.metrics import silhouette_score


def setup_ssl():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


def load_dataset(name: str, n_samples: Optional[int] = None):
    if name == "mnist":
        dataset = torchvision.datasets.MNIST("mnist", train=True, download=True)
    elif name == "emnist":
        dataset = torchvision.datasets.EMNIST(
            "emnist", split="balanced", train=True, download=True
        )
    elif name == "fmnist":
        dataset = torchvision.datasets.FashionMNIST(
            "fashionMNIST", train=True, download=True
        )
    elif name == "custom_npz":
        data = np.load("mnist_1000_distilled_like.npz")
        X = data["images"]  # shape: (1000, 28, 28)
        Y = data["labels"]  # shape: (1000,)
        X = X.reshape(len(X), -1)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        X = torch.tensor(pca.fit_transform(X), dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.long)
        return X, Y
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    X = dataset.data[:n_samples]
    N = len(X) if n_samples is None else n_samples
    X = X.reshape(N, -1) / 255.0

    from sklearn.decomposition import PCA

    pca = PCA(n_components=50)
    X = torch.tensor(pca.fit_transform(X), dtype=torch.float32)

    Y = dataset.targets[:n_samples]
    return X, Y


def create_or_load_graph(X: torch.Tensor, nn: int) -> tuple[Graph, Graph]:
    config = NeighborConfig(metric="euclidean")
    df = pd.DataFrame(X.numpy())
    generator = NeighborGenerator(df=df, config=config)
    return generator.run(nn=nn)


def visualize_embeddings(x: np.ndarray, y: torch.Tensor, dataset_name: str):
    plt.switch_backend("TkAgg")
    plt.figure(figsize=(8, 8))
    plt.title(f"{dataset_name} 2d visualization")

    y = y.numpy()
    for i in range(10):
        points = x[y == i]
        plt.scatter(
            points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5
        )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    setup_ssl()

    DATASET_NAME = "custom_npz"

    X, Y = load_dataset(DATASET_NAME)
    graph, mutual_graph = create_or_load_graph(X, 5)

    fvhd = FVHD(
        n_components=2,
        nn=5,
        rn=2,
        c=3.5,
        eta=0.2,
        optimizer=None,
        optimizer_kwargs={"lr": 0.1},
        epochs=2000,
        device="cpu",
        velocity_limit=True,
        autoadapt=True,
        mutual_neighbors_epochs=300,
        supervised=False
    )

    # print(fvhd.eta_schedule)

    embeddings = fvhd.fit_transform(X, [graph, mutual_graph], labels=Y)

    score = silhouette_score(embeddings, Y)
    print(f"Silhouette Score: {score:.4f}")
    visualize_embeddings(embeddings, Y, DATASET_NAME)


    plt.figure(figsize=(8, 8))
    Y = Y.numpy()
    for i in range(10):
        points = embeddings[Y == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5)
    
    plt.legend()
    plt.title(f"{DATASET_NAME} FVHD Embedding + Centroidy klas")
    plt.show()
