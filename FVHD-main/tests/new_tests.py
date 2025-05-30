import torch
import torchvision
import numpy as np
from fvhd import FVHD
from knn.graph import Graph
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import csv
import time
import pandas as pd
from knn import Graph, NeighborConfig, NeighborGenerator
import ssl
from typing import Optional

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


def run_variant_test(name, **kwargs):
    print(f"\nRunning test: {name}")
    setup_ssl()

    DATASET_NAME = "mnist"

    X, y = load_dataset(DATASET_NAME)
    graph, mutual_graph = create_or_load_graph(X, 5)

    init_method = kwargs.get("init", "random")
    if init_method == "pca":
        pca = PCA(n_components=2)
        init_pos = torch.tensor(pca.fit_transform(X), dtype=torch.float32).unsqueeze(1)
    elif init_method == "labels":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        angles = 2 * np.pi * y_enc / len(np.unique(y))
        init_array = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        init_pos = torch.tensor(init_array, dtype=torch.float32).unsqueeze(1)
    else:
        init_pos = None

    fvhd = FVHD(
        n_components=2,
        nn=kwargs.get("nn", 5),
        rn=kwargs.get("rn", 2),
        c=kwargs.get("c", 0.1),
        epochs=kwargs.get("epochs", 1000),
        eta=kwargs.get("eta", 0.2),
        device="cpu",
        verbose=False,
        mutual_neighbors_epochs=kwargs.get("mutual_neighbors_epochs", None),
        autoadapt=kwargs.get("autoadapt", False),
        velocity_limit=kwargs.get("velocity_limit", False),
        eta_schedule=kwargs.get("eta_schedule", ""),
        gaussian_weights=kwargs.get("gaussian_weights", False),
        force_multiplier=kwargs.get("force_multiplier", 1.0),
        optimizer=kwargs.get("optimizer", None),
        optimizer_kwargs=kwargs.get("optimizer_kwargs", None),
        plot_each=0,
        init_pos=init_pos
    )

    start_time = time.time()
    embedding = fvhd.fit_transform(X, [graph, mutual_graph])
    elapsed_time = time.time() - start_time

    score = silhouette_score(embedding, y)

    plt.figure(figsize=(6, 6))
    plt.title(f"{name} - Silhouette: {score:.4f} - Time: {elapsed_time:.2f}s")
    y = y.numpy()
    for i in range(20):
        points = embedding[y == i]
        plt.scatter(
            points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5
        )
    plt.legend()
    if not os.path.exists("results"):
        os.makedirs("results")
    filename_base = name.replace(" ", "_")
    print("name", DATASET_NAME)
    plt.savefig(f"results/{filename_base}_{DATASET_NAME}.png")
    plt.close()

    with open("results/summary.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, score, elapsed_time])

    print(f"{DATASET_NAME} Test: {name} completed. Silhouette Score: {score:.4f}, Time: {elapsed_time:.2f}s\n")


# Prepare CSV header
if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/summary.csv"):
    with open("results/summary.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Name", "Silhouette Score", "Time (s)"])


    # variants = [
    #     # {"name": "Init by PCA", "init": "pca"},
    #     # {"name": "Init by labels", "init": "labels"},
    #     # {"name": "PCA + Adaptive + Autoadapt", "init": "pca", "eta_schedule": "adaptive", "autoadapt": True},
    #     {"name": "Labels + Gaussian + Velocity", "init": "labels", "gaussian_weights": True, "velocity_limit": True},
    #     # {"name": "Mutual + Gaussian + Decay", "mutual_neighbors_epochs": 50, "gaussian_weights": True, "eta_schedule": "decay"},
    #     # {"name": "Full combo", "init": "pca", "gaussian_weights": True, "mutual_neighbors_epochs": 50, "eta_schedule": "adaptive", "velocity_limit": True, "autoadapt": True, "c": 0.05},
    # ]

    # variants = [
    #     {"name": "Labels + Gaussian + Velocity1", "init": "labels", "gaussian_weights": True, "velocity_limit": True},
    #     {"name": "Weak repulsion + Init labels", "c": 0.01, "init": "labels"},
    #     # {"name": "PCA + Gaussian", "init": "pca", "gaussian_weights": True},
    #     {"name": "Labels + Velocity + Adaptive", "init": "labels", "velocity_limit": True, "eta_schedule": "adaptive"},
    #     # {"name": "Optimizer Adam + Velocity", "init": "labels", "velocity_limit": True, "optimizer": torch.optim.Adam, "optimizer_kwargs": {"lr": 0.01}},
    #     # {"name": "Optimizer SGD + Adaptive", "init": "labels", "eta_schedule": "adaptive", "optimizer": torch.optim.SGD, "optimizer_kwargs": {"lr": 0.05}},
    #     # {"name": "Optimizer RMSprop + PCA", "init": "pca", "optimizer": torch.optim.RMSprop, "optimizer_kwargs": {"lr": 0.01}},
    #     # {"name": "Optimizer AdamW + Gaussian", "init": "labels", "gaussian_weights": True, "optimizer": torch.optim.AdamW, "optimizer_kwargs": {"lr": 0.01}},
    # ]

variants = [
# {"name": "Labels + Weak repulsion + Velocity + Adaptive", "init": "labels", "c": 0.01, "velocity_limit": True, "eta_schedule": "adaptive"},
# {"name": "Labels + PCA + Velocity", "init": "pca", "velocity_limit": True},
# {"name": "Labels + Gaussian + Adaptive", "init": "labels", "use_gaussian_weights": True, "eta_schedule": "adaptive"},
# {"name": "Labels + Weak repulsion + Full Combo", "init": "labels", "c": 0.01, "velocity_limit": True, "use_gaussian_weights": True, "eta_schedule": "adaptive"},
# {"name": "PCA + RMSprop + Weak repulsion", "init": "pca", "optimizer": torch.optim.RMSprop, "optimizer_kwargs": {"lr": 0.01}, "c": 0.01},
# {"name": "Labels + NAdam + Adaptive", "init": "labels", "optimizer": torch.optim.NAdam, "optimizer_kwargs": {"lr": 0.005}, "eta_schedule": "adaptive"},
{"name": "Labels + Gaussian + Velocity1", "init": "labels", "gaussian_weights": True, "velocity_limit": True}, 
]

for variant in variants:
    run_variant_test(**variant)
