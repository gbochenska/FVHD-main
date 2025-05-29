import torch
import torchvision
import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt
import ssl
from typing import Optional
import pandas as pd
from knn import Graph, NeighborConfig, NeighborGenerator

from fvhd import FVHD
from knn.graph import Graph
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

    fvhd = FVHD(
        n_components=2,
        nn=kwargs.get("nn", 5),
        rn=kwargs.get("rn", 2),
        c=kwargs.get("c", 0.2),
        epochs=kwargs.get("epochs", 1000),
        eta=kwargs.get("eta", 0.2),
        device="cpu",
        verbose=False,
        mutual_neighbors_epochs=kwargs.get("mutual_neighbors_epochs", None),
        boost_start_eta=kwargs.get("boost_start_eta", True),
        gaussian_weights=kwargs.get("use_gaussian_weights", False),
        eta_schedule=kwargs.get("eta_schedule", ""),
        autoadapt=kwargs.get("autoadapt", False),
        velocity_limit=kwargs.get("velocity_limit", False),
        force_multiplier=kwargs.get("force_multiplier", 1.0),
        plot_each=0
    )

    start_time = time.time()
    embedding = fvhd.fit_transform(X, [graph, mutual_graph])
    elapsed_time = time.time() - start_time

    score = silhouette_score(embedding, y)

    plt.figure(figsize=(6, 6))
    plt.title(f"{name} - Silhouette: {score:.4f} - Time: {elapsed_time:.2f}s")
    y = y.numpy()
    for i in range(10):
        points = embedding[y == i]
        plt.scatter(
            points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5
        )
    plt.legend()
    if not os.path.exists("results"):
        os.makedirs("results")
    filename_base = name.replace(" ", "_")
    plt.savefig(f"results/{filename_base}.png")
    plt.close()

    

    with open("results/summary.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, score, elapsed_time])

    print(f"Test {name} completed. Silhouette Score: {score:.4f}, Time: {elapsed_time:.2f}s\n")


# Prepare CSV header
if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/summary.csv"):
    with open("results/summary.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Name", "Silhouette Score", "Time (s)"])


# === Test scenarios ===
variants = [
    {"name": "Baseline"},
    {"name": "With mutual neighbors", "mutual_neighbors_epochs": 50},
    {"name": "With Gaussian weights", "use_gaussian_weights": True},
    {"name": "Init by labels", "boost_start_eta": False},
    {"name": "Decay eta", "eta_schedule": "decay"},
    {"name": "Adaptive eta", "eta_schedule": "adaptive"},
    {"name": "With autoadapt", "autoadapt": True},
    {"name": "Velocity limited", "velocity_limit": True},
    {"name": "Strong repulsion", "c": 1.0},
    {"name": "Weak repulsion", "c": 0.01},
    {"name": "Mutual + Gaussian + Decay", "mutual_neighbors_epochs": 50, "use_gaussian_weights": True, "eta_schedule": "decay"},
    {"name": "Init + Adaptive + Autoadapt", "boost_start_eta": False, "eta_schedule": "adaptive", "autoadapt": True},
    {"name": "Gaussian + Velocity", "use_gaussian_weights": True, "velocity_limit": True},
    {"name": "Mutual + Strong repulsion", "mutual_neighbors_epochs": 50, "c": 1.0},
    {"name": "Weak repulsion + Adaptive + Autoadapt", "c": 0.01, "eta_schedule": "adaptive", "autoadapt": True}
]

for variant in variants:
    run_variant_test(**variant)
