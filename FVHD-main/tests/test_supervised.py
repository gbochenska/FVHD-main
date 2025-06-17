import ssl
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import torchvision
import csv

from fvhd import FVHD
from knn import Graph, NeighborConfig, NeighborGenerator
from sklearn.metrics import silhouette_score


import time

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
        nn=5,
        rn=2,
        c=0.2,
        eta=0.2,
        optimizer=None,
        optimizer_kwargs={"lr": 0.1},
        epochs=2000,
        device="cpu",
        velocity_limit=True,
        autoadapt=True,
        mutual_neighbors_epochs=300,
        supervised=kwargs.get("supervised", True),
        lambda1=kwargs.get("l1", 1.0),
        lambda2=kwargs.get("l2", 1.0),
    )

    start_time = time.time()
    embeddings = fvhd.fit_transform(X, [graph, mutual_graph], labels=y)
    elapsed_time = time.time() - start_time

    score = silhouette_score(embeddings, y)
    print(f"Silhouette Score: {score:.4f}")


    plt.figure(figsize=(8, 8))
    y = y.numpy()
    for i in range(10):
        points = embeddings[y == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5)
    
    # Dorysowanie centroidów
    unique_labels = np.unique(y)
    centroids = []
    for label in unique_labels:
        class_points = embeddings[y == label]
        centroid = class_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.vstack(centroids)

    for idx, (x, y) in enumerate(centroids):  # centroidy w 2D
        plt.scatter(x, y, marker='x', color='black', s=100)  # duży X na centroidzie
        plt.text(x, y, str(idx), fontsize=12, color='red')  # numer klasy

    plt.legend()
    plt.title(f"{DATASET_NAME} FVHD Embedding + Centroidy klas")
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


variants = [
    # {"name": "Supervised", "supervised": True},
    # {"name": "Unsupervised", "supervised": False},
    # {"name": "Supervised + l1 = 0.5 + l2 = 1.0", "supervised": True, "l1": 0.5},
    # {"name": "Supervised + l1 = 1.0 + l2 = 0.5", "supervised": True, "l2": 0.5},
    # {"name": "Supervised + l1 = 0.5 + l2 = 0.5", "supervised": True, "l2": 0.5, "l1": 0.5},
    {"name": "Supervised + l1 = 10.0 + l2 = 1.0", "supervised": True, "l1": 10.0},
    {"name": "Supervised + l1 = 1.0 + l2 = 10.0", "supervised": True, "l2": 10.0},
    {"name": "Supervised + l1 = 10.0 + l2 = 10.0", "supervised": True, "l2": 10.0, "l1": 10.0},
    ]

for variant in variants:
    run_variant_test(**variant)
