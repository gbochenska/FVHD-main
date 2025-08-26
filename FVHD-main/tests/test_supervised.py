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

    # pca = PCA(n_components=50)
    # X = torch.tensor(pca.fit_transform(X), dtype=torch.float32)

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
        epochs=kwargs.get("epochs", 2000),
        eta=kwargs.get("eta", 0.2),
        device="cpu",
        verbose=False,
        mutual_neighbors_epochs=kwargs.get("mutual_neighbors_epochs", None),
        boost_start_eta=kwargs.get("boost_start_eta", False),
        gaussian_weights=kwargs.get("use_gaussian_weights", False),
        eta_schedule=kwargs.get("eta_schedule", ""),
        autoadapt=kwargs.get("autoadapt", False),
        velocity_limit=kwargs.get("velocity_limit", False),
        force_multiplier=kwargs.get("force_multiplier", 1.0),
        supervised=kwargs.get("velocity_limit", False),
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
    
    unique_labels = np.unique(y)
    centroids = []
    for label in unique_labels:
        class_points = embeddings[y == label]
        centroid = class_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.vstack(centroids)

    for idx, (x, y) in enumerate(centroids):  
        plt.scatter(x, y, marker='x', color='black', s=100) 
        plt.text(x, y, str(idx), fontsize=12, color='red') 

    plt.legend()
    plt.title(f"{name} - Silhouette: {score:.4f} - Time: {elapsed_time:.2f}s")
    if not os.path.exists("results"):
        os.makedirs("results")
    filename_base = name.replace(" ", "_")
    filename_base += DATASET_NAME
    plt.savefig(f"results/{filename_base}.png")
    plt.close()

    with open("results/summary.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, score, elapsed_time])

    print(f"{DATASET_NAME} Test {name} completed. Silhouette Score: {score:.4f}, Time: {elapsed_time:.2f}s\n")

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/summary.csv"):
    with open("results/summary.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Name", "Silhouette Score", "Time (s)"])


variants = [
    {"name": "Supervised", "supervised": True, },
    # # {"name": "Unsupervised", "supervised": False},
    # {"name": "Supervised + l1 = 0.5 + l2 = 1.0", "supervised": True, "l1": 0.5},
    # {"name": "Supervised + l1 = 1.0 + l2 = 0.5", "supervised": True, "l2": 0.5},
    {"name": "Supervised + l1 = 15 + l2 = 4", "supervised": True, "l2": 4, "l1": 15},
    {"name": "Supervised + l1 = 10.0 + l2 = 1.0", "supervised": True, "l1": 10.0},
    {"name": "Supervised + l1 = 1.0 + l2 = 10.0", "supervised": True, "l2": 10.0},
    {"name": "Supervised + l1 = 10.0 + l2 = 10.0", "supervised": True, "l2": 10.0, "l1": 10.0},
    ]

for variant in variants:
    run_variant_test(**variant)
