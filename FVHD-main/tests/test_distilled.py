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

from fvhd.fvhd import FVHD, FVHDWithTransform
from sklearn.metrics import silhouette_score



def setup_ssl():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


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
        data = np.load("compressed_final_prototypes.npz")
        X = data["images"]  # shape: (1000, 28, 28)
        Y = data["labels"]  # shape: (1000,)
        X = X.reshape(len(X), -1)

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.long)
        return X, Y
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    X = dataset.data[:n_samples]
    N = len(X) if n_samples is None else n_samples
    X = X.reshape(N, -1) / 255.0

    X = X.clone().detach().to(torch.float32)

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

    FULL_DATASET      = "mnist"
    PROTOTYPE_DATASET = "custom_npz"     # plik 1000 prototypów


    # 1) wczytaj prototypy
    X_proto, y_proto = load_dataset(PROTOTYPE_DATASET)
    # 2) ucz FVHD tylko na prototypach
    graph, mutual_graph = create_or_load_graph(X_proto, 5)
    
    fvhd = FVHDWithTransform(n_components=2, nn=5, rn=2, c=0.1, eta=0.005,
                epochs=2000, supervised=False)

    fvhd.fit(X_proto, [graph, mutual_graph], labels=y_proto)

    # 3) wczytaj pełny MNIST i przekształć
    print("Teraz pełne")
    X_full, y_full = load_dataset(FULL_DATASET)    # Użyj TEGO SAMEGO pca

    start_time = time.time()

    Y_full = fvhd.project(X_full, X_proto)
    elapsed_time = time.time() - start_time
    visualize_embeddings(Y_full, y_full, "Pełny MNIST w przestrzeni prototypów")

    score = silhouette_score(Y_full, y_full)

    plt.figure(figsize=(6, 6))
    plt.title(f"{name} - Silhouette: {score:.4f} - Time: {elapsed_time:.2f}s")
    y = y_full.numpy()
    for i in range(10):
        points = Y_full[y == i]
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

    print(f"{FULL_DATASET} Test {name} completed. Silhouette Score: {score:.4f}, Time: {elapsed_time:.2f}s\n")


# Prepare CSV header
if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/summary.csv"):
    with open("results/summary.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Name", "Silhouette Score", "Time (s)"])


# === Test scenarios ===
variants = [
    {"name": "Distilled"}]

for variant in variants:
    run_variant_test(**variant)
