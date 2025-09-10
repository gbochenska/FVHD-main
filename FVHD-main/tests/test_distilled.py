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
        if len(points) == 0:
            continue
        plt.scatter(points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5)
    plt.legend()
    plt.savefig(f"{dataset_name}2d.png")
    plt.close()


from torchvision import transforms
from torch.utils.data import TensorDataset

def get_balanced_mnist_tensor_dataset(root="mnist", n_per_class=100, train=True, flatten=True, seed=42):
    ds = torchvision.datasets.FashionMNIST(root, train=train, download=True)
    targets = ds.targets.cpu().numpy() if isinstance(ds.targets, torch.Tensor) else np.asarray(ds.targets)

    rng = np.random.default_rng(seed)
    idxs = []
    for c in range(10):
        class_idx = np.where(targets == c)[0]
        chosen = rng.choice(class_idx, size=n_per_class, replace=False)
        idxs.extend(chosen)
    rng.shuffle(idxs)

    to_tensor = transforms.ToTensor()  # PIL -> torch.float32 [0,1], shape [1,28,28]

    X_list, y_list = [], []
    for i in idxs:
        img_pil, y = ds[i]
        x = to_tensor(img_pil)
        if flatten:
            x = x.view(-1)
        X_list.append(x)
        y_list.append(int(y))

    X = torch.stack(X_list).to(torch.float32)
    Y = torch.tensor(y_list, dtype=torch.long)
    return X, Y


def load_dataset(name: str, n_samples: Optional[int] = None):
    if name == "mnist":
        dataset = torchvision.datasets.MNIST("mnist", train=True, download=True)
    elif name == "emnist":
        dataset = torchvision.datasets.EMNIST("emnist", split="balanced", train=True, download=True)
    elif name == "fmnist":
        dataset = torchvision.datasets.FashionMNIST("fashionMNIST", train=True, download=True)
    elif name == "custom_npz":
        data = np.load("fmnist_28x28_float64_corrected.npz")
        X = data["images"]  # shape: (1000, 28, 28)
        Y = data["labels"]  # shape: (1000,)
        X = X.reshape(len(X), -1)
        if X.max() > 1.0:
            X = X / 255.0

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.long)

        # Xb, Yb = get_balanced_mnist_tensor_dataset(root="fashionMNIST", n_per_class=100)  
        # X, Y = torch.cat([X, Xb]), torch.cat([Y, Yb])

        return X, Y
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    X = dataset.data[:n_samples]
    N = len(X) if n_samples is None else n_samples
    X = X.reshape(N, -1)
    if X.max() > 1.0:
        X = X / 255.0

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

    FULL_DATASET      = "fmnist"
    PROTOTYPE_DATASET = "custom_npz"     # plik 1000 prototypów

    # --- 1) PROTOTYPY + jedna normalizacja (mu/sd) policzona na prototypach ---
    X_proto, y_proto = load_dataset(PROTOTYPE_DATASET)
    mu = X_proto.mean(dim=0, keepdim=True)
    sd = X_proto.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xp = (X_proto - mu) / sd

    # graf musi mieć tyle samo sąsiadów co FVHD.nn
    k = kwargs.get("nn", 6)
    graph, mutual_graph = create_or_load_graph(Xp, k)

    fvhd = FVHDWithTransform(
        n_components=2,
        nn=kwargs.get("nn", 5),
        rn=kwargs.get("rn", 2),
        c=kwargs.get("c", 0.2),
        epochs=kwargs.get("epochs", 2000),
        eta=kwargs.get("eta", 0.05),
        device="cpu",
        verbose=False,
        mutual_neighbors_epochs=kwargs.get("mutual_neighbors_epochs", None),
        boost_start_eta=kwargs.get("boost_start_eta", False),
        gaussian_weights=kwargs.get("gaussian_weights", False),  # <- poprawka klucza
        eta_schedule=kwargs.get("eta_schedule", "decay"),
        autoadapt=kwargs.get("autoadapt", False),
        velocity_limit=kwargs.get("velocity_limit", True),
        force_multiplier=kwargs.get("force_multiplier", 1.0),
        supervised=kwargs.get("supervised", False),
        top_k=kwargs.get("top_k", 30),
        p=kwargs.get("p", 0.5),
        sigma_k=kwargs.get("sigma_k", 10.0)
    )

    # delikatniejsze momentum
    fvhd.a = kwargs.get("a", 0.9)
    fvhd.b = kwargs.get("b", 0.2)

    start_time = time.time()
    Y_proto = fvhd.fit(Xp, [graph, mutual_graph], labels=y_proto)
    score_proto = silhouette_score(Y_proto, y_proto)
    visualize_embeddings(Y_proto, y_proto, name)
    print(f"[proto] silhouette: {score_proto:.4f}")

    # --- 2) PEŁNY ZBIÓR (też NPZ) + ta sama normalizacja ---
    # print("Teraz pełne")
    # X_full, y_full = load_dataset(FULL_DATASET)
    # Xf = (X_full - mu) / sd

    # Y_full = fvhd.project(Xf, Xp)
    # elapsed_time = time.time() - start_time

    # score = silhouette_score(Y_full, y_full)

    # plt.figure(figsize=(6, 6))
    # plt.title(f"{name} - Silhouette: {score:.4f} - Time: {elapsed_time:.2f}s")
    # y = y_full.numpy()
    # for i in range(10):
    #     pts = Y_full[y == i]
    #     if len(pts) == 0:
    #         continue
    #     plt.scatter(pts[:, 0], pts[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5)
    # plt.legend()
    # if not os.path.exists("results"):
    #     os.makedirs("results")
    # filename_base = name.replace(" ", "_")
    # plt.savefig(f"results/{filename_base}{FULL_DATASET}.png")
    # plt.close()

    # with open("results/summary.csv", mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([name, score, elapsed_time])

    # print(f"{FULL_DATASET} Test {name} completed. Silhouette Score: {score:.4f}, Time: {elapsed_time:.2f}s\n")


# -- init wyników --
if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/summary.csv"):
    with open("results/summary.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Name", "Silhouette Score", "Time (s)"])


# --- 13 WARIANTÓW ---
variants = [
    # # bardzo lokalnie, ostro (ryzyko przyklejeń)
    # {"name":"Distilled p=1.3 topk=16 cT=0.00 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":16,"p":1.3,
    #  "c_transform":0.00,"eta_transform":0.03,"relax_steps":10},

    # {"name":"Distilled p=2.0 topk=16 cT=0.00 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":16,"p":2.0,
    #  "c_transform":0.00,"eta_transform":0.03,"relax_steps":10},

    # # mały/średni kontekst — często dobry kompromis
    # {"name":"Distilled p=0.7 topk=32 cT=0.00 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":32,"p":0.7,
    #  "c_transform":0.00,"eta_transform":0.03,"relax_steps":12},

    # {"name":"Distilled p=0.9 topk=32 cT=0.00 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":32,"p":0.9,
    #  "c_transform":0.00,"eta_transform":0.03,"relax_steps":12},

    # {"name":"Distilled p=1.1 topk=32 cT=0.00 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":32,"p":1.1,
    #  "c_transform":0.00,"eta_transform":0.03,"relax_steps":12},

    # # szerzej (stabilniejsze OOS)
    # {"name":"Distilled p=0.7 topk=48 cT=0.00 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":48,"p":0.7,
    #  "c_transform":0.00,"eta_transform":0.03,"relax_steps":12},

    # {"name":"Distilled p=0.9 topk=48 cT=0.00 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":48,"p":0.9,
    #  "c_transform":0.00,"eta_transform":0.03,"relax_steps":12},

    # {"name":"Distilled p=1.1 topk=48 cT=0.00 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":48,"p":1.1,
    #  "c_transform":0.00,"eta_transform":0.03,"relax_steps":12},

    # # duży kontekst (bardzo stabilnie, ale rozmywa — dobra kontrola)
    # {"name":"Distilled p=0.5 topk=72 cT=0.00 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":72,"p":0.5,
    #  "c_transform":0.00,"eta_transform":0.03,"relax_steps":10},

    # {"name":"Distilled p=0.9 topk=72 cT=0.00 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":72,"p":0.9,
    #  "c_transform":0.00,"eta_transform":0.03,"relax_steps":10},

    # # lekka lokalna repulsja w relaxie (cT>0) — potrafi poprawić separację OOS
    # {"name":"Distilled p=0.9 topk=40 cT=0.02 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":40,"p":0.9,
    #  "c_transform":0.02,"eta_transform":0.03,"relax_steps":12},

    # {"name":"Distilled p=0.9 topk=48 cT=0.02 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":48,"p":0.9,
    #  "c_transform":0.02,"eta_transform":0.03,"relax_steps":12},

    # {"name":"Distilled p=0.7 topk=56 cT=0.02 etaT=0.025",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":56,"p":0.7,
    #  "c_transform":0.02,"eta_transform":0.025,"relax_steps":15},

    # # test ostrzejszych wag + minimalny cT
    # {"name":"Distilled p=1.3 topk=40 cT=0.01 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":40,"p":1.3,
    #  "c_transform":0.01,"eta_transform":0.03,"relax_steps":12},

    # # kontrolne ekstrema
    # {"name":"Distilled p=3.0 topk=20 cT=0.00 etaT=0.03",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":20,"p":3.0,
    #  "c_transform":0.00,"eta_transform":0.03,"relax_steps":8},

    # {"name":"Distilled p=0.3 topk=96 cT=0.00 etaT=0.02",
    #  "gaussian_weights":False,"mutual_neighbors_epochs":0,
    #  "velocity_limit":True,"top_k":96,"p":0.3,
    #  "c_transform":0.00,"eta_transform":0.02,"relax_steps":8},
    {"name":"Distilled nn=6 rn=1 c=0.10 eta=0.040",
     "nn":6,"rn":1,"c":0.10,"eta":0.040,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},

    {"name":"Distilled nn=8 rn=1 c=0.095 eta=0.038",
     "nn":8,"rn":1,"c":0.095,"eta":0.038,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},

    {"name":"Distilled nn=10 rn=1 c=0.090 eta=0.037",
     "nn":10,"rn":1,"c":0.090,"eta":0.037,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},

    {"name":"Distilled nn=12 rn=1 c=0.088 eta=0.036",
     "nn":12,"rn":1,"c":0.088,"eta":0.036,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},

    # rn = 2 — kompromis separacja/ciągłość
    {"name":"Distilled nn=6 rn=2 c=0.095 eta=0.040",
     "nn":6,"rn":2,"c":0.095,"eta":0.040,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},

    {"name":"Distilled nn=8 rn=2 c=0.090 eta=0.038",
     "nn":8,"rn":2,"c":0.090,"eta":0.038,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},

    {"name":"Distilled nn=10 rn=2 c=0.085 eta=0.036",
     "nn":10,"rn":2,"c":0.085,"eta":0.036,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},

    {"name":"Distilled nn=12 rn=2 c=0.082 eta=0.035",
     "nn":12,"rn":2,"c":0.082,"eta":0.035,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},

    # rn = 3 — mocniejsze lokalne spójności
    {"name":"Distilled nn=6 rn=3 c=0.100 eta=0.038",
     "nn":6,"rn":3,"c":0.100,"eta":0.038,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},

    {"name":"Distilled nn=8 rn=3 c=0.092 eta=0.037",
     "nn":8,"rn":3,"c":0.092,"eta":0.037,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},

    {"name":"Distilled nn=10 rn=3 c=0.088 eta=0.035",
     "nn":10,"rn":3,"c":0.088,"eta":0.035,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},

    {"name":"Distilled nn=12 rn=3 c=0.085 eta=0.034",
     "nn":12,"rn":3,"c":0.085,"eta":0.034,
     "gaussian_weights":False,"mutual_neighbors_epochs":0,"velocity_limit":True},
]

for variant in variants:
    run_variant_test(**variant)
