from typing import Any, Dict, Optional, Type
import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

from knn.graph import Graph


class FVHD:
    def __init__(
        self,
        n_components: int = 2,
        nn: int = 2,
        rn: int = 1,
        c: float = 0.1,
        optimizer: Optional[Type[Optimizer]] = None,
        optimizer_kwargs: Dict[str, Any] = None,
        epochs: int = 200,
        eta: float = 0.1,
        device: str = "cpu",
        graph_file: str = "",
        autoadapt: bool = False,
        velocity_limit: bool = False,
        verbose: bool = True,
        mutual_neighbors_epochs: Optional[int] = None,
        eta_schedule: str = "",
        boost_start_eta: bool = True,
        gaussian_weights: bool = False,
        force_multiplier: float = 1.0,
        plot_each: int = 100,
        init_pos: Optional[torch.Tensor] = None,
        supervised: bool = False,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
    ) -> None:
        self.n_components = n_components
        self.nn = nn
        self.rn = rn
        self.c = c
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.epochs = epochs
        self.eta = eta
        self.initial_eta = eta
        self.eta_schedule = eta_schedule
        self.eta_decay_rate = 0.95
        self.eta_adaptive_threshold = 1e-3
        self.eta_min = 1e-4
        self.a = 0.9
        self.b = 0.3
        self.device = device
        self.verbose = verbose
        self.graph_file = graph_file
        self._current_epoch = 0
        self._previous_delta_norm = 1e9

        self.autoadapt = autoadapt
        self.buffer_len = 10
        self.curr_max_velo = torch.tensor([0.0] * self.buffer_len)
        self.curr_max_velo_idx = 1

        self.velocity_limit = velocity_limit
        self.max_velocity = 1.0
        self.vel_dump = 0.95

        self.x = None
        self.delta_x = None
        self.init_pos = init_pos

        self.mutual_neighbors_epochs = mutual_neighbors_epochs
        self.boost_start_eta = boost_start_eta
        self.gaussian_weights = gaussian_weights
        self.force_multiplier = force_multiplier
        self.plot_each = plot_each

        self.supervised = supervised
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def fit_transform(self, X: torch.Tensor, graphs: list[Graph], labels: Optional[np.ndarray] = None) -> np.ndarray:
        x = X.to(self.device)
        graph = graphs[0]
        nn = torch.tensor(graph.indexes[:, :self.nn].astype(np.int32)).to(self.device)
        rn = torch.randint(0, x.shape[0], (x.shape[0], self.rn)).to(self.device)

        nn = nn.reshape(-1)
        rn = rn.reshape(-1)

        if self.optimizer is None:
            return self.force_directed_method(x, nn, rn, graphs, labels)
        return self.optimizer_method(x.shape[0], nn, rn)

    def optimizer_method(self, N, NN, RN):
        if self.x is None:
            self.x = torch.rand((N, 1, self.n_components), requires_grad=True, device=self.device)
        optimizer = self.optimizer(params={self.x}, **self.optimizer_kwargs)
        for i in range(self.epochs):
            loss = self.optimizer_step(optimizer, NN, RN)
            if loss < 1e-10:
                return self.x[:, 0].detach()
            if self.verbose:
                print(f"\r{i} loss: {loss.item()}, X: {self.x[0]}", end="")
                if i % 100 == 0:
                    print()
        return self.x[:, 0].detach().cpu().numpy()

    def _calculate_distances(self, indices):
        diffs = self.x - torch.index_select(self.x, 0, indices).view(
            self.x.shape[0], -1, self.n_components
        )
        dist = torch.sqrt(
            torch.sum((diffs + 1e-8) * (diffs + 1e-8), dim=-1, keepdim=True)
        )
        return diffs, dist

    def optimizer_step(self, optimizer, NN, RN) -> Tensor:
        optimizer.zero_grad()
        nn_diffs, nn_dist = self._calculate_distances(NN)
        rn_diffs, rn_dist = self._calculate_distances(RN)

        loss = torch.mean(nn_dist * nn_dist) + self.c * torch.mean(
            (1 - rn_dist) * (1 - rn_dist)
        )
        loss.backward()
        optimizer.step()
        return loss

    @staticmethod
    def compute_supervised_loss(embeddings, labels):
        unique_labels = labels.unique()
        centroids = [embeddings[labels == label].mean(dim=0) for label in unique_labels]
        centroids = torch.stack(centroids).view(len(centroids), -1)

        intra_loss = torch.mean(torch.stack([
            torch.mean((embeddings[labels == label] - centroid).pow(2).sum(dim=1))
            for centroid, label in zip(centroids, unique_labels)
        ]))

        if len(centroids) > 1:
            dists = torch.cdist(centroids, centroids)
            mask = torch.triu(torch.ones_like(dists), diagonal=1)
            inter_loss = (dists * mask).sum() / mask.sum()
        else:
            inter_loss = torch.tensor(0.0, device=embeddings.device)

        return intra_loss, inter_loss

    def force_directed_method(self, X, NN, RN, graphs, labels=None) -> np.ndarray:
        def plot_embedding(x, labels, epoch, prev_centroids=None):
            import matplotlib.pyplot as plt
            labels = labels.numpy() if labels is not None else np.zeros(len(x), dtype=int)
            unique_labels = np.unique(labels)

            plt.figure(figsize=(6, 6))
            for i in unique_labels:
                points = x[labels == i]
                plt.scatter(points[:, 0], points[:, 1], label=str(i), marker=".", s=1, alpha=0.5)

            if self.supervised:
                centroids = np.vstack([np.mean(x[labels == i], axis=0) for i in unique_labels])
                if prev_centroids is not None:
                    plt.scatter(prev_centroids[:, 0], prev_centroids[:, 1], c='black', marker='o', s=100, edgecolors='k')
                plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100)

            plt.legend()
            plt.title(f"Embeddings - Epoch {epoch}")
            plt.savefig(f"embedding_epoch_{epoch:03}.png")
            plt.close()
            return centroids if self.supervised else None

        nn_new = torch.cat(
            [NN.reshape(X.shape[0], self.nn, 1) for _ in range(self.n_components)],
            dim=-1
        ).to(torch.long)  # ⬅ DODAJ TO

        rn_new = torch.cat(
            [RN.reshape(X.shape[0], self.rn, 1) for _ in range(self.n_components)],
            dim=-1
        ).to(torch.long)  # ⬅ DODAJ TO

        self.x = self.init_pos.to(self.device) if self.init_pos is not None else torch.rand(
            (X.shape[0], 1, self.n_components), device=self.device)
        self.delta_x = torch.zeros_like(self.x)

        prev_centroids = None

        for i in range(self.epochs):
            self._current_epoch = i
            loss = self.__force_directed_step(NN, RN, nn_new, rn_new, graphs)

            if self.supervised and labels is not None:
                intra_loss, inter_loss = self.compute_supervised_loss(self.x[:, 0], labels.to(self.device))
                loss = loss + self.lambda1 * intra_loss - self.lambda2 * inter_loss

                with torch.no_grad():
                    y = labels.to(self.device)
                    centroids = torch.stack([self.x[y == lbl, 0].mean(dim=0) for lbl in torch.unique(y)])
                    target = centroids[y]
                    self.x[:, 0] += 0.05 * (target - self.x[:, 0])

            if self.verbose and i % 100 == 0:
                print(f"\r{i} loss: {loss.item()}")

            if self.plot_each > 0 and i % self.plot_each == 0:
                prev_centroids = plot_embedding(self.x[:, 0].detach().cpu().numpy(), labels, i, prev_centroids)

        return self.x[:, 0].cpu().numpy()

    def __force_directed_step(self, NN, RN, NN_new, RN_new, graphs):
        if self.mutual_neighbors_epochs and self.epochs - self._current_epoch <= self.mutual_neighbors_epochs:
            graph = graphs[1]
            NN = torch.tensor(graph.indexes[:, :self.nn].astype(np.int32)).to(self.device).reshape(-1)
            NN_new = torch.cat([NN.reshape(self.x.shape[0], self.nn, 1) for _ in range(self.n_components)], dim=-1)

        nn_diffs, nn_dist = self._calculate_distances(NN)
        rn_diffs, rn_dist = self._calculate_distances(RN)

        f_nn, f_rn = self.__compute_forces(rn_dist, nn_diffs, rn_diffs, nn_dist, NN_new, RN_new)
        f = -self.force_multiplier * f_nn - self.c * f_rn

        self.delta_x = self.a * self.delta_x + self.b * f
        squared_velocity = torch.sum(self.delta_x * self.delta_x, dim=-1)
        velocity = torch.sqrt(squared_velocity)

        if self.velocity_limit:
            mask = squared_velocity > self.max_velocity ** 2
            self.delta_x[mask] *= (self.max_velocity / velocity[mask]).reshape(-1, 1)

        if self.eta_schedule == "decay":
            self.eta = max(self.eta * self.eta_decay_rate, self.eta_min)
        elif self.eta_schedule == "adaptive":
            delta_norm = torch.norm(self.delta_x)
            if self._previous_delta_norm - delta_norm < self.eta_adaptive_threshold:
                self.eta = max(self.eta * self.eta_decay_rate, self.eta_min)
            self._previous_delta_norm = delta_norm

        self.x += self.eta * self.delta_x

        if self.autoadapt:
            self._auto_adaptation(velocity)

        if self.velocity_limit:
            self.delta_x *= self.vel_dump

        return torch.mean(nn_dist ** 2) + self.c * torch.mean((1 - rn_dist) ** 2)

    def _auto_adaptation(self, sqrt_velocity):
        v_avg = self.delta_x.mean()
        self.curr_max_velo[self.curr_max_velo_idx] = sqrt_velocity.max()
        self.curr_max_velo_idx = (self.curr_max_velo_idx + 1) % self.buffer_len
        v_max = self.curr_max_velo.mean()

        if v_max > 10 * v_avg:
            self.eta /= 1.01
        elif v_max < 10 * v_avg:
            self.eta *= 1.01
        self.eta = max(self.eta, 0.01)

    def __compute_forces(self, rn_dist, nn_diffs, rn_diffs, nn_dist, NN_new, RN_new):
        if self.gaussian_weights:
            sigma = 2.0 * torch.mean(nn_dist)
            weights = torch.exp(- (nn_dist ** 2) / (sigma ** 2))
            weights = weights.view(nn_diffs.shape[0], nn_diffs.shape[1], 1)
            f_nn = weights * nn_diffs
        elif self.mutual_neighbors_epochs and self.epochs - self._current_epoch <= self.mutual_neighbors_epochs:
            nn_attraction = 1.0 / (nn_dist + 1e-8)
            f_nn = nn_attraction * nn_diffs
        else:
            f_nn = nn_diffs

        f_rn = (rn_dist - 1) / (rn_dist + 1e-8) * rn_diffs

        minus_f_nn = torch.zeros_like(f_nn).scatter_add_(src=f_nn, dim=0, index=NN_new)
        minus_f_rn = torch.zeros_like(f_rn).scatter_add_(src=f_rn, dim=0, index=RN_new)

        f_nn -= minus_f_nn
        f_rn -= minus_f_rn

        return (
            torch.sum(f_nn, dim=1, keepdim=True),
            torch.sum(f_rn, dim=1, keepdim=True),
        )
