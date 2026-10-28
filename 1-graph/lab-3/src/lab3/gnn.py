import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected


@dataclass(frozen=True)
class GnnConfig:
    kind: str
    hidden: int
    heads: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int
    seed: int


@dataclass(frozen=True)
class TrainResult:
    best_epoch: int
    cosine_test_auc: float
    val_auc: float
    test_auc: float
    test_ap: float


class LinkEncoder(nn.Module):
    def __init__(self, kind: str, in_dim: int, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        if kind == "gat":
            self.first = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
            self.second = GATConv(hidden * heads, hidden, heads=1, dropout=dropout)
        elif kind == "sage":
            self.first = SAGEConv(in_dim, hidden * heads)
            self.second = SAGEConv(hidden * heads, hidden)
        else:
            raise ValueError(f"unknown encoder {kind}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = functional.elu(self.first(x, edge_index))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        return self.second(x, edge_index)


SCALE = 10.0


def decode(z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    left = functional.normalize(z[pairs[0]], dim=-1)
    right = functional.normalize(z[pairs[1]], dim=-1)
    return (left * right).sum(dim=-1) * SCALE


def scores(model: LinkEncoder, data: Data) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        logits = decode(z, data.edge_label_index)
    return data.edge_label.cpu().numpy(), torch.sigmoid(logits).cpu().numpy()


def split_edges(x: np.ndarray, edges: np.ndarray, device: str) -> tuple[Data, Data, Data]:
    data = Data(x=torch.tensor(x, dtype=torch.float32), edge_index=to_undirected(torch.tensor(edges)))
    split = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False)
    train, val, test = split(data)
    return train.to(device), val.to(device), test.to(device)


def known_pairs(edge_index: torch.Tensor, num_nodes: int, device: str) -> torch.Tensor:
    mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    mask[edge_index[0], edge_index[1]] = True
    mask = mask | mask.T
    return mask.to(device)


def sample_negatives(positive: torch.Tensor, num_nodes: int, known: torch.Tensor) -> torch.Tensor:
    count = positive.size(1)
    negatives = torch.empty((2, count), dtype=torch.long, device=positive.device)
    negatives[0] = positive[0]
    negatives[1] = torch.randint(num_nodes, (count,), device=positive.device)
    bad = known[negatives[0], negatives[1]] | (negatives[0] == negatives[1])
    while bad.any():
        negatives[1, bad] = torch.randint(num_nodes, (int(bad.sum()),), device=positive.device)
        bad = known[negatives[0], negatives[1]] | (negatives[0] == negatives[1])
    return negatives


def train_step(model: LinkEncoder, optimizer: torch.optim.Optimizer, train: Data, known: torch.Tensor) -> float:
    model.train()
    optimizer.zero_grad()
    z = model(train.x, train.edge_index)
    positive = train.edge_label_index
    negative = sample_negatives(positive, train.num_nodes, known)
    pairs = torch.cat([positive, negative], dim=1)
    labels = torch.cat([torch.ones(positive.size(1)), torch.zeros(negative.size(1))]).to(z.device)
    loss = functional.binary_cross_entropy_with_logits(decode(z, pairs), labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_link_prediction(
    x: np.ndarray, edges: np.ndarray, config: GnnConfig, device: str
) -> tuple[LinkEncoder, TrainResult]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    train, val, test = split_edges(x, edges, device)
    known = known_pairs(train.edge_index, train.num_nodes, device)
    model = LinkEncoder(config.kind, x.shape[1], config.hidden, config.heads, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_auc, best_epoch, best_state = -1.0, 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, config.epochs + 1):
        train_step(model, optimizer, train, known)
        labels, probabilities = scores(model, val)
        auc = float(roc_auc_score(labels, probabilities))
        if auc > best_auc:
            best_auc, best_epoch, best_state = auc, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break
    model.load_state_dict(best_state)
    labels, probabilities = scores(model, test)
    result = TrainResult(
        best_epoch=best_epoch,
        cosine_test_auc=round(float(roc_auc_score(labels, decode(test.x, test.edge_label_index).cpu().numpy())), 4),
        val_auc=round(best_auc, 4),
        test_auc=round(float(roc_auc_score(labels, probabilities)), 4),
        test_ap=round(float(average_precision_score(labels, probabilities)), 4),
    )
    return model, result


def embed(model: LinkEncoder, x: np.ndarray, edges: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    edge_index = torch.tensor(edges, dtype=torch.long)
    if edge_index.numel():
        edge_index = to_undirected(edge_index)
    with torch.no_grad():
        z = model(torch.tensor(x, dtype=torch.float32, device=device), edge_index.to(device))
    z = functional.normalize(z, dim=-1)
    return z.cpu().numpy()
