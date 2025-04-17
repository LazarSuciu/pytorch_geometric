import sys

from functools import reduce
import os
import os.path as osp
import math
import pprint
import time

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch_geometric.nn.models import NBFNet_KG
from torch_geometric.sampler import NegativeSampling
from torch_geometric.datasets import IndRelLinkPredDataset, WordNet18

separator = ">" * 30
line = "-" * 30

def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match

def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)

def strict_negative_mask(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # Part I: Sample hard negative tails
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    query_index = torch.stack([pos_h_index, pos_r_index])
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    t_truth_index = data.edge_index[1, edge_id]
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    t_mask.scatter_(1, t_truth_index.unsqueeze(-1), 0)

    # Part II: Sample hard negative heads
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    query_index = torch.stack([pos_t_index, pos_r_index])
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    h_truth_index = data.edge_index[0, edge_id]
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    h_mask.scatter_(1, h_truth_index.unsqueeze(-1), 0)

    return t_mask, h_mask

def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index)
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index)
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch

def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking

def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None):
    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    train_loader = DataLoader(train_triplets, batch_size=cfg['train']['batch_size'], shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=cfg['optimizer']['lr'])

    best_result = float("-inf")
    best_epoch = -1
    
    # Initialize NegativeSampling with triplet mode
    neg_sampler = NegativeSampling(mode="triplet", amount=cfg['task']['num_negative'])

    for epoch in range(cfg['train']['num_epoch']):
        model.train()
        losses = []
        for batch in train_loader:
            batch = negative_sampling(train_data, batch, cfg['task']['num_negative'],
                                                strict=cfg['task']['strict_negative'])
            pred = model(train_data, batch)

            target = torch.zeros_like(pred)
            target[:, 0] = 1

            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

            neg_weight = torch.ones_like(pred)
            if cfg['task']['adversarial_temperature'] > 0:
                with torch.no_grad():
                    neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg['task']['adversarial_temperature'], dim=-1)
            else:
                neg_weight[:, 1:] = 1 / cfg['task']['num_negative']

            loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch + 1}/{cfg['train']['num_epoch']}, Loss: {avg_loss:.4f}")

        result = test(cfg, model, valid_data, filtered_data)
        if result > best_result:
            best_result = result
            best_epoch = epoch
            os.makedirs("model_checkpoints", exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }, "model_checkpoints/best_checkpoint.pth")

    print(f"Loading best checkpoint from epoch {best_epoch + 1}")
    checkpoint = torch.load("model_checkpoints/best_checkpoint.pth")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None):
    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    test_loader = DataLoader(test_triplets, batch_size=cfg['train']['batch_size'], shuffle=False)

    model.eval()
    rankings = []
    num_negatives = []

    for batch in test_loader:
        t_batch, h_batch = all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = strict_negative_mask(filtered_data, batch)

        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)

    for metric in cfg['task']['metric']:
        if metric == "mr":
            score = ranking.float().mean()
        elif metric == "mrr":
            score = (1 / ranking.float()).mean()
        elif metric.startswith("hits@"):
            values = metric[5:].split("_")
            threshold = int(values[0])
            if len(values) > 1:
                num_sample = int(values[1])
                fp_rate = (ranking - 1).float() / num_negative
                score = 0
                for i in range(threshold):
                    num_comb = math.factorial(num_sample - 1) / \
                               math.factorial(i) / math.factorial(num_sample - i - 1)
                    score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                score = score.mean()
            else:
                score = (ranking <= threshold).float().mean()
        print(f"{metric}: {score:.4f}")

    mrr = (1 / ranking.float()).mean()
    return mrr

if __name__ == "__main__":
    cfg = {
        'dataset': {
            'class': 'IndWN18RR',
            'version': 'v1'
        },
        'model': {
            'class': 'NBFNet',
            'input_dim': 32,
            'hidden_dims': [32, 32, 32, 32, 32, 32],
            'message_func': 'distmult',
            'aggregate_func': 'DegreeScalerAggregation',
            'aggregate_kwargs': {
                "aggr": ["mean", "max", "min", "std"],
                "scaler": ["identity", "amplification", "attenuation"],
                "deg": None,
                "train_norm": False
            },
            'short_cut': True,
            'layer_norm': True,
            'dependent': False,
            'num_relation': None,
        },
        'task': {
            'num_negative': 32,
            'strict_negative': True,
            'adversarial_temperature': 1,
            'metric': ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10', 'hits@10_50']
        },
        'optimizer': {
            'class': 'Adam',
            'lr': 0.005
        },
        'train': {
            'gpus': [0],
            'batch_size': 64,
            'num_epoch': 20,
            'log_interval': 100
        }
    }

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', cfg['dataset']['class'])
    dataset = IndRelLinkPredDataset(path, name="WN18RR", version=cfg['dataset']['version'])
    cfg['model']['num_relation'] = dataset.num_relations
    
    deg = degree(dataset[0].edge_index[0], dataset[0].num_nodes).unsqueeze(0).unsqueeze(-1)
    deg = deg + 1
    cfg['model']['aggregate_kwargs']['deg'] = deg
    
    model = NBFNet_KG(input_dim=cfg['model']['input_dim'],
                   hidden_dims=cfg['model']['hidden_dims'],
                   num_relation=cfg['model']['num_relation'],
                   message_func=cfg['model']['message_func'],
                   aggregate_func=cfg['model']['aggregate_func'],
                   aggregate_kwargs=cfg['model']['aggregate_kwargs'],
                   short_cut=cfg['model']['short_cut'],
                   layer_norm=cfg['model']['layer_norm'],
                   dependent=cfg['model']['dependent'])

    torch.manual_seed(1024)

    print("Random seed: 1024")
    print("Config: Hardcoded in script")
    print(pprint.pformat(cfg))

    is_inductive = cfg['dataset']['class'].startswith("Ind")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    if is_inductive:
        filtered_data = None
    else:
        filtered_data = Data(edge_index=dataset.data.target_edge_index, edge_type=dataset.data.target_edge_type)
        filtered_data = filtered_data.to(device)

    train_and_validate(cfg, model, train_data, valid_data, filtered_data=filtered_data)
    print(separator)
    print("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=filtered_data)
    print(separator)
    print("Evaluate on test")
    test(cfg, model, test_data, filtered_data=filtered_data)
    
    
#     # Transductive Training and Testing on (existing) WordNet18 dataset
#     print(separator)
#     print("Starting Transductive Training and Testing with WordNet18")
#     cfg['dataset']['class'] = 'WN18'
#     path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', cfg['dataset']['class'])
#     dataset = WordNet18(path)
#     cfg['model']['num_relation'] = dataset[0].edge_type.max().item() + 1

#     deg = degree(dataset[0].edge_index[0], dataset[0].num_nodes).unsqueeze(0).unsqueeze(-1)
#     deg = deg + 1
#     cfg['model']['aggregate_kwargs']['deg'] = deg

#     model = NBFNet(input_dim=cfg['model']['input_dim'],
#                    hidden_dims=cfg['model']['hidden_dims'],
#                    num_relation=cfg['model']['num_relation'],
#                    message_func=cfg['model']['message_func'],
#                    aggregate_func=cfg['model']['aggregate_func'],
#                    aggregate_kwargs=cfg['model']['aggregate_kwargs'],
#                    short_cut=cfg['model']['short_cut'],
#                    layer_norm=cfg['model']['layer_norm'],
#                    dependent=cfg['model']['dependent'])

#     model = model.to(device)
#     train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
#     train_data = train_data.to(device)
#     valid_data = valid_data.to(device)
#     test_data = test_data.to(device)

#     filtered_data = Data(edge_index=dataset[0].edge_index, edge_type=dataset[0].edge_type)
#     filtered_data = filtered_data.to(device)

#     train_and_validate(cfg, model, train_data, valid_data, filtered_data=filtered_data)
#     print(separator)
#     print("Evaluate on valid")
#     test(cfg, model, valid_data, filtered_data=filtered_data)
#     print(separator)
#     print("Evaluate on test")
#     test(cfg, model, test_data, filtered_data=filtered_data)