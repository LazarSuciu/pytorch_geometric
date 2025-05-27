import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.datasets import RelLinkPredDataset
from torch_geometric.data import Data
from torch_geometric.nn.models import NBFNet
from torch_geometric.nn.models.nbfnet import edge_match
from tqdm import tqdm

def negative_sampling(data, batch, num_negative):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

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
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    query_index = torch.stack([pos_h_index, pos_r_index])
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    query_index = torch.stack([pos_t_index, pos_r_index])
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)
    return t_mask, h_mask

def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing='ij')
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing='ij')
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    return t_batch, h_batch

def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking

def train(model, data, batch_size, num_negative, optimizer):
    model.train()
    triplets = torch.cat([data.target_edge_index, data.target_edge_type.unsqueeze(0)]).t()
    loader = DataLoader(triplets, batch_size=batch_size, shuffle=True)
    total_loss = total_examples = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        batch = negative_sampling(data, batch, num_negative)
        pred = model(data, batch)
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.size(0)
        total_examples += batch.size(0)
    avg_loss = total_loss / total_examples
    print(f"  [train] Avg loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def test(model, data, batch_size, filtered_data=None, split="valid"):
    triplets = torch.cat([data.target_edge_index, data.target_edge_type.unsqueeze(0)]).t()
    loader = DataLoader(triplets, batch_size=batch_size)
    model.eval()
    rankings = []
    for batch in tqdm(loader, desc=f"Evaluating ({split})", leave=False):
        t_batch, h_batch = all_negative(data, batch)
        t_pred = model(data, t_batch)
        h_pred = model(data, h_batch)
        if filtered_data is None:
            t_mask, h_mask = strict_negative_mask(data, batch)
        else:
            t_mask, h_mask = strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, _ = batch.t()
        t_ranking = compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = compute_ranking(h_pred, pos_h_index, h_mask)
        rankings += [t_ranking, h_ranking]
    ranking = torch.cat(rankings)
    mrr = (1 / ranking.float()).mean()
    print(f"  [{split}] MRR: {mrr:.4f}")
    return mrr.item()

if __name__ == "__main__":
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'FB15k-237')
    dataset = RelLinkPredDataset(path, name='FB15k-237')
    train_data = Data(edge_index=dataset.edge_index, edge_type=dataset.edge_type, num_nodes=dataset.num_nodes,
                      target_edge_index=dataset.train_edge_index, target_edge_type=dataset.train_edge_type)
    valid_data = Data(edge_index=dataset.edge_index, edge_type=dataset.edge_type, num_nodes=dataset.num_nodes,
                      target_edge_index=dataset.valid_edge_index, target_edge_type=dataset.valid_edge_type)
    test_data = Data(edge_index=dataset.edge_index, edge_type=dataset.edge_type, num_nodes=dataset.num_nodes,
                     target_edge_index=dataset.test_edge_index, target_edge_type=dataset.test_edge_type)
    num_relations = dataset.num_relations

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NBFNet(
        input_dim=32,
        hidden_dims=[32, 32, 32],
        num_relation=num_relations,
        message_func='distmult',
        aggregate_func='mean',
        short_cut=True,
        layer_norm=True,
        dependent=True,
    ).to(device)
    train_data, valid_data, test_data = train_data.to(device), valid_data.to(device), test_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    batch_size = 32
    num_negative = 32

    best_val_mrr = 0
    best_epoch = 0

    for epoch in range(1, 2):
        print(f"Epoch: {epoch}")
        loss = train(model, train_data, batch_size, num_negative, optimizer)
        val_mrr = test(model, valid_data, batch_size, split="valid")
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch
        print(f"  [progress] Best Val MRR: {best_val_mrr:.4f} at epoch {best_epoch}")

    test_mrr = test(model, test_data, batch_size, split="test")
    print(f"Final Test MRR: {test_mrr:.4f}")