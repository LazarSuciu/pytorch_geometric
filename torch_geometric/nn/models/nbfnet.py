from collections.abc import Sequence
import torch
from torch import nn, Tensor
from torch_geometric.utils import index_to_mask
from torch_geometric.sampler import NegativeSampling
from torch_geometric.nn import GeneralizedRelationalConv
from torch_geometric.data import Data
from typing import Sequence, Tuple, Optional


class NBFNet(nn.Module):
    r"""The Neural Bellman-Ford Network (NBFNet) model for link prediction tasks
    from the `"Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction"
    <https://arxiv.org/abs/2106.06935>`_ paper.

    .. note::
        For an example of using NBFNet, see `examples/nbfnet_kg.py`.

    Args:
        input_dim (int): The dimensionality of input node features.
        hidden_dims (list[int]): A list of integers specifying the dimensions
            of the hidden layers.
        num_relation (int): The number of relations in the knowledge graph.
        message_func (str, optional): The message-passing function to use.
            Options include :obj:`"distmult"`, :obj:`"rotate"` and :obj:`"transe"`.
            (default: :obj:`"distmult"`)
        aggregate_func (str, optional): The aggregation function to use.
            (default: :obj:`"mean"`)
        short_cut (bool, optional): If set to :obj:`True`, enables residual
            connections between GNN layers. (default: :obj:`False`)
        layer_norm (bool, optional): If set to :obj:`True`, applies layer
            normalization to the outputs of each GNN layer. (default: :obj:`False`)
        activation (str, optional): The activation function to apply after
            each layer. (default: :obj:`"relu"`)
        concat_hidden (bool, optional): If set to :obj:`True`, concatenates
            the outputs of all GNN layers. Otherwise, only the final layer's
            output is used. (default: :obj:`False`)
        num_mlp_layer (int, optional): The number of layers in the final MLP
            used for scoring. (default: :obj:`2`)
        dependent (bool, optional): If set to :obj:`True`, uses dependent
            relation embeddings that are computed as projections of query embeddings.
            Otherwise, uses independent relation embeddings. (default: :obj:`True`)
        remove_one_hop (bool, optional): If set to :obj:`True`, dynamically
            removes one-hop edges during training to prevent information leakage.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        num_relation: int,
        message_func: str = "distmult",
        aggregate_func: str = "mean",
        short_cut: bool = False,
        layer_norm: bool = False,
        activation: str = "relu",
        concat_hidden: bool = False,
        num_mlp_layer: int = 2,
        dependent: bool = True,
        remove_one_hop: bool = False,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop

        self.layers = nn.ModuleList([
            GeneralizedRelationalConv(
                self.dims[i], self.dims[i + 1], num_relation, self.dims[0],
                message_func, aggregate_func, layer_norm, activation, dependent, **kwargs
            )
            for i in range(len(self.dims) - 1)
        ])

        feature_dim = (sum(hidden_dims) if concat_hidden else hidden_dims[-1]) + input_dim
        self.query = nn.Embedding(num_relation, input_dim)
        mlp_layers = []
        for _ in range(num_mlp_layer - 1):
            mlp_layers.append(nn.Linear(feature_dim, feature_dim))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def reset_parameters(self) -> None:
        self.query.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for m in self.mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def remove_easy_edges(
        self,
        data: Data,
        h_index: Tensor,
        t_index: Tensor,
        r_index: Optional[Tensor] = None
    ) -> Data:
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        r_index_ext = torch.cat([r_index, r_index + self.num_relation // 2], dim=-1)
        if self.remove_one_hop:
            edge_index = data.edge_index
            easy_edge = torch.stack([h_index_ext, t_index_ext]).flatten(1)
            index = edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)
        else:
            edge_index = torch.cat([data.edge_index, data.edge_type.unsqueeze(0)])
            easy_edge = torch.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
            index = edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)
        data = data.clone()
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        return data

    def negative_sample_to_tail(
        self,
        h_index: Tensor,
        t_index: Tensor,
        r_index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation // 2)
        return new_h_index, new_t_index, new_r_index

    def bellmanford(
        self,
        data: Data,
        h_index: Tensor,
        r_index: Tensor,
        separate_grad: bool = False
    ) -> dict:
        batch_size = len(r_index)
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)
        hiddens, edge_weights = [], []
        layer_input = boundary
        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)
        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(
        self,
        data: Data,
        batch: Tensor,
    ) -> Tensor:
        orig_shape = batch.shape

        if batch.dim() == 2:
            batch = batch.unsqueeze(1)
        elif batch.dim() != 3:
            raise ValueError("Batch must be of shape [N, 3] or [B, K, 3]")

        h_index, t_index, r_index = batch.unbind(-1) 

        if len(orig_shape) == 3:
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)

        if self.training:
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)
        score = self.mlp(feature).squeeze(-1)

        return score.view(orig_shape[:-1])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'input_dim={self.dims[0]}, '
                f'hidden_dims={self.dims[1:]}, '
                f'num_relation={self.num_relation})')
        
def edge_match(
    edge_index: Tensor,
    query_index: Tensor
) -> Tuple[Tensor, Tensor]:
    base = edge_index.max(dim=1)[0] + 1
    scale = base.cumprod(0)
    scale = scale[-1] // scale
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    num_match = end - start
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)
    return order[range], num_match