import torch
from torch import Tensor, nn, spmm
from torch.nn import functional as F
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops


class GeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_relation,
                 query_input_dim,
                 message_func="distmult",
                 aggregate_func="DegreeScalerAggregation",
                 aggregate_kwargs=None,
                 layer_norm=False,
                 activation="relu",
                 dependent=True,
                 **kwargs):
        
        if aggregate_func != "DegreeScalerAggregation":
            aggregate_kwargs = {}

        super().__init__(aggr=aggregate_func, aggr_kwargs=aggregate_kwargs, **kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if self.aggregate_func == "DegreeScalerAggregation":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            # relation embeddings as an independent embedding matrix per each layer
            self.relation = nn.Embedding(num_relation, input_dim)
              
    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None):
        batch_size = len(query)

        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            # layer-specific relation features as a special embedding matrix unique to each layer
            relation = self.relation.weight.expand(batch_size, -1, -1)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)

        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        output = super().propagate(edge_index, size, input=input, relation=relation, boundary=boundary,
                                edge_type=edge_type, edge_weight=edge_weight)
        return output

    def message(self, input_j, relation, boundary, edge_type, edge_index, edge_weight):
        relation_j = relation.index_select(self.node_dim, edge_type)
        
        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the boundary condition
        message = torch.cat([message, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)

        # add self loops to edge index and edge weight, reshape edge weight to match the input shape
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=boundary.shape[1])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)
        
        message = message * edge_weight

        return message

    # useful for large graphs, will not be able to use custom message and aggregate functions if implemented
    # def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
    #     if isinstance(adj_t, SparseTensor):
    #         adj_t = adj_t.set_value(None, layout=None)
    #     return spmm(adj_t, x[0], reduce=self.aggr)

    def update(self, update, input):
        # node update as a function of old states (input) and this layer output (update)
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output