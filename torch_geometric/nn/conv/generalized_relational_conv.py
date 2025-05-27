import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.typing import Adj, Size
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops


class GeneralizedRelationalConv(MessagePassing):
    r"""The Neural Bellman-Ford convolutional operator from the
    `"Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction"
    <https://arxiv.org/abs/2106.06935>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \sigma\left( \text{LayerNorm} \left(
        \mathbf{W} \left[ \mathbf{x}_i \parallel \frac{1}{|\mathcal{N}(i)|}
        \sum_{j \in \mathcal{N}(i)} \phi(\mathbf{x}_j, \mathbf{r}_{(j \to i)})
        \right] \right) \right),

    where :math:`\phi(\mathbf{x}_j, \mathbf{r})` is a message function
    defined by a relational operator (e.g., TransE, DistMult, or RotatE),
    applied to the feature of neighbor node :math:`j` and the relation
    embedding :math:`\mathbf{r}` of edge :math:`(j \to i)`. This message is
    optionally concatenated with a boundary condition and scaled by edge weight.

    Relation embeddings :math:`\mathbf{r}` can either be projected from
    query embeddings (dependent mode), or learned independently
    (independent mode).

    Args:
        input_dim (int): Dimensionality of input node features.
        output_dim (int): Dimensionality of output node features.
        num_relation (int): Number of edge relation types.
        query_input_dim (int): Dimensionality of input query embeddings.
        message_func (str): Message function to use.
            Options: :obj:`"transe"`, :obj:`"distmult"`, :obj:`"rotate"`.
        aggr (str): Aggregation operator to use over incoming messages.
            (default: :obj:`mean`)
        layer_norm (bool, optional): Whether to apply layer normalization
            after aggregation. (default: :obj:`False`)
        activation (str or Callable, optional): Non-linear activation function
            to apply after aggregation. (default: :obj:`"relu"`)
        dependent (bool, optional): If :obj:`True`, relation embeddings are projected
            from query embeddings. Otherwise, an independent embedding is learned.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments for
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
            - Node features: :math:`(B, N, F_{in})`
            - Query embeddings: :math:`(B, F_{query})`
            - Boundary condition: :math:`(B, N, F_{in})`
            - Edge index: :math:`(2, |\mathcal{E}|)`
            - Edge types: :math:`(|\mathcal{E}|,)`
            - Edge weights: :math:`(|\mathcal{E}|,)` *(optional)*
        - **output:** node features :math:`(B, N, F_{out})`
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relation: int,
        query_input_dim: int,
        message_func: str = "distmult",
        aggr: str = "mean",
        layer_norm: bool = False,
        activation: str = "relu",
        dependent: bool = True,
        **kwargs
    ):
        super().__init__(aggr=aggr, node_dim=1, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.dependent = dependent

        if self.aggr == "DegreeScalerAggregation":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            self.relation = nn.Embedding(num_relation, input_dim)

    def forward(
        self,
        x: Tensor,
        query: Tensor,
        boundary: Tensor,
        edge_index: Adj,
        edge_type: Tensor,
        size: Size = None,
        edge_weight: Tensor = None,
    ) -> Tensor:
        batch_size = query.size(0)
        if self.dependent:
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            relation = self.relation.weight.expand(batch_size, -1, -1)
        if edge_weight is None:
            edge_weight = torch.ones(edge_type.size(0), device=x.device)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=boundary.size(1))
        shape = [1] * x.ndim
        shape[1] = -1  # node_dim=1
        edge_weight = edge_weight.view(shape)

        return self.propagate(
            edge_index,
            x=x,
            relation=relation,
            boundary=boundary,
            edge_type=edge_type,
            edge_weight=edge_weight,
            size=size,
        )

    def message(
        self,
        x_j: Tensor,
        relation: Tensor,
        boundary: Tensor,
        edge_type: Tensor,
        edge_weight: Tensor
    ) -> Tensor:
        num_nodes = boundary.size(1)
        x_j = x_j[:, :-num_nodes, :]
        relation_j = relation.index_select(1, edge_type)
        if self.message_func == "transe":
            message = x_j + relation_j
        elif self.message_func == "distmult":
            message = x_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = x_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError(f"Unknown message function `{self.message_func}`")
        message = torch.cat([message, boundary], dim=1)
        message = message * edge_weight
        return message

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        out = self.linear(torch.cat([x, aggr_out], dim=-1))
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out