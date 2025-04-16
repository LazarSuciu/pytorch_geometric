import pytest
import torch
from torch_geometric.nn import GeneralizedRelationalConv
from torch_geometric.utils import degree, add_self_loops


@pytest.mark.parametrize('message_func', ['transe', 'distmult', 'rotate'])
@pytest.mark.parametrize('aggregate_func', ['mean', 'sum', 'DegreeScalerAggregation'])
def test_generalized_relational_conv_forward(message_func, aggregate_func):
    x = torch.randn(4, 8)
    query = torch.randn(4, 8)
    boundary = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_type = torch.tensor([0, 1, 2, 3])
    edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
    aggregate_kwargs = None
    
    if aggregate_func == 'DegreeScalerAggregation':
        
        # Compute degree histogram
        num_nodes = x.size(0)
        deg = degree(edge_index[1], num_nodes=num_nodes)
        
        aggregate_kwargs = {
            "aggr": ["mean", "max", "min", "std"],
            "scaler": ["identity", "amplification", "attenuation"],
            "deg": deg,  # Provide degree tensor for normalization
            "train_norm": False
        }

    conv = GeneralizedRelationalConv(
        input_dim=8,
        output_dim=16,
        num_relation=4,
        query_input_dim=8,
        message_func=message_func,
        aggregate_func=aggregate_func,
        aggregate_kwargs=aggregate_kwargs
    )

    out = conv(x, query, boundary, edge_index, edge_type, size=(4, 4), edge_weight=edge_weight)
    assert out.size() == (4, 16)
    
    assert conv.input_dim == 8
    assert conv.output_dim == 16
    assert conv.num_relation == 4
    assert conv.query_input_dim == 8
    assert conv.message_func == message_func
    assert conv.aggregate_func == aggregate_func
    assert conv.activation == torch.nn.ReLU()

@pytest.mark.parametrize('message_func', ['transe', 'distmult', 'rotate'])
@pytest.mark.parametrize('aggregate_func', ['mean', 'sum', 'DegreeScalerAggregation'])
def test_generalized_relational_conv_no_edge_weights(message_func, aggregate_func):
    x = torch.randn(4, 8)
    query = torch.randn(4, 8)
    boundary = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_type = torch.tensor([0, 1, 2, 3])
    aggregate_kwargs = None
    
    if aggregate_func == 'DegreeScalerAggregation':
        
        # Compute degree histogram
        num_nodes = x.size(0)
        deg = degree(edge_index[1], num_nodes=num_nodes)
        
        aggregate_kwargs = {
            "aggr": ["mean", "max", "min", "std"],
            "scaler": ["identity", "amplification", "attenuation"],
            "deg": deg,  # Provide degree tensor for normalization
            "train_norm": False
        }

    conv = GeneralizedRelationalConv(
        input_dim=8,
        output_dim=16,
        num_relation=4,
        query_input_dim=8,
        message_func=message_func,
        aggregate_func=aggregate_func,
        aggregate_kwargs=aggregate_kwargs
    )

    out = conv(x, query, boundary, edge_index, edge_type, size=(4, 4))
    assert out.size() == (4, 16)