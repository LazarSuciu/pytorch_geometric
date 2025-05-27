import pytest
import torch
from torch_geometric.nn import GeneralizedRelationalConv
from torch_geometric.testing import is_full_test
from torch_geometric.utils._degree import degree


@pytest.mark.parametrize('message_func', ['transe', 'distmult', 'rotate'])
@pytest.mark.parametrize('input_dim', [16, 32])
@pytest.mark.parametrize('output_dim', [32, 64])
@pytest.mark.parametrize('dependent', [True, False])
@pytest.mark.parametrize('edge_weight', [None, torch.rand(20)])
def test_generalized_relational_conv(message_func, input_dim, output_dim, dependent, edge_weight):
    num_nodes = 10
    num_edges = 20
    num_relation = 4
    batch_size = 2
    query_input_dim = input_dim

    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relation, (num_edges,))
    x = torch.rand(batch_size, num_nodes, input_dim) 
    query = torch.rand(batch_size, query_input_dim)  
    boundary = torch.zeros(batch_size, num_nodes, query_input_dim)  
    
    aggr = 'DegreeScalerAggregation'
    aggr_kwargs = {
        "aggr": ["mean", "max", "min", "std"],
        "scaler": ["identity", "amplification", "attenuation"],
        "deg": degree(edge_index[0], num_nodes).unsqueeze(0).unsqueeze(-1),
        "train_norm": False
    }

    conv = GeneralizedRelationalConv(
        input_dim=input_dim,
        output_dim=output_dim,
        num_relation=num_relation,
        query_input_dim=input_dim,
        message_func=message_func,
        aggr=aggr,
        aggr_kwargs=aggr_kwargs if aggr == "DegreeScalerAggregation" else None,
        dependent=dependent,
    )

    out = conv(
        x=x,
        query=query,
        boundary=boundary,
        edge_index=edge_index,
        edge_type=edge_type,
        size=(num_nodes, num_nodes),
        edge_weight=edge_weight,
    )
    assert out.size() == (batch_size, num_nodes, output_dim)

    assert torch.allclose(
        conv(
            x=x,
            query=query,
            boundary=boundary,
            edge_index=edge_index,
            edge_type=edge_type,
            size=(num_nodes, num_nodes),
            edge_weight=edge_weight,
        ),
        out,
        atol=1e-6,
    )

    if is_full_test():
        jit = torch.jit.script(conv)
        out_jit = jit(
            x=x,
            query=query,
            boundary=boundary,
            edge_index=edge_index,
            edge_type=edge_type,
            size=(num_nodes, num_nodes),
            edge_weight=edge_weight,
        )
        assert torch.allclose(out, out_jit, atol=1e-6)

    x.requires_grad_()
    query.requires_grad_()
    out = conv(
        x=x,
        query=query,
        boundary=boundary,
        edge_index=edge_index,
        edge_type=edge_type,
        size=(num_nodes, num_nodes),
        edge_weight=edge_weight,
    )
    out.mean().backward()

    assert x.grad is not None
    if dependent:
        assert query.grad is not None
    for param in conv.parameters():
        assert param.grad is not None