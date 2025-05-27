import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.nn.models.nbfnet import NBFNet

@pytest.fixture
def x():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_type = torch.tensor([0, 1, 2])
    num_nodes = 3
    return Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes)

@pytest.fixture
def sample_batch():
    # [N, 3] shape
    return torch.tensor([[0, 1, 0], [1, 2, 1]])

@pytest.fixture
def sample_batch_3d():
    # [B, K, 3] shape
    return torch.tensor([[[0, 1, 0], [0, 2, 1]], [[1, 2, 1], [1, 0, 2]]])

@pytest.mark.parametrize('short_cut', [True, False])
@pytest.mark.parametrize('concat_hidden', [True, False])
@pytest.mark.parametrize('remove_one_hop', [True, False])
def test_forward_2d(x, sample_batch, short_cut, concat_hidden, remove_one_hop):
    model = NBFNet(
        input_dim=16,
        hidden_dims=[16, 32],
        num_relation=3,
        message_func="distmult",
        aggregate_func="mean",
        short_cut=short_cut,
        layer_norm=True,
        concat_hidden=concat_hidden,
        num_mlp_layer=2,
        dependent=True,
        remove_one_hop=remove_one_hop,
    )
    batch = sample_batch
    output = model(x, batch)
    assert output is not None
    assert output.shape == (batch.size(0),)

@pytest.mark.parametrize('short_cut', [True, False])
@pytest.mark.parametrize('concat_hidden', [True, False])
@pytest.mark.parametrize('remove_one_hop', [True, False])
def test_forward_3d(x, sample_batch_3d, short_cut, concat_hidden, remove_one_hop):
    model = NBFNet(
        input_dim=16,
        hidden_dims=[16, 32],
        num_relation=3,
        message_func="distmult",
        aggregate_func="mean",
        short_cut=short_cut,
        layer_norm=True,
        concat_hidden=concat_hidden,
        num_mlp_layer=2,
        dependent=True,
        remove_one_hop=remove_one_hop,
    )
    batch = sample_batch_3d
    output = model(x, batch)
    assert output is not None
    assert output.shape == batch.shape[:-1]