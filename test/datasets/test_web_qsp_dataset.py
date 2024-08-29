from torch_geometric.datasets import WebQSPDataset
from torch_geometric.testing import onlyFullTest, onlyOnline
import pytest


@pytest.mark.skip(reason="Times out")
@onlyOnline
@onlyFullTest
def test_web_qsp_dataset():
    dataset = WebQSPDataset()
    assert len(dataset) == 4700
    assert str(dataset) == "WebQSPDataset(4700)"

@onlyOnline
@onlyFullTest
def test_web_qsp_dataset_limit():
    dataset = WebQSPDataset(limit=100)
    assert len(dataset) == 100
    assert str(dataset) == "WebQSPDataset(100)"

@onlyOnline
@onlyFullTest
def test_web_qsp_dataset_limit_no_pcst():
    dataset = WebQSPDataset(limit=100, include_pcst=False)
    assert len(dataset) == 100
    assert str(dataset) == "WebQSPDataset(100)"

def test_web_qsp_import_error():
    # WebQSP shouldn't run if dependencies don't exist locally
    try:
        dataset = WebQSPDataset()
        assert False # Test should fail if gets to here.
    except ImportError:
        pass