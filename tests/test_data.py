from dtu_mlops_git_own.data import MyDataset
from torch.utils.data import Dataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)
