from dtu_mlops_git_own.data import corrupt_mnist

import torch

# def test_my_dataset():
#     """Test the MyDataset class."""
#     dataset = MyDataset("data/raw")
#     assert isinstance(dataset, Dataset)

def test_data():
    train_dataset, test_dataset = corrupt_mnist()
    assert len(train_dataset) == 30000
    assert len(test_dataset) == 5000
    
    for dataset in [train_dataset, test_dataset]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)

    train_targets = torch.unique(train_dataset.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test_dataset.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()