"""
Tests for model.py
usage: pytest test/test_model.py
"""

import rootutils

ROOT = rootutils.autosetup()

import pytest
import torch
from src.model import MultibinRegressor


@pytest.fixture
def mock_data():
    return torch.randn(2, 3, 224, 224)  # Assuming input size is 224x224 RGB images


def test_initialization():
    # Test initialization with supported backbones and n_bins
    try:
        MultibinRegressor(backbone="resnet18", n_bins=2)
        MultibinRegressor(backbone="mobilenetv3-small", n_bins=3)
    except Exception as e:
        pytest.fail(f"Initialization failed with exception {e}")

    # Test initialization with unsupported backbone
    with pytest.raises(AssertionError):
        MultibinRegressor(backbone="unsupported_backbone", n_bins=2)

    # Test initialization with invalid n_bins
    with pytest.raises(AssertionError):
        MultibinRegressor(backbone="resnet18", n_bins=1)


def test_forward_pass(mock_data):
    # Initialize model
    model = MultibinRegressor(backbone="resnet18", n_bins=2)
    # Perform forward pass
    ori, conf, dim = model(mock_data)
    # Check shapes
    assert ori.shape == (2, 2, 2)  # (batch_size, n_bins, 2)
    assert conf.shape == (2, 2)  # (batch_size, n_bins)
    assert dim.shape == (2, 3)  # (batch_size, 3)
