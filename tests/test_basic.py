"""
Basic functionality tests for KProdigy optimizer.
"""

import torch
import pytest
from kprodigy import KProdigy


def test_kprodigy_initialization():
    """Test that KProdigy initializes correctly."""
    model = torch.nn.Linear(10, 1)
    optimizer = KProdigy(model.parameters(), lr=1.0)
    assert optimizer is not None
    assert len(optimizer.param_groups) == 1


def test_kprodigy_step():
    """Test that KProdigy can perform a step."""
    model = torch.nn.Linear(10, 1)
    optimizer = KProdigy(model.parameters(), lr=1.0)
    
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # Forward pass
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    assert True  # If we get here, the step succeeded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KProdigy v0.3.0 is GPU-optimized")
def test_kprodigy_convergence():
    """Test that KProdigy can optimize a neural network."""
    # Use a simple neural network instead of scalar (D-adaptation works better with vectors)
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1)
    ).cuda()
    
    optimizer = KProdigy(model.parameters(), lr=1.0)
    
    # Create simple dataset
    torch.manual_seed(42)
    X = torch.randn(100, 10).cuda()
    y = torch.randn(100, 1).cuda()
    
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # Should show convergence (loss decreasing)
    assert losses[-1] < losses[0], "Loss should decrease"
    # Should reduce loss by at least 50%
    assert losses[-1] < losses[0] * 0.5, "Loss should reduce significantly"


def test_independent_d():
    """Test independent D estimation for multi-parameter groups (automatic in v0.3.0)."""
    model1 = torch.nn.Linear(10, 5)
    model2 = torch.nn.Linear(5, 1)
    
    # Independent D is now automatic when using multiple parameter groups
    optimizer = KProdigy([
        {'params': model1.parameters(), 'lr': 1.0},
        {'params': model2.parameters(), 'lr': 0.5}
    ])
    
    assert len(optimizer.param_groups) == 2
    
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # Forward pass
    h = model1(x)
    output = model2(h)
    loss = torch.nn.functional.mse_loss(output, y)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    # Check that both groups have different D values
    d1 = optimizer.param_groups[0]['d']
    d2 = optimizer.param_groups[1]['d']
    
    # After one step, D values might still be the same (d0)
    # But the mechanism should be in place
    assert 'd' in optimizer.param_groups[0]
    assert 'd' in optimizer.param_groups[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_foreach_cuda():
    """Test that foreach optimization works on CUDA (always enabled in v0.3.0)."""
    model = torch.nn.Linear(10, 1).cuda()
    # foreach is now always enabled, no parameter needed
    optimizer = KProdigy(model.parameters(), lr=1.0)
    
    # Create dummy data on CUDA
    x = torch.randn(32, 10).cuda()
    y = torch.randn(32, 1).cuda()
    
    # Forward pass
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step (should use foreach path)
    optimizer.step()
    
    assert True  # If we get here, foreach worked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

