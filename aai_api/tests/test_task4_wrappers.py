import torch
import torch.nn as nn
import pytest
from task2_3_4.task4_xai.aai_explainer import ClassifierWrapper, QualityWrapper, QualityTarget

# Simple Mock Model that returns two things: (Logits, QualityScores)
class SimpleDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5) # dummy layers

    def forward(self, x):
        # Returns a tuple simulating Task 2 model
        # Logits: [Batch, 2] | Quality: [Batch, 3]
        logits = torch.tensor([[1.0, -1.0]]) 
        quality = torch.tensor([[0.8, 0.5, 0.2]]) # Colour, Size, Ripeness
        return logits, quality

def test_classifier_wrapper():
    """Ensures the wrapper only extracts the first element (logits)."""
    model = SimpleDummyModel()
    wrapper = ClassifierWrapper(model)
    
    dummy_input = torch.randn(1, 10)
    output = wrapper(dummy_input)
    
    assert output.shape == (1, 2)
    assert torch.equal(output, torch.tensor([[1.0, -1.0]]))

def test_quality_wrapper():
    """Ensures the wrapper only extracts the second element (quality scores)."""
    model = SimpleDummyModel()
    wrapper = QualityWrapper(model)
    
    dummy_input = torch.randn(1, 10)
    output = wrapper(dummy_input)
    
    assert output.shape == (1, 3)
    assert torch.equal(output, torch.tensor([[0.8, 0.5, 0.2]]))

def test_quality_target_indexing():
    """Tests if QualityTarget correctly picks specific indices for Grad-CAM."""
    # Simulate a quality score output: [Colour=0.9, Size=0.4, Ripeness=0.1]
    mock_output = torch.tensor([0.9, 0.4, 0.1])
    
    # Test Colour (Index 0)
    target_colour = QualityTarget(target_index=0)
    assert target_colour(mock_output).item() == pytest.approx(0.9)
    
    # Test Ripeness (Index 2)
    target_ripeness = QualityTarget(target_index=2)
    assert target_ripeness(mock_output).item() == pytest.approx(0.1)

def test_quality_target_batch_handling():
    """Tests if QualityTarget handles batched tensors correctly."""
    # [Batch of 2, 3 scores]
    mock_batch_output = torch.tensor([
        [0.9, 0.4, 0.1],
        [0.7, 0.3, 0.5]
    ])
    
    target_size = QualityTarget(target_index=1)
    result = target_size(mock_batch_output)
    
    assert result.shape == (2,)
    assert result[0].item() == pytest.approx(0.4)
    assert result[1].item() == pytest.approx(0.3)