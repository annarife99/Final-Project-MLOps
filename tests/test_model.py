import pytest
from transformers import BertConfig
from src.models.model import NLPModel
import torch
def test_shape_match():
    # Create sample config object
    config = BertConfig(num_labels=2)

    # Instantiate your model
    model = NLPModel(config)

    # Create sample input data with shape (batch_size, max_length)
    input_ids = torch.randint(0, config.vocab_size, (32, 128)).long()
    attention_mask = torch.ones((32, 128)).long()
    token_type_ids = torch.zeros((32, 128)).long()

    # Ensure the output shape matches the expected shape (batch_size, num_labels)
    assert model(input_ids, attention_mask, token_type_ids).logits.shape == (32, 2)
