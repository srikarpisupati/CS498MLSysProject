import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .base import ModelWrapper


class _BertModule(nn.Module):
    """Wrap Hugging Face BertModel to always return the last hidden state tensor."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids, return_dict=False)
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return getattr(outputs, "last_hidden_state", outputs)


class BertWrapper(ModelWrapper):
    """Model wrapper for bert-base-uncased sequence representations."""

    def __init__(self, seq_length: int = 128, pretrained: bool = True):
        config = AutoConfig.from_pretrained("bert-base-uncased")
        if not pretrained:
            self.base_model = AutoModel.from_config(config)
        else:
            self.base_model = AutoModel.from_pretrained("bert-base-uncased", config=config)

        self.base_model.eval()
        self.model = _BertModule(self.base_model)
        self.seq_length = seq_length
        self.vocab_size = config.vocab_size

    def get_model(self) -> nn.Module:
        return self.model

    def get_example_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            0,
            self.vocab_size,
            (batch_size, self.seq_length),
            device=device,
            dtype=torch.long,
        )

    def get_name(self) -> str:
        return "bert_base_uncased"

