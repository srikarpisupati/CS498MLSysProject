import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .base import ModelWrapper


class _Gpt2Module(nn.Module):
    """Wrap GPT-2 so we always return logits tensor compatible with benchmarks."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids, return_dict=False)
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return getattr(outputs, "logits", outputs)


class Gpt2Wrapper(ModelWrapper):
    """Model wrapper for GPT-2 causal LM."""

    def __init__(self, seq_length: int = 128, pretrained: bool = True):
        config = AutoConfig.from_pretrained("gpt2")
        if pretrained:
            self.base_model = AutoModelForCausalLM.from_pretrained("gpt2", config=config)
        else:
            self.base_model = AutoModelForCausalLM.from_config(config)

        self.base_model.eval()
        self.model = _Gpt2Module(self.base_model)
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
        return "gpt2"

