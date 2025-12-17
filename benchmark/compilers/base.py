from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class Compiler(ABC):
    
    @abstractmethod
    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    def supports_dynamic_shapes(self) -> bool:
        return False