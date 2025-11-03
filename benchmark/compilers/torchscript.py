import torch
import torch.nn as nn
from .base import Compiler

class TorchScriptCompiler(Compiler):
    # uses torch.jit to get a static-ish graph
    def __init__(self, method="trace"):
        self.method = method
    
    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """Produce a TorchScript module via trace or script for inference.

        We default to trace (fast, assumes input shapes/paths are stable).
        Script is slower but can handle more dynamic Python. Both paths
        run optimize_for_inference before returning.
        """
        model.eval()
        
        if self.method == "trace":
            traced_model = torch.jit.trace(model, example_input)
            traced_model = torch.jit.optimize_for_inference(traced_model)
            return traced_model
        
        elif self.method == "script":
            scripted_model = torch.jit.script(model)
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
            return scripted_model
        
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'trace' or 'script'")
    
    def get_name(self) -> str:
        return f"torchscript_{self.method}"
    
    def supports_dynamic_shapes(self) -> bool:
        # dynamic shapes are finicky here; not relying on them
        return False

