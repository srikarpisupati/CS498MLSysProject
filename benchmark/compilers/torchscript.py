import torch
import torch.nn as nn
from .base import Compiler

class TorchScriptCompiler(Compiler):
    def __init__(self, method="trace"):
        self.method = method
    
    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        model.eval()
        
        if self.method == "trace":
            traced_model = torch.jit.trace(model, example_input, check_trace=False)
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
        return False




