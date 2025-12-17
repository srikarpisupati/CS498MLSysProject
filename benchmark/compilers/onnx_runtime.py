import os
import tempfile
from typing import List

import torch
import torch.nn as nn

from .base import Compiler


class OnnxRuntimeCompiler(Compiler):

    def __init__(self, providers=None, opset_version: int = 17):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime-gpu is not installed") from exc

        self.providers = providers
        self.opset_version = opset_version
        self.input_name = "input"
        self.output_name = "output"

    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        import onnxruntime as ort

        model.eval()
        model_cpu = model.to("cpu")
        example_cpu = example_input.detach().to("cpu")

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name

        torch.onnx.export(
            model_cpu,
            example_cpu,
            onnx_path,
            opset_version=self.opset_version,
            input_names=[self.input_name],
            output_names=[self.output_name],
            dynamic_axes={
                self.input_name: {0: "batch"},
                self.output_name: {0: "batch"},
            },
            do_constant_folding=True,
        )

        providers = self.providers
        if providers is None:
            if torch.cuda.is_available():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        session = ort.InferenceSession(
            onnx_path,
            providers=providers,
            sess_options=session_options,
        )
        os.unlink(onnx_path)

        return _OnnxRuntimeModule(
            session=session,
            input_name=session.get_inputs()[0].name,
            output_names=[output.name for output in session.get_outputs()],
        )

    def get_name(self) -> str:
        return "onnxruntime"

    def supports_dynamic_shapes(self):
        return True


class _OnnxRuntimeModule(nn.Module):

    def __init__(self, session, input_name, output_names):
        super().__init__()
        self.session = session
        self.input_name = input_name
        self.output_names = output_names

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_np = inputs.detach().cpu().numpy()
        ort_inputs = {self.input_name: input_np}

        outputs = self.session.run(self.output_names, ort_inputs)
        torch_outputs = [torch.from_numpy(arr) for arr in outputs]
        
        if len(torch_outputs) == 1:
            return torch_outputs[0].to(inputs.device)
        return tuple(output.to(inputs.device) for output in torch_outputs)

