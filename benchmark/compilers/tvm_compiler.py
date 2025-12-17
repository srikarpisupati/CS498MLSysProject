import shutil
import warnings

import torch
import torch.nn as nn
from torch.utils import dlpack

from .base import Compiler


class TVMCompiler(Compiler):

    def __init__(self, target: str | None = None, opt_level: int = 3):
        try:
            import tvm
            from tvm import relay
            from tvm.contrib import graph_executor
        except ImportError as exc:
            raise RuntimeError(
                "TVM is not installed. Install the bundled "
                "'tlcpack_cu116-0.11.1-*.whl' (in the repo root) or download a CUDA-enabled "
                "TLCPack release per https://tvm.apache.org/docs/install/index.html"
            ) from exc

        self._tvm = tvm
        self._relay = relay
        self._graph_executor = graph_executor
        self._tvm_cuda_enabled = bool(tvm.runtime.enabled("cuda")) and bool(
            tvm.get_global_func("target.build.cuda", allow_missing=True)
        )
        self._nvcc_available = shutil.which("nvcc") is not None

        default_target = "cuda" if torch.cuda.is_available() and self._tvm_cuda_enabled else "llvm"
        requested_target = target or default_target
        self.target, self._tvm_target = self._resolve_target(requested_target)
        self.opt_level = opt_level
        self.input_name = "input0"

    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        model.eval()
        model_cpu = model.to("cpu")
        example_cpu = example_input.detach().to("cpu")

        traced = torch.jit.trace(model_cpu, example_cpu)

        shape_list = [(self.input_name, tuple(example_cpu.shape))]
        relay_mod, params = self._relay.frontend.from_pytorch(traced, shape_list)

        with self._tvm.transform.PassContext(opt_level=self.opt_level):
            lib = self._relay.build(relay_mod, target=self._tvm_target, params=params)

        tvm_device = self._get_tvm_device()
        graph_mod = self._graph_executor.GraphModule(lib["default"](tvm_device))

        return _TVMCompiledModule(
            graph_module=graph_mod,
            tvm_module=self._tvm,
            target=self.target,
            tvm_device=tvm_device,
            input_name=self.input_name,
        )

    def get_name(self) -> str:
        return f"tvm_{self.target}"

    def supports_dynamic_shapes(self) -> bool:
        return False

    def _get_tvm_device(self):
        if "cuda" in self.target:
            if not self._tvm.runtime.enabled("cuda"):
                raise RuntimeError(
                    "TVM was not built with CUDA support but 'cuda' target was requested."
                )
            return self._tvm.cuda(0)
        return self._tvm.cpu(0)

    def _resolve_target(self, requested_target: str):
        target_str = requested_target
        if "cuda" in target_str:
            if not torch.cuda.is_available():
                warnings.warn(
                    "TVM compiler requested CUDA target but PyTorch cannot see a CUDA device. "
                    "Falling back to 'llvm'.",
                    RuntimeWarning,
                )
                return "llvm", self._tvm.target.Target("llvm")

            if not self._tvm_cuda_enabled:
                warnings.warn(
                    "TVM was installed without CUDA codegen support. Falling back to 'llvm'. "
                    "Reinstall TVM with the CUDA-enabled TLCPack wheel to target GPUs.",
                    RuntimeWarning,
                )
                return "llvm", self._tvm.target.Target("llvm")

            if not self._nvcc_available:
                warnings.warn(
                    "NVCC compiler not found in PATH. Falling back to 'llvm'. "
                    "Install CUDA toolkit to enable GPU builds.",
                    RuntimeWarning,
                )
                return "llvm", self._tvm.target.Target("llvm")

            if "-arch" not in target_str:
                arch = self._detect_cuda_arch()
                target_str = f"cuda -arch={arch}"

            return target_str, self._tvm.target.Target(target_str)

        return target_str, self._tvm.target.Target(target_str)

    def _detect_cuda_arch(self) -> str:
        try:
            device_props = torch.cuda.get_device_properties(0)
            sm = f"sm_{device_props.major}{device_props.minor}"
            return sm
        except Exception:
            return "sm_80"


class _TVMCompiledModule(nn.Module):

    def __init__(self, graph_module, tvm_module, target: str, tvm_device, input_name: str):
        super().__init__()
        self.graph_module = graph_module
        self._tvm = tvm_module
        self.target = target
        self.tvm_device = tvm_device
        self.input_name = input_name
        self.uses_cuda = "cuda" in target

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tvm_input = self._to_tvm_ndarray(inputs)
        self.graph_module.set_input(self.input_name, tvm_input)
        self.graph_module.run()
        self.tvm_device.sync()
        output = self.graph_module.get_output(0)
        return self._to_torch_tensor(output, inputs.device)

    def _to_tvm_ndarray(self, tensor: torch.Tensor):
        if self.uses_cuda:
            if tensor.device.type != "cuda":
                tensor = tensor.to("cuda")
            tensor_copy = tensor.detach().clone().contiguous()
            return self._tvm.nd.from_dlpack(dlpack.to_dlpack(tensor_copy))

        if tensor.device.type != "cpu":
            tensor = tensor.to("cpu")
        numpy_array = tensor.detach().cpu().numpy()
        return self._tvm.nd.array(numpy_array, device=self.tvm_device)

    def _to_torch_tensor(self, tvm_output, desired_device: torch.device) -> torch.Tensor:
        if self.uses_cuda:
            torch_tensor = dlpack.from_dlpack(tvm_output.to_dlpack())
            return torch_tensor.to(desired_device)

        torch_tensor = torch.from_numpy(tvm_output.numpy())
        return torch_tensor.to(desired_device)

