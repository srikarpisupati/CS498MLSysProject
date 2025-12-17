import os
import torch
from benchmark.core.config import Config
from benchmark.core.benchmark_runner import BenchmarkRunner
from benchmark.models.resnet import ResNetWrapper
from benchmark.models.mobilenet import MobileNetWrapper
from benchmark.models.vgg import VGGWrapper
from benchmark.models.gpt2 import Gpt2Wrapper
from benchmark.compilers.pytorch_eager import PyTorchEagerCompiler
from benchmark.compilers.torch_inductor import TorchInductorCompiler
from benchmark.compilers.torchscript import TorchScriptCompiler
from benchmark.compilers.onnx_runtime import OnnxRuntimeCompiler
from benchmark.compilers.tvm_compiler import TVMCompiler
from benchmark.utils.device import get_device
from benchmark.utils.output import ResultsWriter

def get_compiler(compiler_name: str):
    if compiler_name == "pytorch_eager":
        return PyTorchEagerCompiler()
    elif compiler_name == "torch_inductor":
        return TorchInductorCompiler(mode="default")
    elif compiler_name == "torchscript" or compiler_name == "torchscript_trace":
        return TorchScriptCompiler(method="trace")
    elif compiler_name == "torchscript_script":
        return TorchScriptCompiler(method="script")
    elif compiler_name == "onnxruntime":
        return OnnxRuntimeCompiler()
    elif compiler_name == "tvm":
        return TVMCompiler()
    else:
        raise ValueError(f"Unknown compiler: {compiler_name}")

def get_model(model_name, input_shape):
    if model_name == "resnet50":
        return ResNetWrapper(input_shape=tuple(input_shape), pretrained=True)
    elif model_name == "mobilenet_v3":
        return MobileNetWrapper(input_shape=tuple(input_shape), pretrained=True)
    elif model_name == "vgg16":
        return VGGWrapper(input_shape=tuple(input_shape), pretrained=True)
    elif model_name == "gpt2":
        seq_len = input_shape[0] if input_shape else 128
        return Gpt2Wrapper(seq_length=seq_len, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    cfg = Config.from_yaml("config.yaml")
    
    print("="*70)
    print("ML COMPILER BENCHMARK FRAMEWORK")
    print("="*70)
    print(f"Models: {', '.join(model_cfg.name for model_cfg in cfg.models)}")
    print(f"Compilers: {', '.join(cfg.compilers)}")
    print(f"Warmup iterations: {cfg.benchmark.warmup_iterations}")
    print(f"Measured iterations: {cfg.benchmark.measured_iterations}")
    print("="*70)
    
    device = get_device()
    
    runner = BenchmarkRunner(
        device=device,
        warmup_iters=cfg.benchmark.warmup_iterations,
        measured_iters=cfg.benchmark.measured_iterations
    )
    
    output_path = f"{cfg.output.save_path}/benchmark_results.csv"
    
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Cleared previous results at: {output_path}\n")
    
    for model_idx, model_cfg in enumerate(cfg.models):
        print(f"\n{'='*70}")
        print(f"PROCESSING MODEL {model_idx + 1}/{len(cfg.models)}: {model_cfg.name}")
        print(f"{'='*70}")
        
        model_wrapper = get_model(model_cfg.name, model_cfg.input_shape)
        model_results = []
        
        for compiler_name in cfg.compilers:
            compiler = get_compiler(compiler_name)
            
            for batch_size in model_cfg.batch_sizes:
                try:
                    run_stats = runner.run_benchmark(model_wrapper, compiler, batch_size)
                    model_results.append(run_stats)
                except Exception as e:
                    print(f"\nERROR: Benchmarking {model_cfg.name} with {compiler_name} (batch={batch_size}): {e}")
                    print("Continuing with next configuration...\n")
        
        if model_results:
            ResultsWriter.write_csv(model_results, output_path, append=(model_idx > 0))
        
        del model_wrapper
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
