# CS498MLSysProject

ML Compiler Benchmark Framework

## Setup

### First Time (One-time installation)

```bash
cd ~/CS498MLSysProject
./setup.sh
```

What `setup.sh` does:
- Installs Miniconda (if missing) and accepts the needed ToS.
- Creates the `ml-benchmark` env from `environment.yml`, which now includes the CUDA toolkit (`cuda-toolkit`, `cuda-nvcc`) required for TVM’s CUDA builds.
- Installs the bundled TVM wheel `tlcpack_cu116-0.11.1-...whl`. If you replace or update the wheel, drop the new file in the repo root before re-running `setup.sh`.

### Every Session

```bash
cd ~/CS498MLSysProject
source startup.sh
```

Or manually:
```bash
conda activate ml-benchmark
```

If you prefer not to activate the shell, you can still run commands via:

```bash
conda run -n ml-benchmark python run_benchmark.py
```

## Running

### Run Benchmarks

```bash
python run_benchmark.py
```

Results saved to `results/benchmark_results.csv`.

### Analyze Results

```bash
python analyze_results.py
```

## Configuration

`config.yaml` controls everything:
- `models`: list of model entries (name, input shape, batch sizes, precision). Add/remove entries to run multiple architectures in one go (e.g., `resnet50`, `mobilenet_v3`, `vgg16`, `gpt2`—language models use `input_shape: [sequence_length]`).
- `compilers`: list of compiler keys (`pytorch_eager`, `torchscript`, `onnxruntime`, `tvm`, etc.).
- `benchmark`: warmup/measured iterations.
- `output`: result format/path.

## TVM Support

- The repository ships with `tlcpack_cu116-0.11.1-...whl`. `setup.sh` installs it automatically so the `tvm` compiler entry “just works”.
- If you need a different CUDA/Python combo, download the matching TLCPack wheel from the [official release page](https://github.com/tlc-pack/tlcpack/releases), place it in the repo root, and rerun `setup.sh` (or `pip install <wheel>` inside the env).
- CUDA builds require `nvcc`; the environment already installs `cuda-toolkit` 11.8 so TVM can JIT kernels for the P100 (sm_60). If `nvcc` is missing, rerun `setup.sh` or check your CUDA installation.
- When CUDA is unavailable, TVM automatically falls back to LLVM/CPU so benchmarks can still complete (albeit slower).

## ONNX Runtime Support

- `onnxruntime-gpu==1.15.1` is installed via `environment.yml`; no manual steps required.
- The `onnxruntime` compiler entry exports the PyTorch model once to ONNX (with dynamic batch axis) and runs it using the CUDA Execution Provider, falling back to CPU if CUDA is unavailable.
- Because ONNX Runtime reuses highly optimized kernels, compilation time is close to zero compared to TVM.
