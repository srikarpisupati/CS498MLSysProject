# CS498MLSysProject

ML Compiler Benchmark Framework

## Quick Setup

Run the setup script to install conda and create the environment:

```bash
./setup.sh
```

This script will:
1. Install Miniconda (if not already installed)
2. Accept conda Terms of Service
3. Create the `ml-benchmark` conda environment from `environment.yml`
4. Install all dependencies including PyTorch 2.1.0, torchvision, and transformers

## Usage

After setup, activate the environment and run benchmarks:

```bash
conda activate ml-benchmark
python run_benchmark.py
```

Or run directly:

```bash
~/miniconda3/envs/ml-benchmark/bin/python run_benchmark.py
```

## GPU Support

**Current Status:** The benchmark is currently running on **CPU only**. 

The node has a Tesla P100 GPU, but the NVIDIA driver kernel module is not loaded/accessible.

### Quick Fix Steps:

1. **Diagnose the issue:**
   ```bash
   ./fix_gpu.sh          # Comprehensive diagnostic
   python check_gpu.py    # Quick GPU status check
   ```

2. **Fix options (see GPU_FIX_GUIDE.md for details):**
   - **Cluster/Cloud:** Request GPU allocation via scheduler/admin
   - **Has sudo:** Install NVIDIA drivers (`sudo ubuntu-drivers autoinstall`)
   - **Ask admin:** Share `GPU_FIX_GUIDE.md` with system administrator

3. **Once GPU is accessible:**
   - Re-run the benchmark - it will automatically use GPU if available
   - Expect 5-10x speedup compared to CPU results
   - Memory metrics will show actual GPU memory usage

See `GPU_FIX_GUIDE.md` for detailed troubleshooting steps.

## Environment

The conda environment includes:
- Python 3.10
- PyTorch 2.1.0 with CUDA 11.8 support
- torchvision 0.16.0
- transformers 4.35.0 (compatible with PyTorch 2.1.0)
- numpy, pyyaml, and other dependencies

**Note:** PyTorch was installed with CUDA support, but GPU requires system NVIDIA drivers to be accessible.