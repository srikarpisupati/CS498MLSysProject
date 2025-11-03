# CS498MLSysProject

Minimal ML Compiler Benchmark

### Quickstart

1) Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate ml-benchmark
```

2) Run the benchmark:
```bash
python run_benchmark.py
```

Results are saved to `results/benchmark_results.csv`.

### Configuration
- Edit `config.yaml` to choose model, compilers, batch sizes, and iterations.
- The script auto-detects CPU/GPU via PyTorch and uses what's available.

### Analyze Results (optional)
```bash
python analyze_results.py  # or: python analyze_results.py results/benchmark_results.csv
```

That's it.