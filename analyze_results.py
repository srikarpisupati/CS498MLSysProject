import csv
import sys
import os

def analyze_results(csv_path="results/benchmark_results.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        return
    
    by_model = {}
    all_compilers = set()
    all_batch_sizes = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('compiler'):
                continue
                
            model = row['model']
            compiler = row['compiler']
            batch_size = int(row['batch_size'])
            
            all_compilers.add(compiler)
            all_batch_sizes.add(batch_size)
            
            if model not in by_model:
                by_model[model] = {}
            
            key = (compiler, batch_size)
            by_model[model][key] = {
                'latency_mean_ms': float(row['latency_mean_ms']),
                'latency_std_ms': float(row['latency_std_ms']),
                'latency_p50_ms': float(row['latency_p50_ms']),
                'latency_p95_ms': float(row['latency_p95_ms']),
                'throughput_samples_per_sec': float(row['throughput_samples_per_sec']),
                'peak_memory_mb': float(row['peak_memory_mb']),
                'avg_memory_mb': float(row['avg_memory_mb']),
                'compile_time_sec': row.get('compile_time_sec', 'N/A')
            }
    
    compilers = sorted(all_compilers)
    batch_sizes = sorted(all_batch_sizes)
    
    print("="*80)
    print("BENCHMARK RESULTS ANALYSIS")
    print("="*80)
    print()
    
    # Show detailed results for each model
    for model_idx, (model, results) in enumerate(sorted(by_model.items())):
        if model_idx > 0:
            print("\n" + "="*80)
        
        print(f"MODEL: {model}")
        print("="*80)
        print()
        
        for compiler in compilers:
            compiler_has_data = any((compiler, bs) in results for bs in batch_sizes)
            if not compiler_has_data:
                continue
                
            print(f"  Compiler: {compiler}")
            print("  " + "-" * 76)
            
            for batch_size in batch_sizes:
                key = (compiler, batch_size)
                if key not in results:
                    continue
                
                stat = results[key]
                print(f"    Batch Size {batch_size}:")
                print(f"      Latency (mean ± std): {stat['latency_mean_ms']:.3f} ± {stat['latency_std_ms']:.3f} ms")
                print(f"      Latency (p50/p95):    {stat['latency_p50_ms']:.3f} / {stat['latency_p95_ms']:.3f} ms")
                print(f"      Throughput:           {stat['throughput_samples_per_sec']:.2f} samples/sec")
                print(f"      Peak Memory:          {stat['peak_memory_mb']:.2f} MB")
                print(f"      Avg Memory:           {stat['avg_memory_mb']:.2f} MB")
                
                if stat['compile_time_sec'] != 'N/A':
                    compile_time = float(stat['compile_time_sec'])
                    print(f"      Compile Time:         {compile_time:.3f} s")
                else:
                    print(f"      Compile Time:         N/A")
                print()
        
        eager_compiler = 'pytorch_eager'
        other_compilers = [c for c in compilers if c != eager_compiler]
        
        if eager_compiler in [c for c, _ in results.keys()] and other_compilers:
            print("  " + "="*76)
            print(f"  SPEEDUP vs {eager_compiler} baseline:")
            print("  " + "="*76)
            print()
            
            for other_compiler in other_compilers:
                for batch_size in batch_sizes:
                    eager_key = (eager_compiler, batch_size)
                    other_key = (other_compiler, batch_size)
                    
                    if eager_key in results and other_key in results:
                        eager = results[eager_key]
                        other = results[other_key]
                        
                        latency_speedup = eager['latency_mean_ms'] / other['latency_mean_ms']
                        throughput_speedup = other['throughput_samples_per_sec'] / eager['throughput_samples_per_sec']
                        
                        print(f"    {other_compiler} (Batch {batch_size}):")
                        print(f"      Latency:     {eager['latency_mean_ms']:.2f} ms → {other['latency_mean_ms']:.2f} ms")
                        print(f"      Speedup:     {latency_speedup:.2f}x ({((latency_speedup-1)*100):+.1f}%)")
                        print(f"      Throughput:  {eager['throughput_samples_per_sec']:.2f} → {other['throughput_samples_per_sec']:.2f} samples/sec")
                        print(f"      Improvement: {throughput_speedup:.2f}x ({((throughput_speedup-1)*100):+.1f}%)")
                        
                        if other['compile_time_sec'] != 'N/A':
                            compile_time = float(other['compile_time_sec'])
                            print(f"      Compile Cost: {compile_time:.2f} s")
                        print()
    
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    
    print(f"{'Model':<20} {'Compiler':<25} {'Batch':<6} {'Latency(ms)':<12} {'Throughput':<15} {'Memory(MB)':<12} {'Compile(s)':<12}")
    print("-" * 110)
    
    for model in sorted(by_model.keys()):
        results = by_model[model]
        for compiler in compilers:
            for batch_size in batch_sizes:
                key = (compiler, batch_size)
                if key in results:
                    stat = results[key]
                    compile_str = f"{float(stat['compile_time_sec']):.2f}" if stat['compile_time_sec'] != 'N/A' else "N/A"
                    print(f"{model:<20} {compiler:<25} {batch_size:<6} {stat['latency_mean_ms']:<12.3f} {stat['throughput_samples_per_sec']:<15.2f} {stat['peak_memory_mb']:<12.2f} {compile_str:<12}")
    
    print("="*80)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.csv"
    analyze_results(csv_path)
