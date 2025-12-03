import csv
import os
from typing import List
from ..core.metrics import BenchmarkMetrics

class ResultsWriter:
    
    @staticmethod
    def write_csv(results: List[BenchmarkMetrics], output_path: str, append: bool = False):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if not results:
            print("No results to write!")
            return
        
        fieldnames = list(results[0].to_dict().keys())
        file_exists = os.path.exists(output_path)
        
        mode = 'a' if append else 'w'
        with open(output_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # Write header only if file is new or we're overwriting
            if not append or not file_exists:
                writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())
        
        action = "appended to" if append and file_exists else "saved to"
        print(f"\n✓ Results {action}: {output_path}")

