from dataclasses import dataclass, field
from typing import List
import yaml

@dataclass
class BenchmarkConfig:
    warmup_iterations: int
    measured_iterations: int

@dataclass
class ModelConfig:
    name: str
    input_shape: List[int]
    batch_sizes: List[int]
    precision: str

@dataclass
class OutputConfig:
    format: str
    save_path: str

@dataclass
class Config:
    benchmark: BenchmarkConfig
    models: List[ModelConfig]
    compilers: List[str]
    output: OutputConfig
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'models' in data:
            model_entries = data['models']
        elif 'model' in data:
            model_entries = [data['model']]
        else:
            raise ValueError("Config must specify 'model' or 'models'")
        
        model_configs = [ModelConfig(**entry) for entry in model_entries]
        
        return cls(
            benchmark=BenchmarkConfig(**data['benchmark']),
            models=model_configs,
            compilers=data['compilers'],
            output=OutputConfig(**data['output'])
        )