import json
from pathlib import Path
from typing import Any, Dict
import torch

class LocalWriter:
    def __init__(self, save_dir: str = "experiments", run_name: str = "local", **kwargs):
        self.save_dir = Path(save_dir) / run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.save_dir / "metrics.json"
        self.metrics = {}
        
    def add_scalar(self, tag: str, value: float, step: int = None):
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append({"step": step, "value": value})
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
