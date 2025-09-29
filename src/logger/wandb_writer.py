import wandb
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import torch


class WandBWriter:
    
    def __init__(
        self, 
        project_name: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        mode: str = "online",
        loss_names: list = None,
        log_checkpoints: bool = False,
        id_length: int = 8,
        **kwargs
    ):
        self.project_name = project_name
        self.entity = entity
        self.run_name = run_name
        self.mode = mode
        self.loss_names = loss_names or ["loss"]
        self.log_checkpoints = log_checkpoints
        self.id_length = id_length
        
        wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=self.run_name,
            mode=self.mode,
            config=kwargs.get("config", {})
        )
        
        self.step = 0
        self.mode = "train"
        
    def set_step(self, step: int, mode: str = "train") -> None:
        self.step = step
        self.mode = mode
        
    def add_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        if step is None:
            step = self.step
            
        wandb.log({f"{self.mode}/{tag}": value}, step=step)
        
    def add_image(self, tag: str, image: torch.Tensor, step: Optional[int] = None) -> None:
        if step is None:
            step = self.step
            

        if image.dim() == 3 and image.shape[0] in [1, 3]:
            image = image.permute(1, 2, 0)
            
        wandb.log({f"{self.mode}/{tag}": wandb.Image(image.cpu().numpy())}, step=step)
        
    def add_table(self, tag: str, table: pd.DataFrame, step: Optional[int] = None) -> None:
        if step is None:
            step = self.step
            
        wandb.log({f"{self.mode}/{tag}": wandb.Table(dataframe=table)}, step=step)
        
    def add_checkpoint(self, checkpoint_path: str, save_dir: str) -> None:

        if self.log_checkpoints:
            wandb.save(str(checkpoint_path))
            
    def finish(self) -> None:

        wandb.finish()