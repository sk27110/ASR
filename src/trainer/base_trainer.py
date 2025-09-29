from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from itertools import repeat
from src.metrics.tracker import SimpleMetricTracker


def inf_loop(data_loader):

    for loader in repeat(data_loader):
        yield from loader


class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        metrics: Dict[str, list],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        text_encoder: Any,
        config: Dict[str, Any],
        device: torch.device,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        logger: Any,
        writer: Any,
        epoch_len: Optional[int] = None,
        skip_oom: bool = True,
        batch_transforms: Optional[Dict[str, Any]] = None,
    ):
        self.is_train = True
        self.config = config
        self.cfg_trainer = config.get("trainer", {})
        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = self.cfg_trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.text_encoder = text_encoder
        self.batch_transforms = batch_transforms or {}

        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }


        self._last_epoch = 0
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.get("n_epochs", 100)


        self.save_period = self.cfg_trainer.get("save_period", 1)
        self.monitor = self.cfg_trainer.get("monitor", "off")

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)

        self.writer = writer

        self.metrics = metrics
        loss_names = config.get("writer", {}).get("loss_names", ["loss"])
        self.train_metrics = SimpleMetricTracker()
        self.evaluation_metrics = SimpleMetricTracker()

        save_dir = self.cfg_trainer.get("save_dir", "checkpoints")
        run_name = config.get("writer", {}).get("run_name", "experiment")
        self.checkpoint_dir = Path(save_dir) / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg_trainer.get("resume_from"):
            self._resume_checkpoint(self.cfg_trainer["resume_from"])
        

        if self.cfg_trainer.get("from_pretrained"):
            self._from_pretrained(self.cfg_trainer["from_pretrained"])

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise

    def _train_process(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            logs = {"epoch": epoch, **result}
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            best, should_stop, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if should_stop:
                break

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Train Epoch {epoch}", total=self.epoch_len)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                batch = self.process_batch(batch, is_training=True)
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                raise e

            current_metrics = self.train_metrics.result()
            progress_bar.set_postfix({k: f"{v:.4f}" for k, v in current_metrics.items()})

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self._log_scalars(self.train_metrics)
                if self.lr_scheduler is not None:
                    self.writer.add_scalar("learning_rate", self.lr_scheduler.get_last_lr()[0])
                self._log_batch(batch_idx, batch)

            if batch_idx + 1 >= self.epoch_len:
                break

        eval_results = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            part_results = self._evaluate_epoch(epoch, part, dataloader)
            eval_results.update({f"{part}_{k}": v for k, v in part_results.items()})

        return {**self.train_metrics.result(), **eval_results}

    def _evaluate_epoch(self, epoch: int, part: str, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Eval {part}"):
                self.process_batch(batch, is_training=False)

        self.writer.set_step(epoch * self.epoch_len, part)
        self._log_scalars(self.evaluation_metrics)
        
        return self.evaluation_metrics.result()

    def process_batch(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        raise NotImplementedError

    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device_tensors = self.cfg_trainer.get("device_tensors", [])
        for tensor_name in device_tensors:
            if tensor_name in batch and isinstance(batch[tensor_name], torch.Tensor):
                batch[tensor_name] = batch[tensor_name].to(self.device)
        return batch

    def transform_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type, {})
        
        for transform_name, transform_fn in transforms.items():
            if transform_name in batch:
                batch[transform_name] = transform_fn(batch[transform_name])
        return batch

    def _clip_grad_norm(self):
        max_grad_norm = self.cfg_trainer.get("max_grad_norm")
        if max_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), max_grad_norm)

    @torch.no_grad()
    def _get_grad_norm(self, norm_type: float = 2) -> float:
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _monitor_performance(self, logs: Dict[str, float], not_improved_count: int):
        best = False
        should_stop = False

        if self.mnt_mode != "off" and self.mnt_metric in logs:
            current_value = logs[self.mnt_metric]
            
            if self.mnt_mode == "min":
                improved = current_value <= self.mnt_best
            else: 
                improved = current_value >= self.mnt_best

            if improved:
                self.mnt_best = current_value
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(f"Early stopping after {self.early_stop} epochs without improvement")
                should_stop = True

        return best, should_stop, not_improved_count

    def _save_checkpoint(self, epoch: int, save_best: bool = False, only_best: bool = False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        if save_best:
            best_path = self.checkpoint_dir / "model_best.pth"
            torch.save(state, best_path)
            self.logger.info(f"Saved best model: {best_path}")

        if not only_best or (only_best and not save_best):
            checkpoint_path = self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth"
            torch.save(state, checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _resume_checkpoint(self, resume_path: str):
        self.logger.info(f"Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device)
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]
        self.model.load_state_dict(checkpoint["state_dict"])
        
        if (checkpoint["config"].get("model") == self.config.get("model") and
            checkpoint["config"].get("optimizer") == self.config.get("optimizer")):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.lr_scheduler and checkpoint.get("lr_scheduler"):
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        else:
            self.logger.warning("Optimizer/scheduler not resumed due to config mismatch")

        self.logger.info(f"Resumed training from epoch {self.start_epoch}")

    def _from_pretrained(self, pretrained_path: str):
        self.logger.info(f"Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    def _log_scalars(self, metric_tracker: SimpleMetricTracker):
        for metric_name, value in metric_tracker.result().items():
            self.writer.add_scalar(metric_name, value)

    @abstractmethod
    def _log_batch(self, batch_idx: int, batch: Dict[str, Any], mode: str = "train"):
        pass