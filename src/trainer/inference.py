from pathlib import Path
from typing import Dict, Any, Optional
import torch
from tqdm.auto import tqdm

from src.metrics.tracker import SimpleMetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        text_encoder: Any,
        save_path: Optional[Path] = None,
        metrics: Optional[Dict[str, list]] = None,
        batch_transforms: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.cfg_inferencer = config.get("inferencer", {})
        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms or {}
        self.text_encoder = text_encoder
        self.save_path = Path(save_path) if save_path else None
        self.evaluation_dataloaders = dataloaders
        self.metrics = metrics or {}
        self.evaluation_metrics = SimpleMetricTracker()
        pretrained_path = self.cfg_inferencer.get("from_pretrained")
        if pretrained_path:
            self._from_pretrained(pretrained_path)
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)

    def run_inference(self) -> Dict[str, Dict[str, float]]:
        self.model.eval()
        results = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            self.logger.info(f"Running inference on {part}")
            part_results = self._inference_part(part, dataloader)
            results[part] = part_results
        return results

    def _inference_part(self, part: str, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.evaluation_metrics.reset()
        part_save_path = self.save_path / part if self.save_path else None
        if part_save_path:
            part_save_path.mkdir(exist_ok=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Inference {part}")):
                batch = self.process_batch(batch, part, batch_idx, part_save_path)
        return self.evaluation_metrics.result()

    def process_batch(
        self,
        batch: Dict[str, Any],
        part: str,
        batch_idx: int,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        outputs = self.model(**batch)
        batch.update(outputs)
        for metric in self.metrics.get("inference", []):
            metric_value = metric(**batch)
            self.evaluation_metrics.update(metric.name, metric_value)
        if save_path and "log_probs" in batch:
            self._save_predictions(batch, batch_idx, save_path)
        return batch

    def _save_predictions(self, batch: Dict[str, Any], batch_idx: int, save_path: Path):
        log_probs = batch["log_probs"]
        log_probs_length = batch.get("log_probs_length")
        texts = batch.get("text", [])
        audio_paths = batch.get("audio_path", [])
        batch_size = log_probs.shape[0]
        for i in range(batch_size):
            if log_probs_length is not None:
                length = int(log_probs_length[i])
                probs = log_probs[i, :length]
            else:
                probs = log_probs[i]
            argmax_inds = probs.argmax(-1).cpu().numpy()
            pred_text = self.text_encoder.ctc_decode(argmax_inds.tolist())
            raw_pred = self.text_encoder.decode(argmax_inds.tolist())
            output = {
                "prediction": pred_text,
                "raw_prediction": raw_pred,
                "probabilities": probs.cpu(),
                "target_text": texts[i] if i < len(texts) else "",
                "audio_path": audio_paths[i] if i < len(audio_paths) else "",
            }
            output_id = batch_idx * batch_size + i
            output_path = save_path / f"prediction_{output_id:06d}.pt"
            torch.save(output, output_path)

    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch

    def transform_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        transforms = self.batch_transforms.get("inference", {})
        for transform_name, transform_fn in transforms.items():
            if transform_name in batch:
                batch[transform_name] = transform_fn(batch[transform_name])
        return batch
