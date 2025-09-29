from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch

from src.metrics.tracker import SimpleMetricTracker
from src.metrics.utils import wer, cer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def process_batch(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        model_inputs = {
            'spectrogram': batch['spectrogram'],
            'spectrogram_length': batch['spectrogram_lengths']
        }
        model_inputs.update(batch)
        outputs = self.model(**model_inputs)
        batch.update(outputs)
        losses = self.criterion(**batch)
        batch.update(losses)
        if is_training:
            self.optimizer.zero_grad()
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        metrics_tracker = self.train_metrics if is_training else self.evaluation_metrics
        for loss_name in self.config.get("writer", {}).get("loss_names", ["loss"]):
            if loss_name in batch:
                metrics_tracker.update(loss_name, batch[loss_name].item())
        metric_funcs = self.metrics["train" if is_training else "inference"]
        for metric in metric_funcs:
            metric_value = metric(**batch)
            metrics_tracker.update(metric.name, metric_value)
        return batch

    def _log_batch(self, batch_idx: int, batch: Dict[str, Any], mode: str = "train"):
        if mode == "train":
            if "spectrogram" in batch and batch_idx % (self.log_step * 5) == 0:
                self.log_spectrogram(batch["spectrogram"])
        else:
            self.log_spectrogram(batch["spectrogram"])
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram: torch.Tensor):
        if hasattr(self.writer, 'add_image') and len(spectrogram) > 0:
            spec_to_plot = spectrogram[0].detach().cpu()
            spec_to_plot = (spec_to_plot - spec_to_plot.mean()) / (spec_to_plot.std() + 1e-8)
            if spec_to_plot.dim() == 2:
                spec_to_plot = spec_to_plot.unsqueeze(0)
            self.writer.add_image("spectrogram", spec_to_plot)

    def log_predictions(
        self,
        text: list,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        audio_path: list,
        examples_to_log: int = 5,
        **batch
    ):
        if not hasattr(self.writer, 'add_table'):
            return
        argmax_inds = log_probs.detach().cpu().argmax(-1)
        lengths = log_probs_length.cpu().numpy()
        predictions = []
        raw_predictions = []
        for i, (inds, length) in enumerate(zip(argmax_inds, lengths)):
            inds_seq = inds[:int(length)].tolist()
            raw_pred = self.text_encoder.decode(inds_seq)
            pred_text = self.text_encoder.ctc_decode(inds_seq)
            predictions.append(pred_text)
            raw_predictions.append(raw_pred)
        rows = {}
        for i, (pred, target, raw_pred, path) in enumerate(
            zip(predictions, text, raw_predictions, audio_path)
        ):
            if i >= examples_to_log:
                break
            target_norm = self.text_encoder.normalize_text(target)
            wer_score = wer(target_norm, pred) * 100
            cer_score = cer(target_norm, pred) * 100
            rows[Path(path).name] = {
                "target": target_norm,
                "prediction": pred,
                "raw_prediction": raw_pred,
                "wer": f"{wer_score:.2f}%",
                "cer": f"{cer_score:.2f}%",
            }
        if rows:
            df = pd.DataFrame.from_dict(rows, orient="index")
            self.writer.add_table("predictions", df)
