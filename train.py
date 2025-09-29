import warnings
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.trainer.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_logging

warnings.filterwarnings("ignore", category=UserWarning)


def get_dataloaders(config: DictConfig, text_encoder, device: str):
    dataloaders = {}
    batch_transforms = {}
    for split in ['train', 'val', 'test']:
        if split in config.datasets:
            dataset_cfg = config.datasets[split]
            dataset = instantiate(dataset_cfg, text_encoder=text_encoder)
            dataloader_kwargs = OmegaConf.to_container(config.dataloader, resolve=True)
            collate_fn = None
            if 'collate_fn' in dataloader_kwargs:
                collate_fn = dataset.collate_fn
                del dataloader_kwargs['collate_fn']
            dataloader = instantiate(
                config.dataloader,
                dataset=dataset,
                shuffle=(split == 'train'),
                collate_fn=collate_fn
            )
            dataloaders[split] = dataloader
    if 'batch_transforms' in config:
        for transform_type in ['train', 'inference']:
            if transform_type in config.batch_transforms:
                transforms_dict = {}
                for tensor_name, transform_cfg in config.batch_transforms[transform_type].items():
                    transform = instantiate(transform_cfg)
                    if isinstance(transform, torch.nn.Module):
                        transform = transform.to(device)
                    transforms_dict[tensor_name] = transform
                batch_transforms[transform_type] = transforms_dict
    return dataloaders, batch_transforms


@hydra.main(version_base=None, config_path="src/config", config_name="main")
def main(config: DictConfig):
    set_random_seed(config.trainer.seed)
    logger = setup_logging(config)
    logger.info("Starting experiment")
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device
    logger.info(f"Using device: {device}")
    text_encoder = instantiate(config.text_encoder)
    logger.info(f"Text encoder vocabulary size: {len(text_encoder)}")
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)
    logger.info(f"Created dataloaders for splits: {list(dataloaders.keys())}")
    model = instantiate(config.model).to(device)
    logger.info("Model initialized")
    criterion = instantiate(config.loss_function).to(device)
    logger.info(f"Loss function: {type(criterion).__name__}")
    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        if hasattr(config.metrics, metric_type):
            for metric_cfg in config.metrics[metric_type]:
                metric = instantiate(metric_cfg, text_encoder=text_encoder)
                metrics[metric_type].append(metric)
                logger.info(f"Added {metric_type} metric: {metric.name}")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    lr_scheduler = None
    if hasattr(config, 'scheduler'):
        lr_scheduler = instantiate(config.scheduler, optimizer=optimizer)
        logger.info(f"LR Scheduler: {type(lr_scheduler).__name__}")
    writer = instantiate(config.writer)
    logger.info(f"Writer: {type(writer).__name__}")
    epoch_len = config.trainer.get("epoch_len")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
