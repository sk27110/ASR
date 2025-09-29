import logging
import os
import random
import secrets
import shutil
import string
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.io_utils import ROOT_PATH


def set_worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_id(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def log_git_state(save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    commit_path = save_dir / "git_commit.txt"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        commit_path.write_text(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit_path.write_text("git_not_available")
    patch_path = save_dir / "git_diff.patch"
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            patch_path.write_text(result.stdout)
        else:
            patch_path.write_text("no_changes")
    except (subprocess.CalledProcessError, FileNotFoundError):
        patch_path.write_text("git_not_available")


def setup_experiment_directory(config: DictConfig) -> Path:
    save_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    if save_dir.exists():
        if config.trainer.get("resume_from"):
            saved_config = OmegaConf.load(save_dir / "config.yaml")
            run_id = saved_config.writer.run_id
            print(f"Resuming experiment: {run_id}")
            return save_dir
        elif config.trainer.get("override", False):
            print(f"Overriding existing directory: {save_dir}")
            shutil.rmtree(save_dir)
        else:
            raise ValueError(
                f"Save directory {save_dir} already exists. "
                "Set trainer.override=True to overwrite or change run_name."
            )
    save_dir.mkdir(parents=True, exist_ok=True)
    run_id = generate_id(length=config.writer.get("id_length", 8))
    OmegaConf.set_struct(config, False)
    config.writer.run_id = run_id
    OmegaConf.set_struct(config, True)
    OmegaConf.save(config, save_dir / "config.yaml")
    log_git_state(save_dir)
    print(f"Created new experiment: {run_id} at {save_dir}")
    return save_dir


def setup_logging(config: DictConfig) -> logging.Logger:
    save_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("asr_trainer")
    logger.info(f"Experiment directory: {save_dir}")
    return logger


def get_experiment_config(experiment_path: Path) -> DictConfig:
    config_path = experiment_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)
