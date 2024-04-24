import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin


parser = argparse.ArgumentParser(description="Generic runner for VAE models")
parser.add_argument(
    "--config",
    "-c",
    dest="filename",
    metavar="FILE",
    help="path to the config file",
    default="configs/vae.yaml",
)

args = parser.parse_args()
with open(args.filename, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger = TensorBoardLogger(
    save_dir=config["logging_params"]["save_dir"],
    name=config["model_params"]["name"],
)

# For reproducibility
seed_everything(config["exp_params"]["manual_seed"], True)

model = vae_models[config["model_params"]["name"]](**config["model_params"])
ckpt = torch.load(
    "/home/chenyu/Desktop/visuotactile_representations/DipVAE/logs/DIPVAE/Train_openteach_66/checkpoints/epoch=23-step=10271.ckpt"
)
experiment = VAEXperiment(model, config["exp_params"])
experiment.load_state_dict(ckpt["state_dict"])


data = VAEDataset(
    **config["data_params"], pin_memory=len(config["trainer_params"]["gpus"]) != 0
)

data.setup()
runner = Trainer(
    logger=tb_logger,
    strategy=DDPPlugin(find_unused_parameters=False),
    **config["trainer_params"],
)


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.validate(experiment, datamodule=data)
