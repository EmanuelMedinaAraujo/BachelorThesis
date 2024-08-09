import wandb
from tqdm.gui import tqdm
from omegaconf import DictConfig, OmegaConf


def log_used_device(device):
    tqdm.write(f"Using {device} as device")

class Logger:

    def __init__(self, dataset_length, log_in_wandb=False, cfg: DictConfig = None,
                 log_in_console=False):
        self.log_in_wandb = log_in_wandb
        self.log_in_console = log_in_console
        self.dataset_length = dataset_length

        if self.log_in_wandb:
            if cfg is None:
                raise RuntimeError("Could not read the configuration correctly")
            init_wandb(cfg)

    def log_training(self, loss , epoch_num: int, accuracy):
        if self.log_in_console:
            tqdm.write(f"Epoch {epoch_num + 1}: Accuracy: {accuracy:>0.2f}%, Mean loss: {loss:>7f}")
        if self.log_in_wandb:
            wandb.log({"train/acc": accuracy, "train/loss": loss})
            return

    def log_test(self, accuracy, loss):
        if self.log_in_console:
            tqdm.write(f"Test: Accuracy: {accuracy :>0.2f}%, Avg loss: {loss:>8f}\n")
        if self.log_in_wandb:
            wandb.log({"test/acc": accuracy, "test/loss": loss})


def init_wandb(cfg: DictConfig):
    wandb.require("core")
    wandb.init(
        # set the wandb project where this run will be logged
        project=cfg.hyperparams.project_name,

        # track hyperparameters and run metadata
        config=OmegaConf.to_container(cfg)
    )
