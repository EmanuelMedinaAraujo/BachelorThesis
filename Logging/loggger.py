import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm.gui import tqdm


class Logger:
    """
    A class responsible for logging the process of model training and testing in the console and in wandb.
    For logging in the console tqdm is used.
    """

    def __init__(self, log_in_wandb=False, cfg: DictConfig = None,
                 log_in_console=False):
        """
        Initializes the logger.

        :param log_in_wandb: If True, the logger will log to wandb.
        :param cfg: The configuration that will be used by wandb to track the hyperparameters and run metadata.
        :param log_in_console: If True, the logger will log to the console using tqdm.
        """
        self.log_in_wandb = log_in_wandb
        self.log_in_console = log_in_console

        if self.log_in_wandb:
            if cfg is None:
                raise RuntimeError("Could not read the configuration correctly")
            init_wandb(cfg)

    @staticmethod
    def log_used_device(device):
        tqdm.write(f"Using {device} as device")

    def log_training(self, loss, epoch_num: int, accuracy):
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
