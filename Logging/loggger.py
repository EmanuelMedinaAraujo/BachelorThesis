import wandb
from omegaconf import DictConfig, OmegaConf


def log_used_device(device):
    print(f"Using {device} as device")


class Logger:

    def __init__(self, dataset_length, log_in_wandb=False, cfg: DictConfig = None):
        self.log_in_wandb = log_in_wandb
        self.dataset_length = dataset_length

        if self.log_in_wandb:
            if cfg is None:
                raise RuntimeError("Could not read the configuration correctly")
            init_wandb(cfg)

    def log_training(self, loss , epoch_num: int, accuracy):
        print(f"Epoch {epoch_num + 1}:\tAccuracy: {accuracy:>0.2f}%, Mean loss: {loss:>7f}")
        if self.log_in_wandb:
            wandb.log({"train/acc": accuracy, "train/loss": loss})
            return

    def log_test(self, accuracy, loss):
        print(f"Test Error: \n Accuracy: {accuracy :>0.2f}%, Avg loss: {loss:>8f} \n")
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
