import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm.gui import tqdm


def init_wandb(cfg: DictConfig):
    wandb.init(
        # set the wandb project where this run will be logged
        project=cfg.hyperparams.project_name,
        # track hyperparameters and run metadata
        config=OmegaConf.to_container(cfg)
    )


class GeneralLogger:
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

    def log_test(self, accuracy, loss, current_step = None):
        if self.log_in_console:
            tqdm.write(f"Test: Accuracy: {accuracy :>0.2f}%, Avg loss: {loss:>8f}\n")
        if self.log_in_wandb:
            wandb.log({"test/acc": accuracy, "test/loss": loss}, step=current_step)

    def watch_model(self, model):
        if self.log_in_wandb:
            wandb.watch(model)

    def log_train_rollout(self, approx_kl, clip_fraction, clip_range, entropy_loss, explained_variance, learning_rate,
                          loss, n_updates, policy_gradient_loss, value_loss, std, current_step = None):
        if self.log_in_console:
            tqdm.write(f"train/approx_kl: {approx_kl}")
            tqdm.write(f"train/clip_fraction: {clip_fraction}")
            tqdm.write(f"train/clip_range: {clip_range}")
            tqdm.write(f"train/entropy_loss: {entropy_loss}")
            tqdm.write(f"train/explained_variance: {explained_variance}")
            tqdm.write(f"train/learning_rate: {learning_rate}")
            tqdm.write(f"train/loss: {loss}")
            tqdm.write(f"train/n_updates: {n_updates}")
            tqdm.write(f"train/policy_gradient_loss: {policy_gradient_loss}")
            tqdm.write(f"train/std: {std}")
            tqdm.write(f"train/value_loss: {value_loss}")
            print("--------------------------------")
        if self.log_in_wandb:
            wandb.log({"train/approx_kl": approx_kl, "train/clip_fraction": clip_fraction,
                       "train/clip_range": clip_range, "train/entropy_loss": entropy_loss,
                       "train/explained_variance": explained_variance, "train/learning_rate": learning_rate,
                       "train/loss": loss, "train/n_updates": n_updates,
                       "train/policy_gradient_loss": policy_gradient_loss,
                       "train/std": std, "train/value_loss": value_loss}, step=current_step)

    def log_rollout(self, ep_rew_mean, success_rate, rollout_buf_mean_rew, current_step = None):
        if self.log_in_console:
            tqdm.write(f"Rollout: Mean reward: {ep_rew_mean:>8f}, Success rate: {success_rate:>0.2f}%")
        if self.log_in_wandb:
            wandb.log({"rollout/mean_reward": ep_rew_mean,
                       "rollout/success_rate": success_rate,
                       "rollout/buf_mean_reward": rollout_buf_mean_rew}, step=current_step)

    def log_plot(self, path, current_step = None):
        if self.log_in_wandb:
            wandb.log({"plot": wandb.Image(path)}, step=current_step)

    def finish_logging(self):
        if self.log_in_wandb:
            wandb.finish()
