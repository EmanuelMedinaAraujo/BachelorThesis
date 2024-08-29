from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class LoggerCallback(BaseCallback):

    def __init__(self, logger, visualization_history, goal_to_vis, param_to_vis, hyperparams, visualization_call,
                 device,
                 verbose: int = 2):
        super(LoggerCallback, self).__init__(verbose)
        self.custom_logger = logger
        self.visualization_history = visualization_history
        self.goal_to_vis = goal_to_vis
        self.param_to_vis = param_to_vis
        self.hyperparams = hyperparams
        self.visualization_call = visualization_call
        self.device = device
        self.skip_first_log = True
        self.success_buf = []
        self.rollout_counter = 0
        self.rew_buf = 0

    def _on_step(self) -> bool:
        if self.hyperparams.visualization.do_visualization:
            if self.n_calls % self.hyperparams.visualization.interval == 0:
                self.visualization_call(model=self.model, device=self.device, param=self.param_to_vis,
                                        goal=self.goal_to_vis,
                                        param_history=self.visualization_history,
                                        hyperparams=self.hyperparams)
        if self.locals.get('infos') is not None:
            for info in self.locals['infos']:
                if info.get('success', False):
                    self.success_buf.append(1.0)
                else:
                    self.success_buf.append(0.0)

        self.rew_buf += self.locals.get('rewards')[0]
        self.rollout_counter += 1

        return True

    def _on_rollout_end(self) -> None:
            rollout_buf_mean_rew = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            success_rate = safe_mean(self.success_buf)
            mean_reward = self.rew_buf / self.rollout_counter
            self.custom_logger.log_rollout(mean_reward, success_rate,rollout_buf_mean_rew)

            # Reset success buffer
            self.success_buf = []
            self.rollout_counter=0
            self.rew_buf=0

            if not self.skip_first_log:
                self.custom_logger.log_train_rollout(self.model.logger.name_to_value['train/approx_kl'],
                                                     self.model.logger.name_to_value['train/clip_fraction'],
                                                     self.model.logger.name_to_value['train/clip_range'],
                                                     self.model.logger.name_to_value['train/entropy_loss'],
                                                     self.model.logger.name_to_value['train/explained_variance'],
                                                     self.model.logger.name_to_value['train/learning_rate'],
                                                     self.model.logger.name_to_value['train/loss'],
                                                     self.model.logger.name_to_value['train/n_updates'],
                                                     self.model.logger.name_to_value['train/policy_gradient_loss'],
                                                     self.model.logger.name_to_value['train/value_loss'],
                                                     self.model.logger.name_to_value['train/std'])
            else:
                self.skip_first_log = False
