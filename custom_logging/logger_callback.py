import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

from conf.config import TrainConfig
from util.forward_kinematics import calculate_distances
from vis.problem_vis import visualize_problem


class LoggerCallback(BaseCallback):

    def __init__(
        self,
        logger,
        visualization_history,
        goal_to_vis,
        param_to_vis,
        cfg: TrainConfig,
        device,
        test_dataloader,
        tolerable_accuracy_error,
        verbose: int = 2,
    ):
        super(LoggerCallback, self).__init__(verbose)
        self.custom_logger = logger
        self.visualization_history = visualization_history
        self.goal_to_vis = goal_to_vis
        self.param_to_vis = param_to_vis
        self.cfg = cfg
        self.device = device
        self.test_dataloader = test_dataloader

        self.skip_first_log = True

        self.success_buf = []
        self.rollout_counter = 0
        self.rew_buf = 0
        self.tolerable_accuracy_error = tolerable_accuracy_error
        self.last_vis_step = 0
        self.last_testing_step = 0

    def _on_step(self) -> bool:
        if self.cfg.do_vis:
            interval = (
                self.cfg.vis.analytical.interval
                if self.cfg.use_stb3
                else self.cfg.vis.stb3.interval
            )
            if self.num_timesteps - self.last_vis_step > interval:
                self.last_vis_step = self.num_timesteps
                visualize_problem(
                    model=self.model,
                    device=self.device,
                    param=self.param_to_vis,
                    goal=self.goal_to_vis,
                    param_history=self.visualization_history,
                    cfg=self.cfg,
                    logger=self.custom_logger,
                    current_step=self.num_timesteps,
                )

        reward = self.locals.get("rewards")[0]
        self.rew_buf += reward
        if reward > self.tolerable_accuracy_error:
            self.success_buf.append(1.0)
        else:
            self.success_buf.append(0.0)
        self.rollout_counter += 1

        # Evaluate the model
        testing_interval = (
            self.cfg.hyperparams.stb3.test_interval
            if self.cfg.use_stb3
            else self.cfg.hyperparams.analytical.testing_interval
        )
        if self.num_timesteps - self.last_testing_step > testing_interval:
            self.last_testing_step = self.num_timesteps
            with torch.no_grad():
                env = self.model.get_env()
                # Get goal and parameter from model environment
                old_goal = env.env_method("get_wrapper_attr", "goal")
                old_param = env.env_method("get_wrapper_attr", "parameter")

                accuracy, mean_reward = self.test_model(env)
                self.custom_logger.log_test(
                    accuracy, mean_reward, self.num_timesteps, self.num_timesteps
                )

                # Reset goal and parameter
                env.env_method("set_goal", old_goal[0])
                env.env_method("set_parameter", old_param[0])
        return True

    def _on_rollout_end(self) -> None:
        rollout_buf_mean_rew = safe_mean(
            [ep_info["r"] for ep_info in self.model.ep_info_buffer]
        )
        success_rate = safe_mean(self.success_buf)
        mean_reward = self.rew_buf / self.rollout_counter

        self.custom_logger.log_rollout(
            mean_reward, success_rate, rollout_buf_mean_rew, self.num_timesteps
        )

        # Reset success buffer
        self.success_buf = []
        self.rollout_counter = 0
        self.rew_buf = 0

        # Skip first log because it is not a full rollout
        if not self.skip_first_log:
            self.custom_logger.log_train_rollout(
                self.model.logger.name_to_value["train/approx_kl"],
                self.model.logger.name_to_value["train/clip_fraction"],
                self.model.logger.name_to_value["train/clip_range"],
                self.model.logger.name_to_value["train/entropy_loss"],
                self.model.logger.name_to_value["train/explained_variance"],
                self.model.logger.name_to_value["train/learning_rate"],
                self.model.logger.name_to_value["train/loss"],
                self.model.logger.name_to_value["train/n_updates"],
                self.model.logger.name_to_value["train/policy_gradient_loss"],
                self.model.logger.name_to_value["train/value_loss"],
                self.model.logger.name_to_value["train/std"],
                self.num_timesteps,
            )
        else:
            self.skip_first_log = False

    def test_model(self, env):
        counter_success = 0
        distance_sum = 0
        for param, goal in self.test_dataloader:
            # Set visualization goal and parameter to training_env
            env.env_method("set_goal", goal)
            env.env_method("set_parameter", param)

            observation = torch.concat([param.flatten(), goal]).detach().cpu().numpy()
            # cell and hidden state of the LSTM
            if self.cfg.hyperparams.stb3.use_recurrent_policy:
                # cell and hidden state of the LSTM
                lstm_states = None
                # Episode start signals are used to reset the lstm states
                episode_starts = np.ones((1,), dtype=bool)
                pred, _ = self.model.predict(
                    observation, state=lstm_states, episode_start=episode_starts
                )
            else:
                pred, _ = self.model.predict(observation)
            pred = torch.tensor(pred).to(self.device)

            # Evaluate prediction
            updated_param = torch.cat((param, pred.unsqueeze(1)), dim=-1)
            distance = calculate_distances(updated_param, goal).detach().item()
            distance_sum += distance
            if distance <= self.tolerable_accuracy_error:
                counter_success += 1
        accuracy = counter_success / len(self.test_dataloader)
        mean_reward = distance_sum / len(self.test_dataloader)
        return accuracy, mean_reward
