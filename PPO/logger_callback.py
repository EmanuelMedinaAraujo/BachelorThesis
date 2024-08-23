from stable_baselines3.common.callbacks import BaseCallback


class LoggerCallback(BaseCallback):

    def __init__(self, logger, visualization_history, goal_to_vis, param_to_vis, hyperparams, visualization_call,
                 device,
                 verbose: int = 2):
        super().__init__(verbose)
        self.counter = 0
        self.custom_logger = logger
        self.visualization_history = visualization_history
        self.goal_to_vis = goal_to_vis
        self.param_to_vis = param_to_vis
        self.hyperparams = hyperparams
        self.visualization_call = visualization_call
        self.device = device
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:
        super()._on_step()
        if self.hyperparams.visualization.do_visualization:
            if self.counter % self.hyperparams.visualization.interval == 0:
                self.visualization_call(model=self.model, device=self.device, param=self.param_to_vis,
                                        goal=self.goal_to_vis,
                                        param_history=self.visualization_history,
                                        hyperparams=self.hyperparams)
        # print(f"Step: {self.counter}")
        self.counter += 1
        return True

    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()
        print("Rollout end")
        return

    def _on_training_end(self) -> None:
        super()._on_training_end()
        print("Training end")
        return

    def _on_training_start(self) -> None:
        super()._on_training_start()
        print("Training start")
        return

    def on_rollout_start(self) -> None:
        super().on_rollout_start()
        print("Rollout start")
        return
