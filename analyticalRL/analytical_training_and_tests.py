import os
import sys
from datetime import datetime

import optuna
from tqdm import tqdm

from analyticalRL.networks.distributions.one_peak_distributions.beta_rsample_dist_network import *  # noqa
from analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.ground_truth_loss_network import *  # noqa
from analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.manual_reparam_network import *  # noqa
from analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.mu_distance_loss_network import *  # noqa
from analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.rsample_network import *  # noqa
from analyticalRL.networks.distributions.two_peak_distributions.two_peak_norm_dist_lstm_network import *  # noqa
from analyticalRL.networks.distributions.two_peak_distributions.two_peak_norm_dist_lstm_network_variant import *  # noqa
from analyticalRL.networks.distributions.two_peak_distributions.two_peak_norm_dist_network import *  # noqa
from analyticalRL.networks.simple_kinematics_network import *  # noqa
from conf.conf_dataclasses.config import TrainConfig
from data_generation.goal_generator import generate_achievable_goal
from data_generation.parameter_generator import ParameterGeneratorForPlanarRobot
from vis.model_type_vis.analytical_vis import visualize_analytical_problem, visualize_analytical_distribution


def create_model(device, cfg: TrainConfig, logger):
    model_name = cfg.hyperparams.analytical.output_type
    model_class = globals().get(model_name)
    if model_class:
        return model_class(
            num_joints=cfg.number_of_joints,
            num_layer=cfg.hyperparams.analytical.num_hidden_layer,
            layer_sizes=cfg.hyperparams.analytical.hidden_layer_sizes,
            logger=logger,
            error_tolerance=cfg.tolerable_accuracy_error,
        ).to(device)
    else:
        raise ValueError(
            f"Unknown output type: {cfg.hyperparams.analytical.output_type}. Please adjust the hyperparams config."
        )


def do_analytical_learning(device, cfg: TrainConfig, logger, test_dataset, visualization_history, visualization_goals,
                           visualization_params, visualization_ground_truth, tensor_type, trial: optuna.Trial = None,
                           ):
    model = create_model(device, cfg, logger)
    # Use optimizer specified in the config
    optimizer = getattr(torch.optim, cfg.hyperparams.analytical.optimizer)(
        model.parameters(), lr=cfg.hyperparams.analytical.learning_rate,  # maximize=True,
    )

    # Create Problem Generator
    problem_generator = ParameterGeneratorForPlanarRobot(
        batch_size=cfg.hyperparams.analytical.batch_size,
        device=device,
        tensor_type=tensor_type,
        num_joints=cfg.number_of_joints,
        parameter_convention=cfg.parameter_convention,
        min_len=cfg.min_link_length,
        max_len=cfg.max_link_length,
    )
    last_mean_loss = None
    if cfg.do_vis:
        visualize_analytical_model(cfg, device, 0, logger, model, visualization_goals,
                                   visualization_ground_truth, visualization_history, visualization_params)
    for epoch_num in tqdm(
            range(cfg.hyperparams.analytical.epochs),
            colour="green",
            file=sys.stdout
    ):
        # Keep track of the last mean loss for optuna
        last_mean_loss = train_loop(
            model=model,
            optimizer=optimizer,
            problem_generator=problem_generator,
            problems_per_epoch=cfg.hyperparams.analytical.problems_per_epoch,
            batch_size=cfg.hyperparams.analytical.batch_size,
            device=device,
            logger=logger,
            epoch_num=epoch_num,
        )

        # Test the model every hyperparams.testing_interval epochs
        if epoch_num % cfg.hyperparams.analytical.testing_interval == 0:
            test_loop(
                test_dataset=test_dataset,
                model=model,
                logger=logger,
                num_epoch=epoch_num,
            )

        # Visualize the same problem every hyperparams.visualization.interval epochs
        if cfg.do_vis and epoch_num % cfg.vis.analytical.interval == 0 and epoch_num != 0:
            visualize_analytical_model(cfg, device, epoch_num, logger, model, visualization_goals,
                                       visualization_ground_truth, visualization_history, visualization_params)
        if trial is not None:
            trial.report(last_mean_loss, epoch_num)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if cfg.save_trained_model:
        # Get date and time from system as string
        date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not os.path.isdir(cfg.model_save_dir):
            os.makedirs(cfg.model_save_dir)
        path = cfg.model_save_dir + "/" + cfg.hyperparams.analytical.output_type + "_" + date_time_string + "_model.pth"
        torch.save(model, path)
        logger.upload_model(path=path)
    return last_mean_loss


def test_loop(test_dataset, model: KinematicsNetworkBase, logger, num_epoch):
    """
    Tests the model on the given dataset and logs the accuracy and loss with the given logger.

    The accuracy is calculated by counting the number of correct predictions. A prediction is considered correct if the
    distance between the predicted end effector position and the goal is less than the tolerable_accuracy_error.

    The loss is calculated as the sum of the distances between the predicted end effector position and the goal
    divided by the dataset_size.

    Args:
        test_dataset: test data set
        model: The Kinematics Network
        logger: The logger object used for custom_logging
        num_epoch: The current epoch number
    """
    model.eval()
    loss_sum, num_correct = 0, 0
    with torch.no_grad():
        for param, goal, ground_truth in test_dataset:
            loss, loss_sum, num_correct = eval_model(goal=goal,
                                                     ground_truth=ground_truth,
                                                     loss_sum=loss_sum,
                                                     model=model,
                                                     num_correct=num_correct,
                                                     param=param)

    dataset_size = len(test_dataset)
    loss_sum /= dataset_size
    accuracy = num_correct * 100 / dataset_size
    logger.log_test(accuracy=accuracy, loss=loss_sum, current_step=num_epoch)
    model.train()


def train_loop(model: KinematicsNetworkBase, optimizer, problem_generator, problems_per_epoch, batch_size, device,
               logger, epoch_num):
    """
    The training loop for the kinematics network. This function trains the model on random parameters and goals and
    logs the training loss and accuracy at the end of each epoch.

    The accuracy is calculated by counting the number of correct predictions. A prediction is considered correct if the
    distance between the predicted end effector position and the goal is less than the tolerable_accuracy_error.

    The loss is calculated as the sum of the distances between the predicted end effector position and the goal
     divided by the dataset_size.

    Args:
        model: The Kinematics Network
        optimizer: The optimizer used to update the model's parameters
        problem_generator: The problem generator used to generate random parameters with the correct batch size
        batch_size: The batch size used for training
        problems_per_epoch: The number of problems to solve in each epoch
        device: The device used for torch operations
        logger: The logger object used to log training data
        epoch_num: The current epoch number
    """
    model.train()
    num_correct, loss_sum = 0, 0
    num_problems = problems_per_epoch // batch_size
    for _ in range(num_problems):
        # Generate random parameters and goals
        param = problem_generator.get_random_parameters()
        goal, ground_truth = generate_achievable_goal(param, device)

        param, goal, ground_truth = param.to(device), goal.to(device), ground_truth.to(device)
        loss, loss_sum, num_correct = eval_model(goal=goal,
                                                 ground_truth=ground_truth,
                                                 loss_sum=loss_sum,
                                                 model=model,
                                                 num_correct=num_correct,
                                                 param=param)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = num_correct * 100 / problems_per_epoch
    logger.log_training(
        loss=loss_sum / num_problems, epoch_num=epoch_num, accuracy=accuracy
    )
    return loss_sum / num_problems


def eval_model(goal, ground_truth, loss_sum, model, num_correct, param):
    predictions = model((param, goal))
    loss, correct_predictions = model.loss_fn(param=param, pred=predictions, goal=goal, ground_truth=ground_truth)
    loss_sum += loss.sum().item()
    num_correct += correct_predictions
    return loss, loss_sum, num_correct


def visualize_analytical_model(cfg, device, epoch_num, logger, model, visualization_goals, visualization_ground_truth,
                               visualization_history, visualization_params):
    with torch.no_grad():
        for i in range(cfg.vis.num_problems_to_visualize):
            if cfg.hyperparams.analytical.output_type == "Normal":
                visualize_analytical_problem(
                    model=model,
                    param=visualization_params[i],
                    goal=visualization_goals[i],
                    param_history=visualization_history,
                    cfg=cfg,
                    logger=logger,
                    current_step=epoch_num,
                    chart_index=i + 1,
                )
            else:
                visualize_analytical_distribution(
                    model=model,
                    param=visualization_params[i],
                    goal=visualization_goals[i],
                    ground_truth=visualization_ground_truth[i],
                    cfg=cfg,
                    logger=logger,
                    current_step=epoch_num,
                    chart_index=i + 1,
                    device=device,
                )
