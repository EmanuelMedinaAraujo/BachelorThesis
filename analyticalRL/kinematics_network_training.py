import torch

from analyticalRL.kinematics_network_base import KinematicsNetworkBase
from data_generation.goal_generator import generate_achievable_goal


def train_loop(model: KinematicsNetworkBase, optimizer, problem_generator, problems_per_epoch, batch_size, device,
               logger, epoch_num, error_tolerance,
               is_normal_output):
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
        error_tolerance: The maximum distance between the predicted eef position and the goal that is considered correct
        is_normal_output: True if the model outputs normal angles, False if it outputs distribution parameters
    """
    model.train()
    num_correct, loss_sum = 0, 0

    for _ in range(problems_per_epoch // batch_size):
        # Generate random parameters and goals
        param = problem_generator.get_random_parameters()
        goal, ground_truth = generate_achievable_goal(param, device)

        param, goal, ground_truth = param.to(device), goal.to(device), ground_truth.to(device)
        pred = model((param, goal))

        loss = model.loss_fn(param=param, pred=pred, goal=goal, ground_truth=ground_truth)

        loss_sum += loss.sum().item()
        if is_normal_output:
            distances = model.calc_distances(param=param, pred=pred, goal=goal)
            # Increase num_correct for each value in distances that is less than error_tolerance
            num_correct += torch.le(distances, error_tolerance).int().sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if is_normal_output:
        accuracy = num_correct * 100 / problems_per_epoch
        logger.log_training(
            loss=loss_sum / problems_per_epoch, epoch_num=epoch_num, accuracy=accuracy
        )
    else:
        logger.log_training(
            loss=loss_sum / problems_per_epoch, epoch_num=epoch_num, accuracy=None
        )
    return loss_sum / problems_per_epoch
