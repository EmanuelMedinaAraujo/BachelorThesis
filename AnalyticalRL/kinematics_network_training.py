import torch

from AnalyticalRL.kinematics_network import loss_fn
from DataGeneration.goal_generator import generate_achievable_goal
from Util.forward_kinematics import update_theta_values, calculate_distances


def train_loop(model, optimizer, problem_generator, problems_per_epoch, batch_size, device, logger, epoch_num, error_tolerance):
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
    """
    model.train()
    num_correct, loss_sum = 0, 0

    for _ in range(problems_per_epoch // batch_size):
        # Generate random parameters and goals
        param = problem_generator.get_random_parameters()
        goal = generate_achievable_goal(param, device)

        param, goal = param.to(device), goal.to(device)
        pred = model((param, goal))

        # Update theta values with predictions
        updated_param = update_theta_values(parameters=param, new_theta_values=pred)

        loss = loss_fn(param=updated_param, goal=goal)

        distances = calculate_distances(param=updated_param, goal=goal)
        loss_sum += distances.sum().item()
        # Increase num_correct for each value in distances that is less than error_tolerance
        num_correct += torch.le(distances, error_tolerance).int().sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = (num_correct * 100 / problems_per_epoch)
    logger.log_training(loss=loss_sum / problems_per_epoch, epoch_num=epoch_num, accuracy=accuracy)
