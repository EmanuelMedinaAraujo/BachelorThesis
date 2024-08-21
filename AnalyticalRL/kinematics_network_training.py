import torch

from AnalyticalRL.kinematics_network import loss_fn
from Util.forward_kinematics import update_theta_values, calculate_distances


def train_loop(dataloader, model, optimizer, device, logger, epoch_num, error_tolerance):
    """
    The training loop for the kinematics network. This function trains the model on the given dataloader and
    logs the training loss and accuracy at the end of each epoch.

    The accuracy is calculated by counting the number of correct predictions. A prediction is considered correct if the
    distance between the predicted end effector position and the goal is less than the tolerable_accuracy_error.

    The loss is calculated as the sum of the distances between the predicted end effector position and the goal
     divided by the dataset_size.

    Args:
        dataloader: DataLoader object containing the training data
        model: The Kinematics Network
        optimizer: The optimizer used to update the model's parameters
        device: The device used for torch operations
        logger: The logger object used to log training data
        epoch_num: The current epoch number
        error_tolerance: The maximum distance between the predicted eef position and the goal that is considered correct
    """
    model.train()
    num_correct, loss_sum = 0, 0

    for batch, (param, goal) in enumerate(dataloader):
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

    dataset_size = len(dataloader.dataset)
    accuracy = (num_correct * 100 / dataset_size)
    logger.log_training(loss=loss_sum / dataset_size, epoch_num=epoch_num, accuracy=accuracy)
