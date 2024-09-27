import torch

from analyticalRL.kinematics_network import KinematicsNetwork
from util.forward_kinematics import update_theta_values, calculate_parameter_goal_distances


def test_loop(test_dataset, model:KinematicsNetwork, device, tolerable_accuracy_error, logger, epoche_num):
    """
    Tests the model on the given dataset and logs the accuracy and loss with the given logger.

    The accuracy is calculated by counting the number of correct predictions. A prediction is considered correct if the
    distance between the predicted end effector position and the goal is less than the tolerable_accuracy_error.

    The loss is calculated as the sum of the distances between the predicted end effector position and the goal
    divided by the dataset_size.

    Args:
        test_dataset: test data set
        model: The Kinematics Network
        device: The device used for torch operations
        tolerable_accuracy_error: The maximum distance between the predicted end effector position and the goal
                                         that is considered as a correct prediction
        logger: The logger object used for custom_logging
    """
    model.eval()
    test_loss, num_correct = 0, 0

    with torch.no_grad():
        for param, goal in test_dataset:
            param, goal = param.to(device), goal.to(device)

            pred = model((param, goal))

            # Update theta values with predictions from model
            updated_param = update_theta_values(parameters=param, new_theta_values=pred)

            distances = calculate_parameter_goal_distances(param=updated_param, goal=goal)
            test_loss += distances.sum().item()
            # Increase num_correct for each distance that is less than the tolerable_accuracy_error
            num_correct += (
                torch.le(distances, tolerable_accuracy_error).int().sum().item()
            )

    dataset_size = len(test_dataset)
    test_loss /= dataset_size
    accuracy = num_correct * 100 / dataset_size
    logger.log_test(accuracy=accuracy, loss=test_loss, current_step=epoche_num)
