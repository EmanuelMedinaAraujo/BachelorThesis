import torch

from analyticalRL.networks.kinematics_network_base_class import KinematicsNetworkBase


def test_loop(test_dataset, model:KinematicsNetworkBase, tolerable_accuracy_error, logger, epoche_num, is_normal_output):
    """
    Tests the model on the given dataset and logs the accuracy and loss with the given logger.

    The accuracy is calculated by counting the number of correct predictions. A prediction is considered correct if the
    distance between the predicted end effector position and the goal is less than the tolerable_accuracy_error.

    The loss is calculated as the sum of the distances between the predicted end effector position and the goal
    divided by the dataset_size.

    Args:
        test_dataset: test data set
        model: The Kinematics Network
        tolerable_accuracy_error: The maximum distance between the predicted end effector position and the goal
                                         that is considered as a correct prediction
        logger: The logger object used for custom_logging
        epoche_num: The current epoch number
        is_normal_output: True if the model outputs normal angles, False if it outputs distribution parameters
    """
    model.eval()
    test_loss, num_correct = 0, 0
    with torch.no_grad():
        for param, goal, ground_truth in test_dataset:
            pred = model((param, goal))

            loss = model.loss_fn(param=param, pred=pred, goal=goal, ground_truth=ground_truth)
            test_loss += loss.sum().item()
            if is_normal_output:
                distances = model.calc_distances(param=param, pred=pred, goal=goal)
                num_correct += torch.le(distances, tolerable_accuracy_error).int().sum().item()

    dataset_size = len(test_dataset)
    test_loss /= dataset_size
    accuracy = None
    if is_normal_output:
        accuracy = num_correct * 100 / dataset_size
    logger.log_test(accuracy=accuracy, loss=test_loss, current_step=epoche_num)
    model.train()
