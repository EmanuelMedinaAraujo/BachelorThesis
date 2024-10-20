import torch

from analyticalRL.networks.kinematics_network_base_class import KinematicsNetworkBase
from data_generation.goal_generator import generate_achievable_goal


def test_loop(test_dataset, model: KinematicsNetworkBase, tolerable_accuracy_error, logger, num_epoch,
              is_normal_output):
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
        num_epoch: The current epoch number
        is_normal_output: True if the model outputs normal angles, False if it outputs distribution parameters
    """
    model.eval()
    test_loss, num_correct = 0, 0
    with torch.no_grad():
        for param, goal, ground_truth in test_dataset:
            loss, loss_sum, num_correct = eval_model(tolerable_accuracy_error,
                                                     goal,
                                                     ground_truth,
                                                     is_normal_output,
                                                     loss_sum,
                                                     model,
                                                     num_correct, param)

    dataset_size = len(test_dataset)
    test_loss /= dataset_size
    accuracy = None
    if is_normal_output:
        accuracy = num_correct * 100 / dataset_size
    logger.log_test(accuracy=accuracy, loss=test_loss, current_step=num_epoch)
    model.train()


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
    num_problems = problems_per_epoch // batch_size
    for _ in range(num_problems):
        # Generate random parameters and goals
        param = problem_generator.get_random_parameters()
        goal, ground_truth = generate_achievable_goal(param, device)

        param, goal, ground_truth = param.to(device), goal.to(device), ground_truth.to(device)
        loss, loss_sum, num_correct = eval_model(error_tolerance,
                                                 goal,
                                                 ground_truth,
                                                 is_normal_output,
                                                 loss_sum,
                                                 model,
                                                 num_correct,
                                                 param)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if is_normal_output:
        accuracy = num_correct * 100 / problems_per_epoch
        logger.log_training(
            loss=loss_sum / num_problems, epoch_num=epoch_num, accuracy=accuracy
        )
    else:
        logger.log_training(
            loss=loss_sum / num_problems, epoch_num=epoch_num, accuracy=None
        )
    return loss_sum / num_problems


def eval_model(error_tolerance, goal, ground_truth, is_normal_output, loss_sum, model, num_correct, param):
    pred = model((param, goal))
    loss = model.loss_fn(param=param, pred=pred, goal=goal, ground_truth=ground_truth)
    loss_sum += loss.sum().item()
    if is_normal_output:
        distances = model.calc_distances(param=param, angles_pred=pred, goal=goal)
        # Increase num_correct for each value in distances that is less than error_tolerance
        num_correct += torch.le(distances, error_tolerance).int().sum().item()
    return loss, loss_sum, num_correct
