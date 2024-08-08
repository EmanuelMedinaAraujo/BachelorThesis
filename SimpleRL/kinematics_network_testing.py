import torch

from Util.forward_kinematics import update_theta_values, calculate_distance

def test_loop(dataloader, model, device, tolerable_accuracy_error, logger):
    model.eval()
    test_loss, num_correct = 0, 0

    with torch.no_grad():
        for param, goal in dataloader:
            param, goal = param.to(device), goal.to(device)

            pred = model((param, goal))

            # Update theta values with predictions
            updated_param = update_theta_values(parameters=param, new_theta_values=pred)

            distances = calculate_distance(param=updated_param, goal=goal)
            test_loss += distances.sum().item()

            # Increase num_correct for each value in distances that is less than 0.5
            num_correct += (distances <= tolerable_accuracy_error).sum().item()

    dataset_size = len(dataloader.dataset)
    test_loss /= dataset_size
    accuracy =  (num_correct * 100 / dataset_size)
    logger.log_test(accuracy=accuracy, loss=test_loss)
