from SimpleRL.kinematics_network import loss_fn
from Util.forward_kinematics import update_theta_values, calculate_distance


def train_loop(dataloader, model, optimizer, device, logger, epoch_num, error_tolerance):
    model.train()
    num_correct, loss_sum = 0, 0

    for batch, (param, goal) in enumerate(dataloader):
        param, goal = param.to(device), goal.to(device)
        pred = model((param, goal))

        # Update theta values with predictions
        updated_param = update_theta_values(param, pred)

        loss = loss_fn(param=updated_param, goal=goal)

        distances = calculate_distance(param=updated_param, goal=goal)
        loss_sum += distances.sum().item()
        num_correct += (distances <= error_tolerance).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    dataset_size = len(dataloader.dataset)
    # Increase num_correct for each value in distances that is less than 0.5
    accuracy = (num_correct * 100/ dataset_size)
    logger.log_training(loss=loss_sum/dataset_size, epoch_num=epoch_num, accuracy=accuracy)