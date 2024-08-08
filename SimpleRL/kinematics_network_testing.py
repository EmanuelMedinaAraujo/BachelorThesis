import torch

from Util.forward_kinematics import update_theta_values, calculate_eef_positions
from SimpleRL.kinematics_network import loss_fn

# The tolerance in accuracy that is still regarded as correct
TOLERABLE_ACCURACY_ERROR = 0.5

def test_loop(dataloader, model, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for param, goal in dataloader:
            param, goal = param.to(device), goal.to(device)
            pred = model((param, goal))
            test_loss += loss_fn(param, pred, goal).item()

            eef_positions = calculate_eef_positions(update_theta_values(param, pred))

            distances = torch.square(eef_positions - goal).sum(dim=1).sqrt()
            # Increase correct for each value in distances that is less than 0.5
            correct += (distances <= TOLERABLE_ACCURACY_ERROR).sum().item()

    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    print(f"Test Error: \n Accuracy: {accuracy :>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #if LOG_IN_WANDB:
     #   wandb.log({"acc": accuracy, "loss": test_loss})
        #torch.save(model, MODEL_SAVE_PATH)
