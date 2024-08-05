import torch
import wandb

from torch import nn
from torch.utils.data import DataLoader
from SimpleRL.ParameterDataset import CustomParameterDataset
from Util import dh_conventions
from Util.forward_kinematics import forward_kinematics


MODEL_SAVE_PATH = "ModelSaves/model_prototype1.pth"

log_in_wandb = True

learning_rate = 1e-3
dataset_length = 10000
batch_size = 64
epochs = 70
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# The tolerance in accuracy that is still regarded as correct
tolerable_accuracy_error = 0.5

max_accuracy = 0.0
max_epoch = -1
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, model_input):
        param, goal = model_input

        # Flatten the param
        # Expected output: flatten_param = torch.Size([64, 8]) and goal = torch.Size([64, 2])
        flatten_param = self.flatten(param)

        # Concatenate flatten_param and goal along the second dimension
        # Expected output: flatten_input shape should be torch.Size([64, 10])
        flatten_input = torch.cat((flatten_param, goal), dim=1)

        logits = self.linear_relu_stack(flatten_input)
        return logits

def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (param, goal) in enumerate(dataloader):
        param, goal = param.to(device), goal.to(device)
        pred = model((param, goal))
        loss = loss_fn(param, pred, goal)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(param)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, t):
    global max_accuracy
    global max_epoch
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for param, goal in dataloader:
            param, goal = param.to(device), goal.to(device)
            pred = model((param, goal))
            test_loss += loss_fn(param, pred, goal).item()

            adapted_param = param.clone()
            adapted_param[:, :, 3] = pred

            fk = forward_kinematics(dh_conventions.dh_to_homogeneous(adapted_param))
            eef_positions = fk.get_matrix()[..., :2, 3]

            distances = torch.square(eef_positions - goal).sum(dim=1).sqrt()
            # Increase correct for each value in distances that is less than 0.5
            correct += (distances <= tolerable_accuracy_error).sum().item()

    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    print(f"Test Error: \n Accuracy: {accuracy :>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if log_in_wandb:
        wandb.log({"acc": accuracy, "loss": test_loss})
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        max_epoch = t
        #torch.save(model, MODEL_SAVE_PATH)

def loss_fn(param, pred, goal):
    dh_param = param.clone()
    dh_param[:, :, 3] = pred

    fk = forward_kinematics(dh_conventions.dh_to_homogeneous(dh_param))
    eef_positions = fk.get_matrix()[..., :2, 3]

    # Calculate the mean distance between the eef position and the goal position
    return torch.square(eef_positions - goal).sum(dim=1).sqrt().mean()

def train_model():
    if log_in_wandb:
        init_wandb()

    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #model = torch.load(MODEL_SAVE_PATH)
    train_dataloader = DataLoader(CustomParameterDataset(length = dataset_length, device_to_use=device), batch_size, shuffle=True)
    test_dataloader = DataLoader(CustomParameterDataset(length = dataset_length,device_to_use=device), batch_size, shuffle=True)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer)
        test_loop(test_dataloader, model, t)
    print("The max accuracy is in Epoch {} with the value {}".format(max_epoch + 1, max_accuracy))
    #torch.save(model, MODEL_SAVE_PATH)

    if log_in_wandb:
        wandb.finish()
    print("Done!")


def init_wandb():
    wandb.require("core")
    wandb.init(
        # set the wandb project where this run will be logged
        project="bachelor-thesis-prototype-01",

        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "dataset": "Generated",
            "dataset_length": dataset_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "acc_error": tolerable_accuracy_error,
        },
        dir="WandBCache"
    )


train_model()