import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from SimpleRL.ParameterDataset import CustomParameterDataset
from DataGeneration.util import dh_conventions

max_accuracy = 0.0
max_epoche = -1
learning_rate = 1e-3
batch_size = 64
epochs = 40

# DH Parameter of 2 Link Planar Robot with extended arm (alpha, a, d, theta)
DH_EXAMPLE = torch.tensor([
    [0, 15, 0, 0],
    [0, 10, 0, 0]
])

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


train_dataloader = DataLoader(CustomParameterDataset(), batch_size=64, shuffle=True)
test_dataloader = DataLoader(CustomParameterDataset(), batch_size=64, shuffle=True)

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

    def forward(self, input):
        param, goal = input

        # Flatten the param
        flattenParam = self.flatten(param)
        # Expected output: flattenParam = torch.Size([64, 8]) and goal = torch.Size([64, 2])

        # Concatenate flattenParam and goal along the second dimension (dim=1)
        flattenInput = torch.cat((flattenParam, goal), dim=1)
        # Expected output: flattenInput shape should be torch.Size([64, 10])

        logits = self.linear_relu_stack(flattenInput)
        return logits
    
model = NeuralNetwork().to(device)
    

def train_loop(dataloader, model, loss_fn, optimizer):
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

def test_loop(dataloader, model, t, loss_fn):
    global max_accuracy
    global max_epoche
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for param, goal in dataloader:
            param, goal = param.to(device), goal.to(device)
            pred = model((param, goal))
            test_loss += loss_fn(param, pred, goal).item()

            adaptedParam = param.clone()
            adaptedParam[:, :, 3] = pred
            for i in range(param.shape[0]):
                end_effector_pos = calculate_2D_end_effector_position(adaptedParam[i])
                if torch.square(end_effector_pos - goal[i]).sum().sqrt() < 0.5 :
                    correct += 1

    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if(accuracy > max_accuracy):
        max_accuracy = accuracy
        max_epoche = t
        #torch.save(model, "Models/model_prototype1.pth")

def loss_fn(param, pred, goal):
    batch_size = param.shape[0]
    dh_param = param.clone()
    dh_param[:, :, 3] = pred

    end_effector_positions = []
    for i in range(batch_size):
        end_effector_positions.append(calculate_2D_end_effector_position(dh_param[i]))
    
    end_effector_positions = torch.stack(end_effector_positions)
    # Calculate for each row the mse loss and return the mean

    distance_loss = torch.zeros(end_effector_positions.shape[0], device=device)
    for i in range(end_effector_positions.shape[0]):
        distance_loss[i] = torch.square(end_effector_positions[i] - goal[i]).sum().sqrt()
    return distance_loss.mean()


def calculate_2D_end_effector_position(dh_param):
    forwardKinematics = dh_conventions.dh_to_homogeneous(dh_param)
    final_transform = torch.eye(4, device=device)
    for i in range(forwardKinematics.shape[0]):
        final_transform = torch.matmul(final_transform, forwardKinematics[i])

    end_effector_position_3D = final_transform[:3, 3]

    return end_effector_position_3D[:2]


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#model = torch.load("Models/model_prototype1.pth")
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, t, loss_fn)
print("The max accuracy is in Epoche {} with the value {}".format(max_epoche+1, max_accuracy))
#torch.save(model, "Models/model_prototype1.pth")

print("Done!")
