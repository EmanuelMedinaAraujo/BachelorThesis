import torch
from torch import nn
from torch.utils.data import DataLoader
from SimpleRL.ParameterDataset import CustomParameterDataset
from DataGeneration.util import dh_conventions

MODEL_SAVE_PATH = "ModelSaves/model_prototype1.pth"

learning_rate = 1e-3
batch_size = 64
epochs = 40
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

max_accuracy = 0.0
max_epoch = -1

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
            for i in range(param.shape[0]):
                end_effector_pos = calculate_2d_end_effector_position(adapted_param[i])
                if torch.square(end_effector_pos - goal[i]).sum().sqrt() < 0.5 :
                    correct += 1

    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    print(f"Test Error: \n Accuracy: {accuracy :>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        max_epoch = t
        #torch.save(model, MODEL_SAVE_PATH)

def loss_fn(param, pred, goal):
    dh_param = param.clone()
    dh_param[:, :, 3] = pred

    end_effector_positions = []
    for i in range(batch_size):
        end_effector_positions.append(calculate_2d_end_effector_position(dh_param[i]))
    
    end_effector_positions = torch.stack(end_effector_positions)

    # Calculate for each row the mse loss and return the mean
    distance_loss = torch.zeros(end_effector_positions.shape[0], device=device)
    for i in range(end_effector_positions.shape[0]):
        distance_loss[i] = torch.square(end_effector_positions[i] - goal[i]).sum().sqrt()
    return distance_loss.mean()


def calculate_2d_end_effector_position(dh_param):
    forward_kinematics = dh_conventions.dh_to_homogeneous(dh_param)
    final_transform = torch.eye(4, device=device)
    for i in range(forward_kinematics.shape[0]):
        final_transform = torch.matmul(final_transform, forward_kinematics[i])

    end_effector_position_3d = final_transform[:3, 3]

    return end_effector_position_3d[:2]

def train_model():
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #model = torch.load(MODEL_SAVE_PATH)
    train_dataloader = DataLoader(CustomParameterDataset(), batch_size, shuffle=True)
    test_dataloader = DataLoader(CustomParameterDataset(), batch_size, shuffle=True)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer)
        test_loop(test_dataloader, model, t)
    print("The max accuracy is in Epoch {} with the value {}".format(max_epoch + 1, max_accuracy))
    #torch.save(model, MODEL_SAVE_PATH)

    print("Done!")
