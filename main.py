import torch
from torch.utils.data import DataLoader
from SimpleRL.kinematics_network_testing import TOLERABLE_ACCURACY_ERROR, test_loop
from SimpleRL.kinematics_network_training import train_loop
from SimpleRL.parameter_dataset import CustomParameterDataset
from kinematics_network import KinematicsNetwork

MODEL_SAVE_PATH = "ModelSaves/model_prototype1.pth"

LOG_IN_WANDB = False

LEARNING_RATE = 1e-3
DATASET_LENGTH = 10000
BATCH_SIZE = 64
EPOCHS = 50
# Number of epochs to be run after which a test run is performed
INTERVAL_OF_TESTING = EPOCHS * 0.1

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

num_joints = 2

def train_model():
    #if LOG_IN_WANDB:
    #    init_wandb()

    model = KinematicsNetwork(num_joints).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #model = torch.load(MODEL_SAVE_PATH)
    train_dataloader = DataLoader(CustomParameterDataset(length = DATASET_LENGTH, device_to_use=DEVICE, num_of_joints=num_joints), BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(CustomParameterDataset(length = DATASET_LENGTH, device_to_use=DEVICE, num_of_joints=num_joints), BATCH_SIZE, shuffle=True)
    for epoch_num in range(EPOCHS):
        print(f"Epoch {epoch_num+1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer, batch_size= BATCH_SIZE, device=DEVICE)
        if (epoch_num + 1) % INTERVAL_OF_TESTING == 0:
            test_loop(test_dataloader, model, device=DEVICE)
        else:
            print("\n")
    #torch.save(model, MODEL_SAVE_PATH)

    #if LOG_IN_WANDB:
    #    wandb.finish()
    print("Done!")


train_model()