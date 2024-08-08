import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from SimpleRL.kinematics_network_testing import test_loop
from SimpleRL.kinematics_network_training import train_loop
from SimpleRL.parameter_dataset import CustomParameterDataset
from SimpleRL.kinematics_network import KinematicsNetwork

MODEL_SAVE_PATH = "ModelSaves/model_prototype1.pth"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_and_test_model(cfg: DictConfig):
    # if LOG_IN_WANDB:
    #    init_wandb()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} as device")

    model = KinematicsNetwork(cfg.hyperparams.number_of_joints).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparams.learning_rate)

    # model = torch.load(MODEL_SAVE_PATH)

    train_dataloader = DataLoader(CustomParameterDataset(length=cfg.hyperparams.dataset_length,
                                                         device_to_use=device,
                                                         num_of_joints=cfg.hyperparams.number_of_joints,
                                                         parameter_convention=cfg.hyperparams.parameter_convention,
                                                         min_len=cfg.hyperparams.min_link_length,
                                                         max_len=cfg.hyperparams.max_link_length),
                                  cfg.hyperparams.batch_size, shuffle=True)
    test_dataloader = DataLoader(CustomParameterDataset(length=cfg.hyperparams.dataset_length,
                                                        device_to_use=device,
                                                        num_of_joints=cfg.hyperparams.number_of_joints,
                                                        parameter_convention=cfg.hyperparams.parameter_convention,
                                                        min_len=cfg.hyperparams.min_link_length,
                                                        max_len=cfg.hyperparams.max_link_length),
                                 cfg.hyperparams.batch_size, shuffle=True)

    for epoch_num in range(cfg.hyperparams.epochs):
        print(f"Epoch {epoch_num + 1}\n-------------------------------")
        train_loop(dataloader=train_dataloader,
                   model=model,
                   optimizer=optimizer,
                   batch_size=cfg.hyperparams.batch_size,
                   device=device)
        if (epoch_num + 1) % cfg.hyperparams.testing_interval == 0:
            test_loop(dataloader=test_dataloader,
                      model=model,
                      device=device,
                      tolerable_accuracy_error=cfg.hyperparams.tolerable_accuracy_error)
        else:
            print("\n")
    # torch.save(model, MODEL_SAVE_PATH)

    # if LOG_IN_WANDB:
    #    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    train_and_test_model()
