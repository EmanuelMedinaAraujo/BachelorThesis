import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from Visualization.planar_robot_vis import visualize_planar_robot

from Logging.loggger import Logger, log_used_device
from SimpleRL.kinematics_network_testing import test_loop
from SimpleRL.kinematics_network_training import train_loop
from SimpleRL.parameter_dataset import CustomParameterDataset
from SimpleRL.kinematics_network import KinematicsNetwork

MODEL_SAVE_PATH = "ModelSaves/model_prototype1.pth"


def visualize_problem(model, param, goal, default_line_transparency, frame_size_scalar, default_line_width, device,
                      use_color_per_robot, standard_size, save_to_file, show_plot, show_joint_label):
    model.eval()

    with torch.no_grad():
        pred = model((param, goal))

        # Add a dimension to include theta values
        new_theta_values = pred[0].unsqueeze(-1)
        updated_param = torch.cat((param[0], new_theta_values), dim=-1)

        visualize_planar_robot(parameter=updated_param, goal=goal[0], device=device, standard_size=standard_size,
                               save_to_file=save_to_file,
                               default_line_transparency=default_line_transparency, frame_size_scalar=frame_size_scalar,
                               default_line_width=default_line_width, use_color_per_robot=use_color_per_robot,
                               show_plot=show_plot, show_joint_label=show_joint_label)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_and_test_model(cfg: DictConfig):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    log_used_device(device=device)

    hyperparams = cfg.hyperparams

    model = KinematicsNetwork(hyperparams.number_of_joints).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)

    # model = torch.load(MODEL_SAVE_PATH)

    train_dataloader = DataLoader(CustomParameterDataset(length=hyperparams.dataset_length,
                                                         device_to_use=device,
                                                         num_of_joints=hyperparams.number_of_joints,
                                                         parameter_convention=hyperparams.parameter_convention,
                                                         min_len=hyperparams.min_link_length,
                                                         max_len=hyperparams.max_link_length),
                                  hyperparams.batch_size, shuffle=True)
    test_dataloader = DataLoader(CustomParameterDataset(length=hyperparams.dataset_length,
                                                        device_to_use=device,
                                                        num_of_joints=hyperparams.number_of_joints,
                                                        parameter_convention=hyperparams.parameter_convention,
                                                        min_len=hyperparams.min_link_length,
                                                        max_len=hyperparams.max_link_length),
                                 hyperparams.batch_size, shuffle=True)

    logger = Logger(dataset_length=hyperparams.dataset_length, log_in_wandb=hyperparams.log_in_wandb, cfg=cfg)

    visualization_param, visualization_goal = next(iter(test_dataloader))

    for epoch_num in range(hyperparams.epochs):
        train_loop(dataloader=train_dataloader,
                   model=model,
                   optimizer=optimizer,
                   device=device,
                   logger=logger,
                   epoch_num=epoch_num,
                   error_tolerance=hyperparams.tolerable_accuracy_error)
        if (epoch_num + 1) % hyperparams.testing_interval == 0:
            test_loop(dataloader=test_dataloader,
                      model=model,
                      device=device,
                      logger=logger,
                      tolerable_accuracy_error=hyperparams.tolerable_accuracy_error)
        vis_params = hyperparams.visualization
        if vis_params.do_visualization and epoch_num % vis_params.interval == 0:
            visualize_problem(model=model, device=device, param=visualization_param, goal=visualization_goal,
                              default_line_transparency=vis_params.default_line_transparency,
                              frame_size_scalar=vis_params.frame_size_scalar,
                              default_line_width=vis_params.default_line_width,
                              use_color_per_robot=vis_params.use_color_per_robot,
                              standard_size=vis_params.standard_size,
                              save_to_file=vis_params.save_to_file,
                              show_plot=vis_params.show_plot,
                              show_joint_label=vis_params.show_joint_label)

    # torch.save(model, MODEL_SAVE_PATH)
    print("Done!")


if __name__ == "__main__":
    train_and_test_model()
