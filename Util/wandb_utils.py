import wandb


def init_wandb(learning_rate, dataset_length, epochs, batch_size, tolerable_accuracy_error, project_name):
    wandb.require("core")
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,

        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "dataset": "Generated",
            "dataset_length": dataset_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "acc_error": tolerable_accuracy_error,
        }
    )
