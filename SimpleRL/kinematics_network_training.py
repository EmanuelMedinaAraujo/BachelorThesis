from SimpleRL.kinematics_network import loss_fn


def train_loop(dataloader, model, optimizer, batch_size, device):
    size = len(dataloader.dataset)
    model.train()
    loss_ges = 0
    for batch, (param, goal) in enumerate(dataloader):
        param, goal = param.to(device), goal.to(device)
        pred = model((param, goal))
        loss = loss_fn(param, pred, goal)
        loss_ges += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(param)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # if log_in_wandb:
    #    wandb.log({"acc": accuracy, "loss": loss_ges/size})
