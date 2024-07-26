from torch.utils.data import DataLoader
from SimpleRL.ParameterDataset import CustomParameterDataset

train_dataloader = DataLoader(CustomParameterDataset(), batch_size=64, shuffle=True)
test_dataloader = DataLoader(CustomParameterDataset(), batch_size=64, shuffle=True)

# train_input, train_solution = next(iter(train_dataloader))
# dh_param, goal_coordinate = train_input[0]
# solution = train_solution[0]
# print(f"dhParam shape: {dh_param.size()}")
# print(f"goal end effector shape: {goal_coordinate.size()}")
# print(f"dhParam: {dh_param}")
# print(f"goal end effector: {goal_coordinate}")
# print(f"solution: {solution}")

train_param, train_goal = next(iter(train_dataloader))
dh_param = train_param[0]
goal_coordinate = train_goal[0]
print(f"dhParam shape: {dh_param.size()}")
print(f"goal end effector shape: {goal_coordinate.size()}")
print(f"dhParam: {dh_param}")
print(f"goal end effector: {goal_coordinate}")
