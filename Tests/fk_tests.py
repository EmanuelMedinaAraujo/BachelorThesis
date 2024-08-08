import torch
from Util.dh_conventions import dh_to_homogeneous
from Util.forward_kinematics import forward_kinematics

dh_param = torch.tensor( [[[ 0.0000, 13.2026,  0.0000, -0.3302],
        [ 0.0000,  9.6462,  0.0000, -0.0560]],

        [[ 0.0000, 13.,  0.0000, -0.0],
        [ 0.0000,  9.,  0.0000, -0.0]]
        ])
forward_kinematics_matrix = dh_to_homogeneous(dh_param)

expected_result = [21.4253, -7.9135]
fk_calc = forward_kinematics(forward_kinematics_matrix).get_matrix()
end_effector = fk_calc[..., :2, 3]
x, y = end_effector[0]
print(f"x: {x}, y: {y}")
x1, y1 = end_effector[1]
print(f"x: {x1}, y: {y1}")
print(fk_calc)