import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *

# 规划臂足协同动作，扭身子，同时手臂末端在世界系下固定位置不动
# 输出的文件为一个状态矩阵txt文件，列数为49列，因为读取文件的固定格式为49列
# root位置[0:3]，root姿态[3:7] [x,y,z,w]，线速度[7:10]，角速度[10:13]
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]，panda7没看过是什么顺序
# 关节角度和关节[25:37]


# 质心保持不动，机身x方向转动在yz平面投影是一个圆
# 实例化panda7
# panda7的关节上下限
panda_lb = [-0.87, -1.78, -2.53, -0.69, -1.78, -2.53, -0.87, -1.3, -2.53, -0.69, -1.3, -2.53]
panda_ub = [0.69, 3.4, -0.45, 0.87, 3.4, -0.45, 0.69, 4, -0.45, 0.87, 4, -0.45]
panda_toe_pos_init = [0.300133, -0.287854, -0.481828, 0.300133, 0.287854, -0.481828, -0.349867,
                      -0.287854, -0.481828, -0.349867, 0.287854, -0.481828]
panda7 = utils.QuadrupedRobot(l=0.65, w=0.225, l1=0.126375, l2=0.34, l3=0.34,
                              lb=panda_lb, ub=panda_ub, toe_pos_init=panda_toe_pos_init)
num_row = 50
num_col = 72
fps = 50

ref = np.ones((num_row - 1, num_col))
root_pos = np.zeros((num_row, 3))
root_rot = np.zeros((num_row, 4))
root_lin_vel = np.zeros((num_row - 1, 3))
root_ang_vel = np.zeros((num_row - 1, 3))
root_rot_dot = np.zeros((num_row - 1, 4))
toe_pos = np.zeros((num_row, 12))
dof_pos = np.zeros((num_row, 12))
dof_vel = np.zeros((num_row - 1, 12))
arm_pos = np.zeros((num_row, 3))
arm_rot = np.zeros((num_row, 4))
arm_dof_pos = np.zeros((num_row, 8))
arm_dof_vel = np.zeros((num_row - 1, 8))

# 质心轨迹
root_pos[:, 2] = 0.55
# 质心线速度 默认为0

# 姿态
q0 = [0, 0, 0, 1]  # [x,y,z,w]
q1 = [0, 0, np.sin(np.pi / 45), np.cos(np.pi / 45)]  # 绕z轴转5°

root_rot[:] = q0
interval = 20
start = 5
end = start + interval

for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q0, q1, frac)

start = end
end = start + interval
for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q1, q0, frac)

# 四元数的导数
for i in range(num_row - 1):
    root_rot_dot[i, :] = (root_rot[i + 1, :] - root_rot[i, :]) * fps
# 质心角速度
for i in range(num_row - 1):
    root_ang_vel[i, :] = 2 * utils.quat2angvel_map(root_rot[i, :]) @ root_rot_dot[i, :]


# 计算关节角度
# 获取足端初始位置，初始位置在世界系下保持不变，需要计算质心系的足端位置
toe_pos[:] = panda7.toe_pos_init
def z_1(t):
    # z方向轨迹，0.4s
    x = 0.2  # 对称轴0.2s
    h = 0.1
    return h - 5 / 2 * (t - x) ** 2


pos1 = [0, 0, 0]
pos2 = [0.1, -0.1, 0]
temp_pos_1= np.linspace(pos1, pos2, 20) #伸腿
temp_pos_2= np.linspace(pos2, pos1, 20) #收腿
for i in range(20):
    t = i/fps
    temp_pos_1[i, 2] = z_1(t)
    temp_pos_2[i, 2] = z_1(t)

toe_pos[5:25, :3] += temp_pos_1
toe_pos[5:25, 6:9] -= temp_pos_1
toe_pos[25:45, :3] += temp_pos_2
toe_pos[25:45, 6:9] -= temp_pos_2

# 计算质心系下的足端轨迹
for i in range(toe_pos.shape[0]):
    toe_pos[i, :3] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ toe_pos[i, :3]
    toe_pos[i, 3:6] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ toe_pos[i, 3:6]
    toe_pos[i, 6:9] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ toe_pos[i, 6:9]
    toe_pos[i, 9:12] = np.transpose(utils.quaternion2rotm(root_rot[i, :])) @ toe_pos[i, 9:12]

# go2的关节上下限
q = SX.sym('q', 3, 1)
for j in range(4):
    for i in range(num_row):
        # toe_pos是质心系的足端位置，所以rpy以及质心位置都是[0 0 0]
        # 这里不把上面的部分合起来放到一个式子里是因为下面使用casadi，上面使用了numpy，混合运算会出问题
        pos = (casadi.SX(casadi.DM(utils.quaternion2rotm(root_rot[i, :])).full()) @
               (panda7.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ panda7.toe)[:3])
        cost = 500 * dot((toe_pos[i, 3 * j:3 * j + 3] - pos[:3]), (toe_pos[i, 3 * j:3 * j + 3] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = nlpsol('S', 'ipopt', nlp)
        r = S(x0=[0.1, 0.8, -1.5], lbx=panda7.lb[3 * j:3 * j + 3], ubx=panda7.ub[3 * j:3 * j + 3])
        q_opt = r['x']
        dof_pos[i, 3 * j:3 * j + 3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i, :] = (dof_pos[i + 1, :] - dof_pos[i, :]) * fps

# arm fk，获取一个世界系下的位置，后续机器人固定在这个位置
robot_arm_rot, robot_arm_pos = utils.arm_fk([0, -0.3, 0.6, -0.3, 0, 0])
# 优化求解机械臂关节角度
q_arm = SX.sym('q', 5, 1)
# 机械臂关节角度限制
arm_ub = [3.14, 0.28, 3.14, 1.57, 1.57]
arm_lb = [-3.14, -3.6, 0, -1.57, -1.57]
for i in range(num_row):
    _, pos = utils.arm_fk([q_arm[0], q_arm[1], q_arm[2], q_arm[3], q_arm[4], 0])
    pos = casadi.SX(casadi.DM(utils.quaternion2rotm(root_rot[i, :])).full()) @ pos
    cost = 500 * dot((robot_arm_pos - pos), (robot_arm_pos - pos))
    nlp = {'x': q_arm, 'f': cost}
    S = nlpsol('S', 'ipopt', nlp)
    r = S(x0=[0, -0.3, 0.6, -0.3, 0], lbx=arm_lb, ubx=arm_ub)
    q_opt = r['x']
    arm_dof_pos[i, :5] = q_opt.T
# 机械臂末端在世界系下的姿态
for i in range(num_row):
    robot_arm_rot, arm_pos[i, :] = utils.arm_fk(arm_dof_pos[i, :5])
    arm_rot[i, :] = utils.rotm2quaternion(utils.quaternion2rotm(root_rot[i, :]) @ robot_arm_rot)

# 机械臂关节角速度
for i in range(num_row - 1):
    arm_dof_vel[i, :] = (arm_dof_pos[i + 1, :] - arm_dof_pos[i, :]) * fps

# 组合轨迹
# 最终输出的末端位置是在世界系中，末端相对质心的位置
ref[:, :3] = root_pos[:num_row - 1, :]
ref[:, 3:7] = root_rot[:num_row - 1, :]
ref[:, 7:10] = root_lin_vel
ref[:, 10:13] = root_ang_vel
ref[:, 13:25] = toe_pos[:num_row - 1, :]
ref[:, 25:37] = dof_pos[:num_row - 1, :]
ref[:, 37:49] = dof_vel
ref[:, 49:52] = arm_pos[:num_row - 1, :]
ref[:, 52:56] = arm_rot[:num_row - 1, :]
ref[:, 56:64] = arm_dof_pos[:num_row - 1, :]
ref[:, 64:72] = arm_dof_vel

# # 导出完整轨迹
outfile = 'output_panda/panda_arm_with_leg.txt'
np.savetxt(outfile, ref, delimiter=',')

# # 导出fixed arm轨迹
outfile = 'output_panda_fixed_arm/panda_arm_with_leg.txt'
np.savetxt(outfile, ref[:, :49], delimiter=',')

# 导出 fixed gripper轨迹
outfile = 'output_panda_fixed_gripper/panda_arm_with_leg.txt'
out = np.hstack((ref[:, :56], ref[:, 56:62], ref[:, 64:70]))
np.savetxt(outfile, out, delimiter=',')
