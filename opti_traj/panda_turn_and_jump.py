import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *

# 规划摇摆的轨迹，输出的文件为一个状态矩阵txt文件，列数为49列，因为读取文件的固定格式为49列
# root位置[0:3]，root姿态[3:7] [x,y,z,w]，线速度[7:10]，角速度[10:13]
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]，panda7没看过是什么顺序
# 关节角度和关节[25:37]


# 机身右转30°，左转60°，右转60°，左转30°
# 实例化panda7
# panda7的关节上下限
panda_lb = [-0.87, -1.78, -2.53, -0.69, -1.78, -2.53, -0.87, -1.3, -2.53, -0.69, -1.3, -2.53]
panda_ub = [0.69, 3.4, -0.45, 0.87, 3.4, -0.45, 0.69, 4, -0.45, 0.87, 4, -0.45]
panda_toe_pos_init = [0.300133, -0.287854, -0.481828, 0.300133, 0.287854, -0.481828, -0.349867,
                      -0.287854, -0.481828, -0.349867, 0.287854, -0.481828]
panda7 = utils.QuadrupedRobot(l=0.65, w=0.225, l1=0.126375, l2=0.34, l3=0.34,
                              lb=panda_lb, ub=panda_ub, toe_pos_init=panda_toe_pos_init)
num_row = 250
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
arm_dof_vel = np.zeros((num_row-1, 8))

def h_1(t):
    x = 0.2  # 对称轴
    h = g / 2 * x ** 2
    return h - g / 2 * (t - x) ** 2

# 质心轨迹
for i in range(20):
    root_pos[i, 2] = h_1(i / 50) + 0.55
for i in range(3):
    root_pos[20 * (i + 1):20 * (i + 2), :] = root_pos[:20, :]
# 质心线速度
for i in range(num_row-1):
    root_lin_vel[i,:] = (root_pos[i+1,:] - root_pos[i,:]) * fps
# 姿态
# 计算姿态有点小问题，但是影响不大
q1 = [0, 0, 0, 1]  # [x,y,z,w]
q2 = [0, 0, np.sin(-np.pi / 12), np.cos(-np.pi / 12)]
q3 = [0, 0, np.sin(np.pi / 12), np.cos(np.pi / 12)]
interval = 20
start = 0
end = start + interval

for i in range(end):
    frac = i / end
    root_rot[i, :] = quaternion_slerp(q1, q2, frac)

start = end
end = start + interval
for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q2, q3, frac)

start = end
end = start + interval
for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q3, q2, frac)

start = end
end = start + interval
for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q2, q1, frac)


# 四元数的导数
for i in range(num_row-1):
    root_rot_dot[i,:] = (root_rot[i+1,:] - root_rot[i,:]) * fps
# 质心角速度
for i in range(num_row-1):
    root_ang_vel[i, :] = 2 * utils.quat2angvel_map(root_rot[i,:])@ root_rot_dot[i,:]


# 获取足端初始位置，初始位置在世界系下保持不变，需要计算质心系的足端位置
toe_pos[:] = panda7.toe_pos_init
# 质心系足端轨迹,跳跃伸腿时间为0.04s
for i in range(3):
    toe_pos[i, 2] -= h_1(i / 50)
    toe_pos[i, 5] = toe_pos[i, 8] = toe_pos[i, 11] = toe_pos[i, 2]
for i in range(3,18):
    toe_pos[i,2] = toe_pos[2, 2]
    toe_pos[i, 5] = toe_pos[i, 8] = toe_pos[i, 11] = toe_pos[i, 2]
for i in range(18,20):
    toe_pos[i, 2] -= h_1(i / 50)
    toe_pos[i, 5] = toe_pos[i, 8] = toe_pos[i, 11] = toe_pos[i, 2]
for i in range(3):
    toe_pos[20 * (i + 1):20 * (i + 2), :] = toe_pos[:20, :]
# 质心系的足端位置，用于计算关节角度
toe_pos_body = toe_pos[:]

# 计算世界系下的足端轨迹，这里和swing不同，这里已知质心系下足端位置，而swing已知世界系下足端位置
for i in range(toe_pos.shape[0]):
    toe_pos[i, :3] = utils.quaternion2rotm(root_rot[i,:]) @ toe_pos[i, :3]
    toe_pos[i, 3:6] = utils.quaternion2rotm(root_rot[i,:]) @ toe_pos[i, 3:6]
    toe_pos[i, 6:9] = utils.quaternion2rotm(root_rot[i,:]) @ toe_pos[i, 6:9]
    toe_pos[i, 9:12] = utils.quaternion2rotm(root_rot[i,:]) @ toe_pos[i, 9:12]

# go2的关节上下限
q = SX.sym('q', 3, 1)
for j in range(4):
    for i in range(num_row):
        # toe_pos是质心系的足端位置，所以rpy以及质心位置都是[0 0 0]
        # 这里不把上面的部分合起来放到一个式子里是因为下面使用casadi，上面使用了numpy，混合运算会出问题
        pos = panda7.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ panda7.toe
        cost = 500 * dot((toe_pos_body[i, 3 * j:3 * j + 3] - pos[:3]), (toe_pos_body[i, 3 * j:3 * j + 3] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = nlpsol('S', 'ipopt', nlp)
        r = S(x0=[0.1, 0.8, -1.5], lbx=panda7.lb[3 * j:3 * j + 3], ubx=panda7.ub[3 * j:3 * j + 3])
        q_opt = r['x']
        dof_pos[i, 3 * j:3 * j + 3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i, :] = (dof_pos[i + 1, :] - dof_pos[i, :]) * fps

# arm fk
robot_arm_rot, robot_arm_pos=utils.arm_fk([0, 0, 0, 0, 0, 0])

for i in range(num_row):
    # 机械臂末端位置
    arm_pos[i, :] = utils.quaternion2rotm(root_rot[i, :]) @ robot_arm_pos + root_pos[i, :]
    # 机械臂末端姿态
    arm_rot[i, :] = utils.rotm2quaternion(utils.quaternion2rotm(root_rot[i, :]) @ robot_arm_rot)



# 组合轨迹
# 最终输出的末端位置是在世界系中，末端相对质心的位置
ref[:, :3] = root_pos[:num_row - 1, :]
ref[:, 3:7] = root_rot[:num_row - 1, :]
ref[:, 7:10] = root_lin_vel
ref[:, 10:13] = root_ang_vel
ref[:, 13:25] = panda7.toe_pos_init # 前面计算出质心系下的足端位置是方便计算关节角度
ref[:, 25:37] = dof_pos[:num_row - 1, :]
ref[:, 37:49] = dof_vel
ref[:, 49:52] = arm_pos[:num_row - 1, :]
ref[:, 52:56] = arm_rot[:num_row - 1, :]
ref[:, 56:64] = arm_dof_pos[:num_row - 1, :]
ref[:, 64:72] = arm_dof_vel

# 导出txt
outfile = 'output_panda/panda_turn_and_jump.txt'
np.savetxt(outfile, ref, delimiter=',')