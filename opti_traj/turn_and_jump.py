import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *

# 规划转身跳跃动作的轨迹，输出的文件为一个状态矩阵txt文件，列数为49列，因为读取文件的固定格式为49列
# root位置[0:3]，root姿态[3:7] [x,y,z,w]，线速度[7:10]，角速度[10:13]
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]，panda7没看过是什么顺序
# 关节角度和关节[25:37]


# 机身右转30°，左转60°，右转60°，左转30°
go2 = utils.QuadrupedRobot()
num_row = 80
num_col = 49
fps = 50

turn_and_jump_ref = np.ones((num_row-1, num_col))
root_pos = np.zeros((num_row, 3))
root_rot = np.zeros((num_row, 4))
root_lin_vel = np.zeros((num_row-1, 3))
root_ang_vel = np.zeros((num_row-1, 3))
root_rot_dot = np.zeros((num_row-1, 4))
toe_pos = np.zeros((num_row, 12))
dof_pos = np.zeros((num_row, 12))
dof_vel = np.zeros((num_row-1, 12))

def h_1(t):
    x = 0.2  # 对称轴
    h = g / 2 * x ** 2
    return h - g / 2 * (t - x) ** 2


# 质心轨迹
for i in range(20):
    root_pos[i, 2] = h_1(i / 50) + 0.3
for i in range(3):
    root_pos[20 * (i + 1):20 * (i + 2), :] = root_pos[:20, :]
# 质心线速度
for i in range(num_row-1):
    root_lin_vel[i,:] = (root_pos[i+1,:] - root_pos[i,:]) * fps
    # 四元数的导数
    root_rot_dot[i,:] = (root_rot[i+1,:] - root_rot[i,:]) * fps


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


# 读取文件
motion_files = 'output/keep_the_beat/keep_the_beat_ref_simp.txt'
motion_data = np.loadtxt(motion_files, delimiter=',')
toe_pos_init = motion_data[0, 13:25] # 默认足端位置
toe_pos[:] = toe_pos_init
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

toe_pos_world = np.zeros_like(toe_pos)
# 计算世界系下的足端轨迹
# 然后减去世界系下的质心位置，也就是世界系或者质心定向系的足端相对root的坐标，所以没加root坐标
for i in range(toe_pos.shape[0]):
    toe_pos_world[i, :3] = utils.quaternion2rotm(root_rot[i,:]) @ toe_pos[i, :3]
    toe_pos_world[i, 3:6] = utils.quaternion2rotm(root_rot[i,:]) @ toe_pos[i, 3:6]
    toe_pos_world[i, 6:9] = utils.quaternion2rotm(root_rot[i,:]) @ toe_pos[i, 6:9]
    toe_pos_world[i, 9:12] = utils.quaternion2rotm(root_rot[i,:]) @ toe_pos[i, 9:12]

# 计算关节角度
# go2的关节上下限
lb = [-0.8378,-np.pi/2,-2.7227,-0.8378,-np.pi/2,-2.7227,-0.8378,-np.pi/6,-2.7227,-0.8378,-np.pi/6,-2.7227]
ub = [0.8378,3.4907,-0.8378,0.8378,3.4907,-0.8378,0.8378,4.5379,-0.8378,0.8378,4.5379,-0.8378]
q = SX.sym('q', 3, 1)

for j in range(4):
    for i in range(num_row):
        # print(j,i)
        # toe_pos是质心系下的足端轨迹，所以欧拉角和质心都是[0, 0, 0]
        pos = go2.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        cost = 500*dot((toe_pos[i, 3*j:3*j+3] - pos[:3]), (toe_pos[i, 3*j:3*j+3] - pos[:3]))
        # cost = 500 * dot(([0.179183, -0.172606, 0] - pos[:3]), ([0.179183, -0.172606, 0] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = nlpsol('S', 'ipopt', nlp)
        r = S(x0 = [0.1, 0.8, -1.5], lbx = lb[3*j:3*j+3], ubx = ub[3*j:3*j+3])
        q_opt = r['x']
        # print(q_opt)
        # toe_pos_v = go2.transrpy(q_opt, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        # print(toe_pos_v, toe_pos[i, :3])
        dof_pos[i, 3*j:3*j+3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i,:] = (dof_pos[i+1,:] - dof_pos[i,:]) * fps


# 组合轨迹
turn_and_jump_ref[:, :3] = root_pos[:num_row-1,:]
turn_and_jump_ref[:, 3:7] = root_rot[:num_row-1,:]
turn_and_jump_ref[:, 7:10] = root_lin_vel
turn_and_jump_ref[:, 10:13] = root_ang_vel
turn_and_jump_ref[:, 13:25] = toe_pos_world[:num_row-1,:]
turn_and_jump_ref[:, 25:37] = dof_pos[:num_row-1,:]
turn_and_jump_ref[:, 37:49] = dof_vel

# 导出txt
outfile = 'output/turn_and_jump.txt'
np.savetxt(outfile, turn_and_jump_ref, delimiter=',')
