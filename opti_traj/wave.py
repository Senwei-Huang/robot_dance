import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
import casadi as ca


# 挥手动作：右前腿抬起来，挥舞右前腿，轨迹轮廓在y-z平面来看类似一片银杏叶
# 包含两段轨迹，一是抬腿到指定位置，不在一个平面内，需要通过两个平面图形进行组合，二是在yz平面内挥舞
# 由于机器狗的手臂关节的结构，横着挥手感觉不太好看，所以我想着做一个招财猫的动作
# 具体来说就是身体仰起来30°，然后右手做招财猫的动作
def z_1(y, y0, z0):
    z = z0 - z0 * (y + y0) ** 2 / (y0 ** 2)
    return z


num_row = 80
num_col = 49
fps = 50

wave_ref = np.ones((num_row - 1, num_col))
root_pos = np.zeros((num_row, 3))
root_rot = np.zeros((num_row, 4))
root_lin_vel = np.zeros((num_row - 1, 3))
root_ang_vel = np.zeros((num_row - 1, 3))
root_rot_dot = np.zeros((num_row - 1, 4))
toe_pos = np.zeros((num_row, 12))
dof_pos = np.zeros((num_row, 12))
dof_vel = np.zeros((num_row - 1, 12))

go2 = utils.QuadrupedRobot()
# go2的关节上下限
lb = [-0.8378, -np.pi / 2, -2.7227, -0.8378, -np.pi / 2, -2.7227, -0.8378, -np.pi / 6, -2.7227, -0.8378, -np.pi / 6,
      -2.7227]
ub = [0.8378, 3.4907, -0.8378, 0.8378, 3.4907, -0.8378, 0.8378, 4.5379, -0.8378, 0.8378, 4.5379, -0.8378]

# 质心轨迹
root_pos[:, 2] = 0.3

# 质心姿态
q0 = [0, 0, 0, 1]
q1 = [0, np.sin(-np.pi / 24), 0, np.cos(-np.pi / 24)]
end = 10
for i in range(end):
    frac = i / (end - 1)
    root_rot[i, :] = quaternion_slerp(q0, q1, frac)
start = end
end = 70
root_rot[start: end, :] = root_rot[start - 1, :]
start = end
end = num_row
for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q1, q0, frac)

# 四元数的导数
for i in range(num_row - 1):
    root_rot_dot[i, :] = (root_rot[i + 1, :] - root_rot[i, :]) * fps
# 质心角速度
for i in range(num_row - 1):
    root_ang_vel[i, :] = 2 * utils.quat2angvel_map(root_rot[i, :]) @ root_rot_dot[i, :]

# 足端轨迹
# 除右前腿，其他的都是世界系的位置，都保持不动
# 这是默认的足端位置，坐标系是固定在root处的世界系
toe_pos_init = [0.178, -0.173, -0.3, 0.178, 0.173, -0.3, -0.178, -0.173, -0.3, -0.178, 0.173, -0.3]
# 右前腿的动作通过关节角度规划
toe_pos[:] = toe_pos_init
q_FR_0 = [-0.1, 0.8, -1.5]  # 初始位置
q_FR_1 = [-0.1, -0.8, -1.5]  # 抬手中间位置
q_FR_2 = [-0.1, -0.8, -1]  # 抬手下方
q_FR_3 = [-0.1, -0.8, -2]  # 抬手上方

dof_pos[:10, :3] = np.linspace(q_FR_0, q_FR_1, 10)
dof_pos[10:20, :3] = np.linspace(q_FR_1, q_FR_2, 10)
dof_pos[20:30, :3] = np.linspace(q_FR_2, q_FR_3, 10)
dof_pos[30:40, :3] = np.linspace(q_FR_3, q_FR_2, 10)
dof_pos[40:50, :3] = np.linspace(q_FR_2, q_FR_3, 10)
dof_pos[50:60, :3] = np.linspace(q_FR_3, q_FR_2, 10)
dof_pos[60:70, :3] = np.linspace(q_FR_2, q_FR_3, 10)
dof_pos[70:80, :3] = np.linspace(q_FR_3, q_FR_0, 10)
# 计算右前腿在质心处世界坐标系下的位置
# 因为四元数转欧拉角的函数没写，所以这里直接欧拉角为[0,0,0]，再左乘旋转矩阵
for i in range(num_row):
    # c = ca.DM(go2.transrpy(dof_pos[i,:3], 0, [0, 0, 0], [0, 0, 0]) @ go2.toe).full()[:3]
    pos = (utils.quaternion2rotm(root_rot[i, :]) @
           ca.DM(go2.transrpy(dof_pos[i, :3], 0, [0, 0, 0], [0, 0, 0]) @ go2.toe).full()[:3])
    toe_pos[i, :3] = pos.T
a = ca.SX(ca.DM(utils.quaternion2rotm(root_rot[2, :])).full())
# 其余三条腿的关节角度
q = ca.SX.sym('q', 3, 1)
for j in range(1, 4):
    for i in range(num_row):
        # 这里的toe_pos是世界系足端轨迹，需要考虑质心姿态，因此左乘一个质心姿态
        pos = (ca.SX(ca.DM(utils.quaternion2rotm(root_rot[i, :])).full()) @
               (go2.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ go2.toe)[:3])
        cost = 500 * ca.dot((toe_pos[i, 3 * j:3 * j + 3] - pos[:3]), (toe_pos[i, 3 * j:3 * j + 3] - pos[:3]))
        # cost = 500 * dot(([0.179183, -0.172606, 0] - pos[:3]), ([0.179183, -0.172606, 0] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = ca.nlpsol('S', 'ipopt', nlp)
        r = S(x0=[0.1, 0.8, -1.5], lbx=lb[3 * j:3 * j + 3], ubx=ub[3 * j:3 * j + 3])
        q_opt = r['x']
        # print(q_opt)
        # toe_pos_v = (ca.SX(ca.DM(utils.quaternion2rotm(root_rot[i, :])).full()) @
        #              (go2.transrpy(q_opt, j, [0, 0, 0], [0, 0, 0]) @ go2.toe)[:3])
        # print(toe_pos_v, toe_pos[i, 3*j:3*j+3])
        dof_pos[i, 3 * j:3 * j + 3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i, :] = (dof_pos[i + 1, :] - dof_pos[i, :]) * fps

# 组合轨迹
wave_ref[:, :3] = root_pos[:num_row - 1, :]
wave_ref[:, 3:7] = root_rot[:num_row - 1, :]
wave_ref[:, 7:10] = root_lin_vel
wave_ref[:, 10:13] = root_ang_vel
wave_ref[:, 13:25] = toe_pos[:num_row - 1, :]
wave_ref[:, 25:37] = dof_pos[:num_row - 1, :]
wave_ref[:, 37:49] = dof_vel

# 导出txt
outfile = 'output/wave.txt'
np.savetxt(outfile, wave_ref, delimiter=',')
