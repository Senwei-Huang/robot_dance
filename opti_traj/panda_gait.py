import numpy as np
import utils
import casadi as ca
import CPG
import matplotlib.pyplot as plt

# 使用CPG模型设计trot步态，使用hopf振荡器获得周期性的相位信号，根据这个相位信号计算足端轨迹

# 实例化panda7
# panda7的关节上下限
panda_lb = [-0.87, -1.78, -2.53, -0.69, -1.78, -2.53, -0.87, -1.3, -2.53, -0.69, -1.3, -2.53]
panda_ub = [0.69, 3.4, -0.45, 0.87, 3.4, -0.45, 0.69, 4, -0.45, 0.87, 4, -0.45]
panda_toe_pos_init = [0.300133, -0.287854, -0.481828, 0.300133, 0.287854, -0.481828, -0.349867,
                      -0.287854, -0.481828, -0.349867, 0.287854, -0.481828]
panda7 = utils.QuadrupedRobot(l=0.65, w=0.225, l1=0.126375, l2=0.34, l3=0.34,
                              lb=panda_lb, ub=panda_ub, toe_pos_init=panda_toe_pos_init)
num_row = 100
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
hopf_signal = np.zeros((num_row,8))

# CPG信号
initPos = np.array([0.5, -0.5, -0.5, 0.5, 0, 0, 0, 0])
gait = 'spacetrot'
cpg = CPG.cpgBuilder(initPos, gait=gait)

# 欧拉法获取振荡信号
t = np.linspace(0, num_row/fps, num_row)
temp = cpg.initPos.reshape(8,-1)

for i in range(num_row):
    v = cpg.hopf_osci(temp)
    temp += v/fps
    hopf_signal[i] = temp.flatten()
# 相位
phase = np.arctan2(hopf_signal[:,4:8], hopf_signal[:,0:4])

# print(phase)
# plt.figure()
# plt.plot(t, hopf_signal[:,0], linewidth=5)
# # plt.plot(t, hopf_signal[:,4], linewidth=5)
# plt.plot(t, hopf_signal[:,1], linewidth=3)
# # plt.plot(t, hopf_signal[:,5], linewidth=2)
# plt.plot(t, hopf_signal[:,2], linewidth=3)
# plt.plot(t, hopf_signal[:,3], linewidth=3)
#
# plt.show()

plt.figure()
plt.plot(t, phase[:,0], linewidth=6)
plt.plot(t, phase[:,1], linewidth=6)
plt.plot(t, phase[:,2], linewidth=2)
plt.plot(t, phase[:,3], linewidth=2)
plt.show()

# 足端位置
vx = 1
ax = vx*cpg.T*cpg.beta
ay = 0
az = 0.08
for i in range(4):
    for j in range(num_row):
        if phase[j,i]<0:
            p = -phase[j,i]/np.pi
            toe_pos[j, 3*i+2] = CPG.endEffectorPos_z(az,p)
        else:
            p = phase[j,i]/np.pi
            toe_pos[j,3*i] = 0

        toe_pos[j,3*i] = CPG.endEffectorPos_xy(ax,p)
        toe_pos[j,3*i+1] = CPG.endEffectorPos_xy(ay,p)

# plt.figure()
# plt.plot(toe_pos[:,0], toe_pos[:,2], linewidth=5)
# plt.show()
# 足端相对质心的坐标
toe_pos += panda7.toe_pos_init
q = ca.SX.sym('q', 3, 1)

for j in range(4):
    for i in range(num_row):
        # print(j,i)
        # toe_pos是质心系下的足端轨迹，所以欧拉角和质心都是[0, 0, 0]
        pos = panda7.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ panda7.toe
        cost = 500*ca.dot((toe_pos[i, 3*j:3*j+3] - pos[:3]), (toe_pos[i, 3*j:3*j+3] - pos[:3]))
        # cost = 500 * dot(([0.179183, -0.172606, 0] - pos[:3]), ([0.179183, -0.172606, 0] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = ca.nlpsol('S', 'ipopt', nlp)
        r = S(x0 = [0.1, 0.8, -1.5], lbx = panda7.lb[3*j:3*j+3], ubx = panda7.ub[3*j:3*j+3])
        q_opt = r['x']
        # print(q_opt)
        # toe_pos_v = go2.transrpy(q_opt, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        # print(toe_pos_v, toe_pos[i, :3])
        dof_pos[i, 3*j:3*j+3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i,:] = (dof_pos[i+1,:] - dof_pos[i,:]) * fps

# 质心位置
# x = vx*t
x=0
root_pos[:,0] = x
root_pos[:,2] = 0.55
# 质心速度
# root_lin_vel[:,0] = vx
root_lin_vel[:,0] = vx
# 机身方向
root_rot[:,3] = 1
# 机身角速度默认为0

# arm fk
robot_arm_rot, robot_arm_pos=utils.arm_fk([0, 0, 0, 0, 0, 0])
# 机械臂末端在机身坐标系下的位置
arm_pos[:] = robot_arm_pos
# 机械臂末端在世界系下的姿态
for i in range(num_row):
    arm_rot[i, :] = utils.rotm2quaternion(utils.quaternion2rotm(root_rot[i,:]) @ robot_arm_rot)

# 组合轨迹
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

# 太空步
gait = 'spacetrot'
# # 导出完整轨迹
outfile = 'output_panda/panda_'+gait+'.txt'
np.savetxt(outfile, ref, delimiter=',')

# # 导出fixed arm轨迹
outfile = 'output_panda_fixed_arm/panda_'+gait+'.txt'
np.savetxt(outfile, ref[:, :49], delimiter=',')

# 导出 fixed gripper轨迹
outfile = 'output_panda_fixed_gripper/panda_'+gait+'.txt'
out = np.hstack((ref[:, :56], ref[:, 56:62], ref[:, 64:70]))
np.savetxt(outfile, out, delimiter=',')

