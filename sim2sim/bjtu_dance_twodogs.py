# -*- coding: utf-8 -*-
import torch
import getsharememory_twodogs
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import time
import csv
import os
import threading
import keyboard
import random
from kinematics import *
from copy import deepcopy
import yaml
import ctypes


model_path_swing = "./model/swing/model_15000.jit"
model_path_turnjump = "./model/turnjump/model_7450.jit"


def s(x):
    return math.sin(x)


def c(x):
    return math.cos(x)


def t(x):
    return math.tan(x)


def limit(a, min_, max_):
    value = min(max(a, min_), max_)
    return value


min_pos = [[-0.69, -0.78, -2.6-0.262],
           [-0.87, -0.78, -2.6-0.262],
           [-0.69, -0.78, -2.6-0.262],
           [-0.87, -0.78, -2.6-0.262]]
max_pos = [[0.87, 3.00, -0.45-0.262],
           [0.69, 3.00, -0.45-0.262],
           [0.87, 3.00, -0.45-0.262],
           [0.69, 3.00, -0.45-0.262]]

reindex_feet1 = [1, 0, 3, 2]
max_effort = [160, 180, 572]
max_vel = [19.3, 21.6, 12.8]
joint_up_limit = [0.69, 3.92, -0.52]
joint_low_limit = [-0.87, -1.46, -2.61]


def quat_rotate_inverse(q, v):  # 获取基座z轴在惯性系下的投影矢量，q是IMU的四元数，v是z轴向量 (0,0,-1)
    """
    使用逆四元数旋转向量。

    参数:
        q (torch.Tensor): 四元数，形状为 [batch_size, 4]，格式为 [x, y, z, w]。
        v (torch.Tensor): 要旋转的向量，形状为 [batch_size, 3]。

    返回:
        torch.Tensor: 旋转后的向量，形状为 [batch_size, 3]。
    """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def rpy2quaternion(roll, pitch, yaw):
    cz = math.cos(yaw * 0.5)
    sz = math.sin(yaw * 0.5)
    cy = math.cos(pitch * 0.5)
    sy = math.sin(pitch * 0.5)
    cx = math.cos(roll * 0.5)
    sx = math.sin(roll * 0.5)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cz * cy - cx * sy * sz
    y = sx * cy * sz + cx * sy * cz
    z = cx * cy * sz - sx * sy * cz
    return x, y, z, w


class BJTUDance:
    # 初始化方法
    def __init__(self):
        self.device = torch.device('cpu')  # cpu cuda
        self.num_obs = 63  # 94
        self.num_acts = 18  # 12
        self.scale = {"lin_vel": 2.0,
                      "ang_vel": 0.25,
                      "dof_pos": 1.0,
                      "dof_vel": 0.05,
                      "height_measurements": 5.0,
                      "clip_observations": 100.,
                      "clip_actions": 1.2,
                      "action_scale": 0.25}
        default_dof_pos = [0.1,0.8,-1.5,  -0.1,0.8,-1.5,  0.1,1.,-1.5,  -0.1,1.,-1.5, 0,0,0, 0,0,0, 0,0]  # LF RF LH RH
        self.default_dof_pos = to_torch(default_dof_pos[0:self.num_acts], device=self.device, requires_grad=False)
        self.dof_pos = torch.zeros(size=(self.num_acts,), device=self.device, requires_grad=False)
        self.dof_vel = torch.zeros(size=(self.num_acts,), device=self.device, requires_grad=False)

        self.actor_state = torch.zeros(size=(self.num_obs,), device=self.device, requires_grad=False)
        self.actions = torch.zeros(size=(self.num_acts,), device=self.device, requires_grad=False)

        self.actor_state2 = torch.zeros(size=(self.num_obs,), device=self.device, requires_grad=False)
        self.actions2 = torch.zeros(size=(self.num_acts,), device=self.device, requires_grad=False)

        self.actions_last = torch.zeros(self.num_acts, device=self.device, requires_grad=False)
        self.actions_last2 = torch.zeros(self.num_acts, device=self.device, requires_grad=False)
        
        p_gains = [150.,150.,150., 150.,150.,150.,  150.,150.,150.,  150.,150.,150.,  150.,600.,150., 20.,15.,10., 10.,10.]
        d_gains = [2.,2.,2.,  2.,2.,2.,  2.,2.,2.,  2.,2.,2.,  2.,2.,2., 0.1,1.,1., 1.,1.]
        self.p_gains = to_torch(p_gains[0:self.num_acts], device=self.device, requires_grad=False)
        self.d_gains = to_torch(d_gains[0:self.num_acts], device=self.device, requires_grad=False)
        self.torques = torch.zeros(self.num_acts, device=self.device, requires_grad=False)
        self.torques2 = torch.zeros(self.num_acts, device=self.device, requires_grad=False)
        torque_limits = [160,180,572,  160,180,572,  160,180,572,  160,180,572,  100,100,100, 100,100,100, 100,100]
        self.torque_limits = to_torch(torque_limits[0:self.num_acts], device=self.device, requires_grad=False)
        print("self.torque_limits: ", self.torque_limits)

        self.joint_qd = np.zeros((4, 3))
        self.joint_qd2 = np.zeros((4, 3))
        self.joint_arm_d = np.zeros((self.num_acts-12,))
        self.joint_arm_d2 = np.zeros((self.num_acts-12,))

        self.foot_pos = np.zeros((4, 3))
        self.stand_height = 0.52

        self.lock = threading.Lock()
        self.shareinfo_tem = 0
        self.shareinfo_feed = getsharememory_twodogs.ShareInfo()
        self._inferenceReady = False
        self.IPC_PROJ_ID = 0x5A0C0001  # key键值  50LEIREN:5A0164C9
        self.SHM_SIZE = 2*1024*1024  # 共享内存大小
        self.SEM_KEY_ID = 0x5C0C0001  # SEM key键值 50LEIREN:5C0164C9
        self.shmaddr, self.semaphore = 0, 0

        self.shmaddr, self.semaphore = getsharememory_twodogs.CreatShareMem()  # change
        self.shareinfo_feed_send = getsharememory_twodogs.ShareInfo()
        print("sizeof ShareInfo is :", ctypes.sizeof(getsharememory_twodogs.ShareInfo()))
        print("sizeof sensor_package is :", ctypes.sizeof(getsharememory_twodogs.ShareInfo().sensor_package2))
        print("sizeof servo_package is :", ctypes.sizeof(getsharememory_twodogs.ShareInfo().servo_package2))
        print("sizeof ocu_package is :", ctypes.sizeof(getsharememory_twodogs.ShareInfo().ocu_package))

        self.event = threading.Event()
        self.key_pressed = None
        self.swing = 0
        self.model_select = 0
        
        # 加载模型
        self.loadPolicy()

    def on_key_press(self, event):
        self.key_pressed = event.name
        self.event.set()  # 设置事件，通知主线程处理

    def listen_keyboard(self):
        keyboard.on_press(self.on_key_press)
        while True:
            time.sleep(0.2)  # 模拟长时间运行
            # 都减15°

    def update_keyboard(self):
        if self.event.is_set():
            if self.key_pressed == 'b':
                self.swing = 1
            if self.key_pressed == 'u':
                self.stand_height += 0.01
            if self.key_pressed == 'i':
                self.stand_height -= 0.01
            if self.key_pressed == 'w':
                self.shareinfo_feed_send.ocu_package.x_des_vel += 0.05
            if self.key_pressed == 's':
                self.shareinfo_feed_send.ocu_package.x_des_vel -= 0.05
            if self.key_pressed == 'd':
                self.shareinfo_feed_send.ocu_package.yaw_turn_dot += 0.05
            if self.key_pressed == 'a':
                self.shareinfo_feed_send.ocu_package.yaw_turn_dot -= 0.05
            if self.key_pressed == '0':
                self.model_select = 0
            if self.key_pressed == '1':
                self.model_select = 1
            if self.key_pressed == '2':
                self.model_select = 2
            if self.key_pressed == '3':
                self.model_select = 3
            if self.key_pressed == '4':
                self.model_select = 4
            self.event.clear()  # 重置事件，等待下一个按键

    def update_data(self):
        self.shareinfo_feed = getsharememory_twodogs.GetFromShareMem(
            self.shmaddr, self.semaphore)
        getLegFK(self.shareinfo_feed.sensor_package.joint_q, self.foot_pos)

    def loadPolicy(self):
        self.model_swing = torch.jit.load(model_path_swing).to(self.device)
        self.model_turnjump = torch.jit.load(model_path_turnjump).to(self.device)
        self.model_swing.eval()
        self.model_turnjump.eval()

    def _compute_torques(self, actions_scaled):
        # PD controller
        torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
      
    def PutToDrive(self):
        for i in range(4):
            self.shareinfo_feed_send.servo_package.motor_enable[i] = 1
            self.shareinfo_feed_send.servo_package2.motor_enable[i] = 1
            for j in range(3):
                self.shareinfo_feed_send.servo_package.kp[i][j] = self.p_gains[i*3+j]
                self.shareinfo_feed_send.servo_package.kd[i][j] = self.d_gains[i*3+j]
                self.shareinfo_feed_send.servo_package.joint_q_d[i][j] = self.joint_qd[i][j]

                self.shareinfo_feed_send.servo_package2.kp[i][j] = self.p_gains[i*3+j]
                self.shareinfo_feed_send.servo_package2.kd[i][j] = self.d_gains[i*3+j]
                self.shareinfo_feed_send.servo_package2.joint_q_d[i][j] = self.joint_qd2[i][j]

        for k in range(self.num_acts-12):
            # self.shareinfo_feed_send.servo_package.joint_arm_d[k] = self.shareinfo_feed.sensor_package.joint_arm[k]
            self.shareinfo_feed_send.servo_package.joint_arm_d[k] = self.joint_arm_d[k]
            self.shareinfo_feed_send.servo_package2.joint_arm_d[k] = self.joint_arm_d2[k]

            self.shareinfo_feed_send.servo_package.kp_arm[k] = self.p_gains[12 + k]
            self.shareinfo_feed_send.servo_package2.kp_arm[k] = self.p_gains[12 + k]

            self.shareinfo_feed_send.servo_package.kd_arm[k] = self.d_gains[12 + k]
            self.shareinfo_feed_send.servo_package2.kd_arm[k] = self.d_gains[12 + k]
            # print("joint_arm",self.shareinfo_feed.sensor_package2.joint_arm[k])

        getsharememory_twodogs.PutToShareMem(self.shareinfo_feed_send, self.shareinfo_feed_send.ocu_package, self.shmaddr, self.semaphore)
        getsharememory_twodogs.PutToShareMem(self.shareinfo_feed_send, self.shareinfo_feed_send.servo_package, self.shmaddr, self.semaphore)
        getsharememory_twodogs.PutToShareMem(self.shareinfo_feed_send, self.shareinfo_feed_send.servo_package2, self.shmaddr, self.semaphore)

    def PutToNet(self):
        x, y, z, w = rpy2quaternion(self.shareinfo_feed.sensor_package.imu_euler[0], 
                                    self.shareinfo_feed.sensor_package.imu_euler[1], self.shareinfo_feed.sensor_package.imu_euler[2])
        base_quat = to_torch([x, y, z, w], dtype=torch.float32, device=self.device).unsqueeze(0)

        base_lin_vel_w = [self.shareinfo_feed.sensor_package.body_vel[0],
                          self.shareinfo_feed.sensor_package.body_vel[0], self.shareinfo_feed.sensor_package.body_vel[0]]
        base_ang_vel_w = [self.shareinfo_feed.sensor_package.imu_wxyz[0],
                          self.shareinfo_feed.sensor_package.imu_wxyz[0], self.shareinfo_feed.sensor_package.imu_wxyz[0]]

        base_lin_vel_w = to_torch(base_lin_vel_w, dtype=torch.float32, device=self.device).unsqueeze(0)
        base_ang_vel_w = to_torch(base_ang_vel_w, dtype=torch.float32, device=self.device).unsqueeze(0)

        base_lin_vel = quat_rotate_inverse(base_quat, base_lin_vel_w).squeeze(0)
        base_ang_vel = quat_rotate_inverse(base_quat, base_ang_vel_w).squeeze(0)

        gravity_vec = to_torch([0, 0, -1], dtype=torch.float32, device=self.device).unsqueeze(0)
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec).squeeze(0)
        
        # print("base_quat: ", base_quat)
        # print("base_lin_vel: ", base_lin_vel)
        # print("base_ang_vel: ", base_ang_vel)
        # print("projected_gravity: ", projected_gravity)

        self.actor_state[0:3] = base_lin_vel * self.scale["lin_vel"]
        self.actor_state[3:6] = base_ang_vel * self.scale["ang_vel"]
        self.actor_state[6:9] = projected_gravity
        
        # print("actor_state[0:9]: ", self.actor_state[0:9])

        for i in range(4):  # LF RF LH RH
            for j in range(3):
                self.dof_pos[i*3+j] = self.shareinfo_feed.sensor_package.joint_q[reindex_feet1[i]][j]
                self.dof_vel[i*3+j] = self.shareinfo_feed.sensor_package.joint_qd[reindex_feet1[i]][j]
                
        for i in range(self.num_acts-12):
            self.dof_pos[12+i] = self.shareinfo_feed.sensor_package.joint_arm[i]
            self.dof_vel[12+i] = self.shareinfo_feed.sensor_package.joint_arm_dq[i]  # 机械臂的关节角速度
                
        self.actor_state[9 : 9+self.num_acts] = (self.dof_pos - self.default_dof_pos) * self.scale["dof_pos"]
        self.actor_state[9+self.num_acts : 9+self.num_acts*2] = self.dof_vel * self.scale["dof_vel"]
        self.actor_state[9+self.num_acts*2 : 9+self.num_acts*3] = self.actions

    def PutToNet2(self):
        x2, y2, z2, w2 = rpy2quaternion(self.shareinfo_feed.sensor_package2.imu_euler[0], 
                                        self.shareinfo_feed.sensor_package2.imu_euler[1], self.shareinfo_feed.sensor_package2.imu_euler[2])
        quaternion2 = [x2, y2, z2, w2]
        quat_tensor2 = torch.tensor([quaternion2], dtype=torch.float32, device=self.device)
        vector2 = torch.tensor([[0, 0, -1]], dtype=torch.float32, device=self.device)

        self.actor_state2[0:3] = quat_rotate_inverse(
            quat_tensor2, vector2).unsqueeze(1)

        for i in range(4):
            for j in range(3):
                self.actor_state2[3+i*3 + j] = (math.sin(self.shareinfo_feed.sensor_package2.joint_q[reindex_feet1[i]][j]))
                self.actor_state2[15+i*3 + j] = (math.cos(self.shareinfo_feed.sensor_package2.joint_q[reindex_feet1[i]][j]))
        for j in range(3):
            self.actor_state2[27 + j] = 0.25 * (self.shareinfo_feed.sensor_package2.imu_wxyz[j])
        for i in range(12):
            self.actor_state2[30 + i] = self.actions_last2[i]

        self.actor_state2[42] = self.stand_height
        self.actor_state2[43] = (self.shareinfo_feed.ocu_package.x_des_vel * 0.25)
        self.actor_state2[44] = (0 * 0.25)
        self.actor_state2[45] = (self.shareinfo_feed.ocu_package.yaw_turn_dot * 0.05)
        self.actor_state2[46] = 1
        self.actor_state2[47] = 1
        self.actor_state2[48:] = 0
        
    def inference_(self):
        last_vel = 0
        last_yaw = 0

        while True:
            start_RL_Time = time.perf_counter()
            self.update_data()
            self.update_keyboard()

            if (self.shareinfo_feed_send.ocu_package.x_des_vel != last_vel):
                print('x_vel', self.shareinfo_feed_send.ocu_package.x_des_vel)
            if (self.shareinfo_feed_send.ocu_package.yaw_turn_dot != last_yaw):
                print('yaw_dot', self.shareinfo_feed_send.ocu_package.yaw_turn_dot)

            last_vel = self.shareinfo_feed_send.ocu_package.x_des_vel
            last_yaw = self.shareinfo_feed_send.ocu_package.yaw_turn_dot

            self.PutToNet()
            self.PutToNet2()

            # self.actor_state = torch.zeros(size=(self.num_obs,), device=self.device, requires_grad=False)
            # self.actor_state = torch.ones(size=(self.num_obs,), device=self.device, requires_grad=False)
            with torch.no_grad():
                # actions = self.model_turnjump(self.actor_state)
                # actions2 = self.model_turnjump(self.actor_state2)
                actions = self.model_swing(self.actor_state)
                actions2 = self.model_swing(self.actor_state2)
            # print("actions: ", actions)
            # print("self.actor_state: ", self.actor_state)
            
            clip_actions = self.scale["clip_actions"]
            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
            self.actions2 = torch.clip(actions2, -clip_actions, clip_actions).to(self.device)
            self.actions_last = self.actions.clone()
            self.actions_last2 = self.actions2.clone()
            self.actions_scaled = self.actions * self.scale["action_scale"]
            self.actions2_scaled = self.actions2 * self.scale["action_scale"]

            self.torques = self._compute_torques(self.actions_scaled).view(self.torques.shape)
            self.torques2 = self._compute_torques(self.actions2_scaled).view(self.torques2.shape)

            for i in range(4):
                for j in range(3):
                    self.joint_qd[reindex_feet1[i]][j] = self.actions_scaled.tolist()[i*3+j]
                    self.joint_qd2[reindex_feet1[i]][j] = self.actions2_scaled.tolist()[i*3+j]
                    
            for i in range(self.num_acts-12):
                self.joint_arm_d[i] = self.actions_scaled.tolist()[12 + i]
                self.joint_arm_d2[i] = self.actions2_scaled.tolist()[12 + i]
            
            self.PutToDrive()

            end = time.perf_counter()
            last_time = time.perf_counter() - start_RL_Time
            if (last_time > 0.02):
                print("time over:", time.perf_counter()-start_RL_Time)
            if last_time < 0.02:
                time.sleep(0.02 - last_time)  # 保证50Hz频率


if __name__ == "__main__":
    bjtudance = BJTUDance()
    bjtudance.keyboard_thread = threading.Thread(target=bjtudance.listen_keyboard)
    bjtudance.keyboard_thread.start()
    while (True):
        bjtudance.inference_()
