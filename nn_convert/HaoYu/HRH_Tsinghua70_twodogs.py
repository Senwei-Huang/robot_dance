from actorNN_HRH import  ModelLoad
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
from StateEstimator import *
from copy import deepcopy
import yaml
import ctypes
file_path = '/home/ixxuan/Project/Tsinghua/Apr18_16-57-48_/test.csv'
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"The file {file_path} has been deleted.")
with open('./model_config.yaml', 'r') as file:
    model_path = yaml.safe_load(file)
model0_path = model_path["model"]["model0"]
model1_path = model_path["model"]["model1"]
model2_path = model_path["model"]["model2"]
model3_path = model_path["model"]["model3"]
model4_path = model_path["model"]["model4"]

l1=0.1
l2=0.34
l3=0.34
def s(x):
    return math.sin(x)
def c(x):
    return math.cos(x)
def t(x):
    return math.tan(x)
def limit(a,min_,max_):
   value = min(max(a,min_),max_)
   return value

min_pos =[[-0.69, -0.78, -2.6-0.262],
          [-0.87, -0.78, -2.6-0.262],
          [-0.69, -0.78, -2.6-0.262],
          [-0.87, -0.78, -2.6-0.262]]
max_pos = [[0.87, 3.00, -0.45-0.262],
           [0.69, 3.00, -0.45-0.262],
           [0.87, 3.00, -0.45-0.262],
           [0.69, 3.00, -0.45-0.262]]

reindex_feet1 = [1, 0, 3, 2]
max_effort = [160,180,572]
max_vel = [19.3,21.6,12.8]
joint_up_limit = [0.69,3.92,-0.52]
joint_low_limit = [-0.87,-1.46,-2.61]

class Transform_package:
    def __init__(self):
        self.imu_euler = np.zeros((3,))
        self.imu_wxyz = np.zeros((3,))
        self.joint_q = np.zeros((4, 3))
        self.joint_qd = np.zeros((4, 3))
        self.x_des_vel = 0.0
        self.y_des_vel = 0.0
        self.yaw_turn_dot = 0.0

def quat_rotate_inverse(q, v):#获取基座z轴在惯性系下的投影矢量，q是IMU的四元数，v是z轴向量 (0,0,-1)
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

class Tsinghua:
    # 初始化方法
    def __init__(self):
        self.joint_dq_d = np.zeros((4, 3))
        self.joint_dq_d2 = np.zeros((4, 3))
        self.joint_qd_d = np.zeros((4, 3))
        self.foot_pos = np.zeros((4,3))
        self.tem_pos = np.zeros((4,3))
        self.stand_height = 0.52
        self.episode_length_buf = torch.zeros(1, device='cpu')
        self.lock = threading.Lock()
        self.actions_last = torch.zeros(12,device='cpu')
        self.actions_last2 = torch.zeros(12,device='cpu')
        self.ontology_sense_matrix = torch.zeros(58,6, device='cpu')
        self.ontology_sense_matrix2 = torch.zeros(58,6, device='cpu')
        self.scale={"lin_vel" : 2.0,
                    "ang_vel" : 0.25,
                    "dof_pos" : 1.0,
                    "dof_vel" : 0.05,
                    "height_measurements" : 5.0,
                    "clip_observations" : 100.,
                    "clip_actions" : 1.2}
        self.shareinfo_tem = 0
        self.shareinfo_feed = getsharememory_twodogs.ShareInfo()
        self._inferenceReady = False
        self.IPC_PROJ_ID = 0x5A0C0001 #key键值  50LEIREN:5A0164C9
        self.SHM_SIZE = 2*1024*1024 #共享内存大小
        self.SEM_KEY_ID = 0x5C0C0001 #SEM key键值 50LEIREN:5C0164C9
        self.shmaddr,self.semaphore = 0,0
        self.actor_state = torch.zeros(size = (512,), device='cpu')
        self.actions = torch.zeros(size = (12,), device='cpu')
        self.euler = 0

        self.actor_state2 = torch.zeros(size = (512,), device='cpu')
        self.actions2 = torch.zeros(size = (12,), device='cpu')
        
        self.shmaddr,self.semaphore = getsharememory_twodogs.CreatShareMem()##change
        self.shareinfo_feed_send = getsharememory_twodogs.ShareInfo()
        print("sizeof ShareInfo is :",ctypes.sizeof(getsharememory_twodogs.ShareInfo()))
        # print("sizeof tsinghua_rec_package is :",ctypes.sizeof(getsharememory_twodogs.ShareInfo().tsinghua_rec_package))
        # print("sizeof tsinghua_send_package is :",ctypes.sizeof(getsharememory_twodogs.ShareInfo().tsinghua_send_package))
        print("sizeof sensor_package is :",ctypes.sizeof(getsharememory_twodogs.ShareInfo().sensor_package2))
        print("sizeof servo_package is :",ctypes.sizeof(getsharememory_twodogs.ShareInfo().servo_package2))
        print("sizeof ocu_package is :",ctypes.sizeof(getsharememory_twodogs.ShareInfo().ocu_package))

        
        self.event = threading.Event()
        self.key_pressed = None
        self.jump = 0
        self.leg_touch = np.zeros(4)
        self.swing_count = 0
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
            #都减15°

    def update_data(self):
        self.shareinfo_feed = getsharememory_twodogs.GetFromShareMem(self.shmaddr,self.semaphore)
        getLegFK(self.shareinfo_feed.sensor_package.joint_q,self.foot_pos)


    def update_ontology_sense_buffer(self):    #TODO 
        buf = self.actor_state[0:58].unsqueeze(1)
        ontology_buf = torch.cat((self.ontology_sense_matrix[:,1:],buf),dim=-1)
        self.ontology_sense_matrix = ontology_buf

    def update_ontology_sense_buffer2(self):    #TODO 
        buf2 = self.actor_state2[0:58].unsqueeze(1)
        ontology_buf2 = torch.cat((self.ontology_sense_matrix2[:,1:],buf2),dim=-1)
        self.ontology_sense_matrix2 = ontology_buf2

    def loadPolicy(self):
        self.remodel_0 = ModelLoad()
        self.remodel_1 = ModelLoad()
        self.remodel_2 = ModelLoad()
        self.remodel_3 = ModelLoad()
        self.remodel_4 = ModelLoad()
        self.remodel_0.cfg.model.path = model0_path
        self.remodel_1.cfg.model.path = model1_path
        self.remodel_2.cfg.model.path = model2_path
        self.remodel_3.cfg.model.path = model3_path
        self.remodel_4.cfg.model.path = model4_path
        self.remodel_0.load_model()
        self.remodel_1.load_model()
        self.remodel_2.load_model()
        self.remodel_3.load_model()
        self.remodel_4.load_model()
    
    def PutToDrive(self):
            for ii in range(8):
                # self.shareinfo_feed_send.servo_package.joint_arm_d[ii] = self.shareinfo_feed.sensor_package.joint_arm[ii]
                self.shareinfo_feed_send.servo_package2.joint_arm_d[ii] = 0.0

                self.shareinfo_feed_send.servo_package.kp_arm[ii] = 20
                self.shareinfo_feed_send.servo_package2.kp_arm[ii] = 20

                self.shareinfo_feed_send.servo_package.kd_arm[ii] = 1
                self.shareinfo_feed_send.servo_package2.kd_arm[ii] = 1
                # print("joint_arm",self.shareinfo_feed.sensor_package2.joint_arm[ii])
                
            # print("eluer2 ",self.euler)
            for i in range(4):
                self.shareinfo_feed_send.servo_package.motor_enable[i] = 1
                self.shareinfo_feed_send.servo_package2.motor_enable[i] = 1
                for j in range(3):
                    self.shareinfo_feed_send.servo_package.kp[i][j]= max_effort[j]*0.8
                    self.shareinfo_feed_send.servo_package.kd[i][j]= max_vel[j]*0.1
                    self.shareinfo_feed_send.servo_package.joint_q_d[i][j] = (self.shareinfo_feed.sensor_package.joint_q[i][j] + self.joint_dq_d[i][j])#(self.shareinfo_feed.sensor_package.joint_q[i][j] + self.joint_dq_d[i][j])

                    self.shareinfo_feed_send.servo_package2.kp[i][j]= max_effort[j]*0.8
                    self.shareinfo_feed_send.servo_package2.kd[i][j]= max_vel[j]*0.1
                    self.shareinfo_feed_send.servo_package2.joint_q_d[i][j] = (self.shareinfo_feed.sensor_package2.joint_q[i][j] + self.joint_dq_d2[i][j])
            
            getsharememory_twodogs.PutToShareMem(self.shareinfo_feed_send,self.shareinfo_feed_send.ocu_package,self.shmaddr,self.semaphore)
            getsharememory_twodogs.PutToShareMem(self.shareinfo_feed_send,self.shareinfo_feed_send.servo_package,self.shmaddr,self.semaphore)
            getsharememory_twodogs.PutToShareMem(self.shareinfo_feed_send,self.shareinfo_feed_send.servo_package2,self.shmaddr,self.semaphore)


    def inference_(self):
        last_vel = 0
        last_yaw = 0

        while True:
            start_RL_Time = time.perf_counter()
            self.update_data()

            if self.event.is_set():
                if self.key_pressed == 'b':
                    self.swing = 1
                if self.key_pressed == 'u':
                    self.stand_height += 0.01
                if self.key_pressed =='i':
                    self.stand_height -= 0.01 
                if self.key_pressed =='w':
                    self.shareinfo_feed_send.ocu_package.x_des_vel += 0.05
                if self.key_pressed =='s':
                    self.shareinfo_feed_send.ocu_package.x_des_vel -= 0.05
                if self.key_pressed =='d':
                    self.shareinfo_feed_send.ocu_package.yaw_turn_dot += 0.05
                if self.key_pressed =='a':
                    self.shareinfo_feed_send.ocu_package.yaw_turn_dot -= 0.05
                if self.key_pressed =='0':
                    self.model_select = 0
                if self.key_pressed =='1':
                    self.model_select = 1
                if self.key_pressed =='2':
                    self.model_select = 2
                if self.key_pressed =='3':
                    self.model_select = 3
                if self.key_pressed =='4':
                    self.model_select = 4
                self.event.clear()  # 重置事件，等待下一个按键
            if(self.shareinfo_feed_send.ocu_package.x_des_vel != last_vel):
                print('x_vel',self.shareinfo_feed_send.ocu_package.x_des_vel)
            if(self.shareinfo_feed_send.ocu_package.yaw_turn_dot != last_yaw):
                print('yaw_dot',self.shareinfo_feed_send.ocu_package.yaw_turn_dot)
            last_vel = self.shareinfo_feed_send.ocu_package.x_des_vel
            last_yaw = self.shareinfo_feed_send.ocu_package.yaw_turn_dot
            self.PutToNet()
            self.PutToNet2()
            self.update_ontology_sense_buffer()
            self.update_ontology_sense_buffer2()

            self.ontology_sense_matrix_tem = self.ontology_sense_matrix.view(-1)
            self.ontology_sense_matrix_tem2 = self.ontology_sense_matrix2.view(-1)

            with torch.no_grad():
                if self.model_select == 0:
                    self.actions = self.remodel_0.inference(self.actor_state, self.ontology_sense_matrix_tem)
                    self.actions2 = self.remodel_0.inference(self.actor_state2, self.ontology_sense_matrix_tem2)
                if self.model_select == 1:
                    self.actions = self.remodel_1.inference(self.actor_state, self.ontology_sense_matrix_tem)
                    self.actions2 = self.remodel_1.inference(self.actor_state2, self.ontology_sense_matrix_tem2)
                if self.model_select == 2:
                    self.actions = self.remodel_2.inference(self.actor_state, self.ontology_sense_matrix_tem)
                    self.actions2 = self.remodel_2.inference(self.actor_state2, self.ontology_sense_matrix_tem2)
                if self.model_select == 3:
                    self.actions = self.remodel_3.inference(self.actor_state, self.ontology_sense_matrix_tem)
                    self.actions2 = self.remodel_3.inference(self.actor_state2, self.ontology_sense_matrix_tem2)
                if self.model_select == 4:
                    self.actions = self.remodel_4.inference(self.actor_state, self.ontology_sense_matrix_tem)
                    self.actions2 = self.remodel_4.inference(self.actor_state2, self.ontology_sense_matrix_tem2)
            self.actions_last = self.actions.clone()
            self.actions_last2 = self.actions2.clone()

            for i in range(4):
                for j in range(3):
                    self.joint_dq_d[reindex_feet1[i]][j] = self.actions.tolist()[i*3+j]
                    self.joint_dq_d2[reindex_feet1[i]][j] = self.actions2.tolist()[i*3+j]
            self.PutToDrive()
            end = time.perf_counter()
            last_time = time.perf_counter() - start_RL_Time
            if(last_time > 0.02):
                print("time over:",time.perf_counter()-start_RL_Time)
            if last_time < 0.02:
                time.sleep(0.02 - last_time)#保证50Hz频率
            
            

    def PutToNet(self):
        self.euler = self.shareinfo_feed.sensor_package.imu_euler[2]
        x, y, z, w = rpy2quaternion(self.shareinfo_feed.sensor_package.imu_euler[0], self.shareinfo_feed.sensor_package.imu_euler[1], self.shareinfo_feed.sensor_package.imu_euler[2])
        # print("eluer ",self.euler)
        quaternion = [x,y,z,w]
        quat_tensor = torch.tensor([quaternion], dtype=torch.float32, device='cpu')
        vector = torch.tensor([[0, 0, -1]], dtype=torch.float32, device='cpu')
        self.actor_state[0:3] = quat_rotate_inverse(quat_tensor, vector).unsqueeze(1)

        for i in range(4):
            for j in range(3):
                self.actor_state[3+i*3 + j] = (math.sin(self.shareinfo_feed.sensor_package.joint_q[reindex_feet1[i]][j]))
                self.actor_state[15+i*3 + j] = (math.cos(self.shareinfo_feed.sensor_package.joint_q[reindex_feet1[i]][j]))
        for j in range(3):
            self.actor_state[27 + j] = 0.25 * (self.shareinfo_feed.sensor_package.imu_wxyz[j])
        for i in range(12): 
            self.actor_state[30 + i] = self.actions_last[i]
        self.actor_state[42] = self.stand_height
        self.actor_state[43] = (self.shareinfo_feed.ocu_package.x_des_vel * 0.25)
        self.actor_state[44] = (0 * 0.25)
        self.actor_state[45] = (self.shareinfo_feed.ocu_package.yaw_turn_dot * 0.05)
        self.actor_state[46] = 1
        self.actor_state[47] = 1
        self.actor_state[48:] = 0

    def PutToNet2(self):
        x2, y2, z2, w2 = rpy2quaternion(self.shareinfo_feed.sensor_package2.imu_euler[0], self.shareinfo_feed.sensor_package2.imu_euler[1], self.shareinfo_feed.sensor_package2.imu_euler[2])
        quaternion2 = [x2,y2,z2,w2]
        quat_tensor2 = torch.tensor([quaternion2], dtype=torch.float32, device='cpu')
        vector2 = torch.tensor([[0, 0, -1]], dtype=torch.float32, device='cpu')
        self.actor_state2[0:3] = quat_rotate_inverse(quat_tensor2, vector2).unsqueeze(1)

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

if __name__ == "__main__":
    # for key,weights in torch.load('./70HRH/model_5600.pt').items():
    #     print(key,weights.shape)
    
    tsinghua = Tsinghua()
    tsinghua.keyboard_thread = threading.Thread(target=tsinghua.listen_keyboard)
    tsinghua.keyboard_thread.start()
    while(True):
        tsinghua.inference_()
        


