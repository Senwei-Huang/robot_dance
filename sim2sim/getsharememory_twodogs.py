import sysv_ipc
import os
import ctypes

    # 定义C语言中的key_t类型
key_t = ctypes.c_int

    # 定义C语言中的shmget函数
shmget = ctypes.CDLL(None).shmget
shmget.argtypes = [key_t, ctypes.c_size_t, ctypes.c_int]
shmget.restype = ctypes.c_int
    # 定义C语言中的shmat函数
shmat = ctypes.CDLL(None).shmat
shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
shmat.restype = ctypes.c_void_p
    # 定义shmdt函数
shmdt = ctypes.CDLL(None).shmdt
shmdt.argtypes = [ctypes.c_void_p]
shmdt.restype = ctypes.c_int

libc = ctypes.CDLL('libc.so.6')
# 定义 ftok 函数原型
libc.ftok.argtypes = [ctypes.c_char_p, ctypes.c_int]
libc.ftok.restype = ctypes.c_int

# 定义 Sensor_Package 结构体
class Sensor_Package(ctypes.Structure):
    _fields_ = [
        ('joint_q', ctypes.c_float*3*4),# *3*4 = 4行3列  *4*3 = 3行4列
        ('joint_qd', ctypes.c_float*3*4),
        ('joint_tau', ctypes.c_float*3*4),
        ('joint_arm', ctypes.c_float*8),
        ('joint_arm_dq', ctypes.c_float*8),
        ('motor_output', ctypes.c_float*3*4),
        ('imu_euler', ctypes.c_float*3),
        ('imu_wxyz', ctypes.c_float *3), 
        ('imu_acc', ctypes.c_float*3),
        ('imu_orientation', ctypes.c_float*4),
        ('body_vel', ctypes.c_float*3),
        ('body_pos', ctypes.c_float*3),
        ('ankle_f', ctypes.c_float*4),
        ('is_ready_go', ctypes.c_char),
        ('footforce_r', ctypes.c_float*3*4)
    ]
class Servo_Package_Socket(ctypes.Structure):
    _fields_ = [
        ("tau_d", ctypes.c_float *3*4),
        ("motor_output_v", ctypes.c_float *3*4),
        ("motor_mode", ctypes.c_char *3*4),
        ("is_put_on_ground", ctypes.c_char),
        ("data_params", ctypes.c_float * 156)
    ]
class Servo_Package(ctypes.Structure):
    _fields_ = [
        ("joint_q_d", ctypes.c_float *3*4),
        ("joint_qd_d", ctypes.c_float *3*4),
        ("joint_tau_d", ctypes.c_float *3*4),
        ("joint_arm_d", ctypes.c_float *8),
        ("kp_arm", ctypes.c_float *8),
        ("kd_arm", ctypes.c_float *8),
        ("kp", ctypes.c_float *3*4),
        ("kd", ctypes.c_float *3*4),
        ("motor_enable", ctypes.c_int * 4),
        ("motor_mode", ctypes.c_char *3*4),
        ("is_put_on_ground", ctypes.c_char),
        ("cmd_info", ctypes.c_int)
]
class Send_Package_Socket(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int * 4),
        ("q", ctypes.c_float *3*4),
        ("q_d", ctypes.c_float *3*4),
        ("f", ctypes.c_float *3*4),
        ("f_d", ctypes.c_float *3*4),
        ("xyz", ctypes.c_float *3*4),
        ("xyz_d", ctypes.c_float *3*4),
        ("fxyz", ctypes.c_float *3*4),
        ("fxyz_d", ctypes.c_float *3*4),
        ("extra", ctypes.c_float * 32)
    ]
class Ocu_package(ctypes.Structure):
    _fields_ = [
        ("yaw_turn_dot", ctypes.c_float),
        ("x_des_vel", ctypes.c_float),
        ("y_des_vel", ctypes.c_float),
        ("gait_info_cmd", ctypes.c_uint8),
        ("kp", ctypes.c_float),
        ("kd", ctypes.c_float),
        ("update_count", ctypes.c_uint8)
    ]

class ShareInfo(ctypes.Structure):
    _fields_ = [
        ("sensor_package", Sensor_Package),
        ("sensor_package2", Sensor_Package),
        ("servo_package", Servo_Package),
        ("servo_package2", Servo_Package),
        ("ocu_package", Ocu_package)
    ]
class ShareMemory:
    def __init__(self, name="SHAREMEMORY_NAME"):
        self.name = name
        self.shmid = self.create_shared_memory(1, 1)

        # Attach the shared memory
        self.shmaddr = self.attach_shared_memory(self.shmid)

        # Assuming shareinfo is a part of the shared memory, map it using ctypes
        self.shareinfo = ctypes.cast(self.shmaddr, ctypes.POINTER(ShareInfo)).contents

    @staticmethod
    def get_instance(name="SHAREMEMORY_NAME"):
        if not hasattr(ShareMemory, "_instance"):
            ShareMemory._instance = ShareMemory(name)
        return ShareMemory._instance

    @staticmethod
    def create_shared_memory(key_id, size):
        shmid = shmget(key_id, size, 0o666 | 0o1000)  # IPC_CREAT = 0o01000
        if shmid == -1:
            raise OSError("Failed to create shared memory")
        return shmid

    def attach_shared_memory(self, shmid):
        shmaddr = shmat(shmid, None, 0)  # Attach for read and write
        if int(ctypes.cast(shmaddr, ctypes.c_void_p).value) == -1:
            raise OSError("Failed to attach shared memory")
        return shmaddr

    def put_to_share_mem(self, src_buffer, length):
        ctypes.memmove(self.shmaddr, ctypes.addressof(src_buffer), length)

    def get_from_share_mem(self, dst_buffer, length):
        ctypes.memmove(ctypes.addressof(dst_buffer), self.shmaddr, length)

    def share_mem_close(self, destroy_shm=False):
        shmdt(self.shmaddr)  # Detach the shared memory
        if destroy_shm:
            # If required, also remove the shared memory
            self.remove_shared_memory(self.shmid)

    def remove_shared_memory(self, shmid):
        # This would use a function like shmctl in C, to remove shared memory
        # Python equivalent might use some system call or custom implementation
        pass

    def __del__(self):
        self.share_mem_close(True)  # Ensure clean-up on deletion


def CreatShareMem():
    # 定义路径和项目 ID
    path = b'/home'  # 路径必须是字节字符串
    shm_id = 0x5a
    sem_id = shm_id + 2
    shm_key = (libc.ftok(path, shm_id))
    sem_key = (libc.ftok(path, sem_id))
    if shm_key  == -1:
        raise Exception("ftok 生成键值失败")

    print(f"shm_key: {hex(shm_key)}")
    print(f"sem_key: {hex(sem_key)}")
    # 创建信号量对象
    semaphore = sysv_ipc.Semaphore(sem_key, 0o1000, initial_value=1)        
    # 获取共享内存的ID
    shmid = shmget(shm_key, 2*1024*1024, 0o666 | 0o1000)  # IPC_CREAT = 0o1000
    # print('shmid :',shmid)
    # 将共享内存连接到当前进程的地址空间
    shmaddr = shmat(shmid, 0, 0)
    return shmaddr,semaphore

def CreatShareMem2():
    # 定义路径和项目 ID
    path = b'/home'  # 路径必须是字节字符串
    shm_id = 0xaa
    sem_id = shm_id + 2
    shm_key = (libc.ftok(path, shm_id))
    sem_key = (libc.ftok(path, sem_id))
    if shm_key  == -1:
        raise Exception("ftok 生成键值失败")

    print(f"shm_key: {hex(shm_key)}")
    print(f"sem_key: {hex(sem_key)}")
    # 创建信号量对象
    semaphore = sysv_ipc.Semaphore(sem_key, 0o1000, initial_value=1)        
    # 获取共享内存的ID
    shmid = shmget(shm_key, 2*1024*1024, 0o666 | 0o1000)  # IPC_CREAT = 0o1000
    # print('shmid :',shmid)
    # 将共享内存连接到当前进程的地址空间
    shmaddr = shmat(shmid, 0, 0)
    return shmaddr,semaphore

def GetFromShareMem(shmaddr,semaphore):
    # a = ShareInfo or()#size 1904  + coushu 13420
    # print('size',ctypes.sizeof(a))

    # 定义共享内存的大小
    #SHM_SIZE = 2*1024*1024

    # 定义共享内存的键值（假设为 IPC_PROJ_ID）
    #IPC_PROJ_ID = 0x5A0C0001
    
    # 获取信号量（进入临界区）
    try:
        semaphore.acquire()
        result_ptr = ctypes.cast(shmaddr, ctypes.POINTER(ShareInfo))
        # print('joint_q')
        # for i in range(4):
        #     print(result_ptr.contents.sensor_package.joint_q[i][0],
        #           result_ptr.contents.sensor_package.joint_q[i][1],
        #           result_ptr.contents.sensor_package.joint_q[i][2])
        # print('eluer')
        # print(result_ptr.contents.sensor_package.imu_euler[0],
        #           result_ptr.contents.sensor_package.imu_euler[1],
        #           result_ptr.contents.sensor_package.imu_euler[2])
        # print('joint_q2')
        # for i in range(4):
        #     print(result_ptr.contents.sensor_package2.joint_q[i][0],
        #           result_ptr.contents.sensor_package2.joint_q[i][1],
        #           result_ptr.contents.sensor_package2.joint_q[i][2])
        # print('eluer2')
        # print(result_ptr.contents.sensor_package2.imu_euler[0],
        #           result_ptr.contents.sensor_package2.imu_euler[1],
        #           result_ptr.contents.sensor_package2.imu_euler[2])
        return result_ptr.contents
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        semaphore.release()
    return result_ptr.contents

def PutToShareMem(shareinfo,package,shmaddr,semaphore):
    # 获取信号量（进入临界区
    semaphore.acquire()
    # 获取package在内存中偏移量
    offset = ctypes.addressof(package) - ctypes.addressof(shareinfo)
    try:
    # 写入结构体
        ctypes.memmove(shmaddr+offset, ctypes.addressof(package), ctypes.sizeof(package))
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 释放信号量（离开临界区）
        semaphore.release()