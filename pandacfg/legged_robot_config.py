# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from posixpath import relpath
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from .base_config import BaseConfig
import torch.nn as nn
class LeggedRobotCfg(BaseConfig):
    class play:
        load_student_config = False
        mask_priv_obs = False
    
    class env:
        num_envs = 4096  # 6144 4096

        n_scan = 132
        n_priv = 3 + 3 + 3  # 9
        n_priv_latent = 4 + 1 + 12 + 12  # 29
        n_proprio = 3 + 2 + 3 + 4 + 36 + 5  # 53
        history_len = 10

        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv  # n_scan + n_proprio + n_priv  # 187 + 47 + 5 + 12  # 753
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        obs_type = "og"

        history_encoding = True
        reorder_dofs = True
        
        # action_delay_range = [0, 5]
        # additional visual inputs 

        include_foot_contacts = True
        
        randomize_start_pos = False  # False x y
        rand_pos_range = [-0.3, 0.3] # -+0.3 -+1.0
        randomize_start_x = True  # False
        rand_x_range = 0.3
        randomize_start_y = False  # False
        rand_y_range = 0.5

        randomize_start_vel = False  # False  not use
        rand_vel_range = 0.

        randomize_start_ori = True
        randomize_start_roll = True
        rand_roll_range = 0.2
        randomize_start_pitch = True  # False
        rand_pitch_range = 0.3  # 1.6
        randomize_start_yaw = False  # False
        rand_yaw_range = 1.2

        contact_buf_len = 100

        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

    class depth:
        use_camera = False
        camera_num_envs = 192  # distillation策略训练（使用相机）创建的环境数量
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        # position = [0.436, 0, 0.072]  #front camera  A1 [0.27, 0, 0.03]   [0.448, 0, 0.072]  [0.4412, 0, -0.0550]
        angle = [-5, 5]  # positive pitch down 相机俯仰角度 给正负5度随机均匀分布误差
        angle_error = [-5, 5]  # 角度误差

        # 上相机的配置
        camera1_config = {
            'width': 106,
            'height': 60,
            'enable_tensors': True,
            'horizontal_fov': 87,
            'position': [0.436, 0, 0.072], 
            'angle': 0  # 无倾斜
        }
        
        # 下相机的配置
        camera2_config = {
            'width': 106,
            'height': 60,
            'enable_tensors': True,
            'horizontal_fov': 87,
            'position': [0.4412, 0, -0.0550], 
            'angle': 28  # 俯仰 (度)
        }

        update_interval = 5  # 5 works without retraining, 8 worse

        resized = (87, 58)  # original (1280, 720) --> (106, 60)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.01  # 0.0 0.05 0.02 0.01 0.005
        
        scale = 1
        invert = True

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 1.2
    
    class noise:
        add_noise = True  # False
        add_height_noise = False
        noise_level = 1.0 # scales other values  1.0
        quantize_height = True
        # class noise_scales:
        #     rotation = 0.2  # 0.0 0.05 0.2 0.05 狗上IMU的roll和pitch真实误差0.1度左右
        #     dof_pos = 0.1  # 0.01 0.1 0.01
        #     dof_vel = 0.5  # 0.05 0.2 0.5 1.5
        #     lin_vel = 0.5  # 0.05
        #     ang_vel = 0.5  # 0.05 0.2 0.1 0.5 0.2
        #     gravity = 0.2  # 0.02 0.05
        #     height_measurements = 0.2  # 0.02 0.2 0.1
        class noise_scales:
            rotation = 0.05
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.05
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.05 # [m] influence computation time by a lot 尺寸缩放系数
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005 # [m]  尺寸缩放系数
        border_size = 5 # [m] 整个地形模块的边缘大小
        height = [0.02, 0.06]  # 粗糙地形高度范围
        simplify_grid = False
        gap_size = [0.02, 0.1]  # [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]  # [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0  # 1.0 0.7
        dynamic_friction = 1.0 # 1.0 0.6 
        restitution = 0.0  # 0.0 0.4
        measure_heights = True
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.05  # 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 18  # 18 20
        terrain_width = 4  # 4 15 每个地形模块的宽度  y方向  指向屏幕内
        terrain_length_stairs = 20  # 楼梯地形模块的长度 20 25
        terrain_width_stairs = 10  # 楼梯地形模块的宽度 15 25
        use_diff_terrain_size = 1  # 不同地形模块采用不一样的尺寸
        terrain_dict = {"smooth slope": 0.,   # 0
                        "rough slope up": 0.0,   # 1
                        "rough slope down": 0.0,   # 2
                        "rough stairs up": 0.,    # 3
                        "rough stairs down": 0.,   # 4
                        "discrete": 0.,    # 5
                        "stepping stones": 0.0,   # 6
                        "gaps": 0.,    # 7
                        "smooth flat": 0,   # 8
                        "pit": 0.0,  # 9
                        "wall": 0.0,  # 10
                        "platform": 0.,  # 11
                        "large stairs up": 0.,  # 12  0. 0.2
                        "large stairs down": 0.,  # 13  0. 0.2
                        "parkour": 0.2,  # 14
                        "parkour_hurdle": 0.2,  # 跑酷栏  15
                        "parkour_flat": 0.2,  # 16 平地
                        "parkour_step": 0.2,  #  17 0.2
                        "parkour_gap": 0.2,  #  18 0.2
                        "demo": 0.0,    # 19
                        "step_1": 0.0,  # 20 0.1
                        "step_2": 0.0,  # 21 0.1
                        "gap_1": 0.0}  # 22 0.1

        terrain_proportions = list(terrain_dict.values())
        n_terrain_types = 0  # 地形种类
        n_stairs_types = 0  # 楼梯地形种类
        for i in terrain_proportions:
            if i > 0.0:
                n_terrain_types += 1
        if terrain_dict["rough stairs up"] > 0.0:
            n_stairs_types += 1
        if terrain_dict["rough stairs down"] > 0.0:
            n_stairs_types += 1
        if terrain_dict["large stairs up"] > 0.0:
            n_stairs_types += 1
        if terrain_dict["large stairs down"] > 0.0:
            n_stairs_types += 1
        num_copy = 4  # 每类地形复制多少份 8 4 2 1
        num_rows = 10 # number of terrain rows (levels)  # spreaded is benifitiall !   横向复制地形  从右至左 难度逐渐增大
        num_cols = n_terrain_types*num_copy # 40 number of terrain cols (types)  纵向复制地形   一列上有五类地形（尺寸随机）
        # num_cols = 20 # 50 number of terrain cols (types)  纵向复制地形   一列上有五类地形（尺寸随机）
        print("num_copy: ", num_copy)
        print("n_terrain_types: ", n_terrain_types)
        print("n_stairs_types: ", n_stairs_types)
        print("num_rows: ", num_rows)
        print("num_cols: ", num_cols)

        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True
        num_goals = 8

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 6. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.4  # 为了防止转向速度过低时的微小扰动或噪音影响系统的稳定性，如果绝对值没有超过设定的阈值，则将其设为0。
        # Easy ranges
        class ranges:
            lin_vel_x = [0.0, 1.5] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

        # Easy ranges  不开启命令课程学习时的参数
        class max_ranges:
            lin_vel_x = [0.3, 0.8]  # min max [m/s] [0.3, 0.8]
            lin_vel_x_flat = [0.0, 1.5]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # [0.15, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0., 0.]  # min max [rad/s] heading_command = True时不使用
            heading_yaw = [-1., 1.]  # [rad/s] 根据航向误差计算出来的偏航角速度范围 (in heading mode ang_vel_yaw is recomputed from heading error)
            heading = [-1.6, 1.6]

        class crclm_incremnt:  # 未使用
            lin_vel_x = 0.1 # min max [m/s]
            lin_vel_y = 0.1  # min max [m/s]
            ang_vel_yaw = 0.1 # min max [rad/s]
            heading = 0.5

        waypoint_delta = 0.7

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 仿真运行多少个周期 抽取一次新的动作值

    class asset:
        file = ""
        hip_dof_name = "None"
        thigh_dof_name = "None"
        calf_dof_name = "None"
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.005  # 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.4, 2.0]  # [0.6, 2.] [0.4, 1.3] 

        randomize_restitution = True
        restitution_range = [0.0, 0.4]

        randomize_joint_friction = False
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.001, 0.05]  # [0.01, 1.15]

        randomize_joint_damping = False
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.01, 0.1]  # [0.3, 1.5]

        randomize_joint_armature = True
        randomize_joint_armature_each_joint = False
        joint_armature_range = [0.0001, 0.05]   # Factor [0.0001, 0.05] [0.001, 0.1]

        randomize_armature = False
        armature_value = [0.0354, 0.022, 0.0513]
        armature_range = [-0.01, 0.01]

        add_lag = False  # action delay
        randomize_lag_timestep = False
        lag_timestep_range = [0, 1, 2]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # Factor [0.8, 1.2] [0.2, 2] [0.5, 1.5]
        damping_multiplier_range = [0.8, 1.2]

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]
        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        randomize_coulomb_friction = True
        joint_coulomb_range = [0.1, 1.0]
        joint_viscous_range = [0.1, 0.9]

        randomize_base_mass = True 
        added_mass_range = [0., 5.] # A1 12kg  [0., 3.]  [0., 16.]Panda7 65kg

        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        randomize_com = True
        com_displacement_range = [-0.2, 0.2]

        randomize_link_mass = True
        added_link_mass_range = [0.9, 1.1]

        push_robots = True
        push_interval_s = 8
        push_vel = True
        max_push_vel_xy = 0.5
        push_ang = True
        max_push_ang_vel = 0.6

        delay_update_global_steps = 24 * 2000  # 24 * 8000
        action_delay = True  # False True
        action_curr_step = [0, 1, 2]  # [1, 1]  [0, 1, 2] [1, 2]
        action_curr_step_scratch = [0, 1, 2]  # [0, 1]  [0, 1, 2]
        action_delay_view = 1
        action_buf_len = 8

        obs_delay = False  # True
        obs_buf_len = 4
        
    class rewards:
        class scales:
            # tracking rewards
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            # regularization rewards
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.
            dof_acc = -2.5e-7
            collision = -10.
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.00001
            hip_pos = -0.5
            dof_error = -0.04
            feet_stumble = -1
            feet_edge = -1
            
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma) 未使用
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1  # 未使用
        soft_torque_limit = 0.4  # 未使用
        base_height_target = 1.  # 未使用
        target_feet_height = 0.05  # 未使用
        max_contact_force = 40. # forces above this value are penalized 未使用

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005  # 仿真频率 0.005 0.001
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
 
    class policy:
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [64, 20]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
    
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
    
    class depth_encoder:
        if_depth = LeggedRobotCfg.depth.use_camera
        depth_shape = LeggedRobotCfg.depth.resized
        buffer_len = LeggedRobotCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24

    class estimator:
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = LeggedRobotCfg.env.n_priv
        num_prop = LeggedRobotCfg.env.n_proprio
        num_scan = LeggedRobotCfg.env.n_scan

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 50000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'rough_panda7'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt