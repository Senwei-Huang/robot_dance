from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2DanceCfg_trans(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "thigh"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0

            lin_vel_z = -0
            track_root_pos = 0.
            track_root_rot = 0
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 0
            termination = -1.0
            survival = 0.5
            no_action = 2.

    class env(LeggedRobotCfg.env):
        # 打拍子参考动作
        motion_files = "opti_traj/output_json"
        # motion_files = "opti_traj/json"
        # 这个值没有使用，仅仅是作为motionLoader实例化时的参数，因为使用的legged_robot_transition继承自
        # legged_robot，其读取的文件类型是txt，没有时间参数；而由于多文件的时间间隔不完全一致，因此使用json文件
        # 记录时间间隔信息
        frame_duration = 1 / 100
        RSI = 1  # 参考状态初始化
        episode_length_s = 20
        dance_sequence = None


class GO2DanceCfg_trans_PPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_trans'
        dance_task_list = ["go2_dance_beat", "go2_dance_wave", "go2_dance_pace", "go2_dance_turn_and_jump",
                           "go2_dance_trot", "go2_dance_swing"]
        # dance_task_list = ["go2_dance_wave"]
        resume_path = 'legged_gym/logs/go2_trans/Nov25_17-06-09_/model_1500.pt'
        # resume_path = 'legged_gym/log/GO2_new/trans/Nov25_11-05-24_/model_200.pt'
