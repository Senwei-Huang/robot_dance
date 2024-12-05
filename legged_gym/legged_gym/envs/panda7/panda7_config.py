from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class panda7BeatCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.55] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip':150.,'thigh':150.,'calf':150.,
                     'joint1':150.,'joint2':150.,'joint3':150,
                     'joint4':20.,'joint5':15.,'joint6':10.,
                     'joint7':10.,'joint8':10.}  # [N*m/rad]#
        damping = {'hip': 2.0,'thigh': 2.0,'calf': 2.0,
                   'joint1': 12., 'joint2': 12., 'joint3': 12.,
                   'joint4': 0.8, 'joint5': 1., 'joint6': 1.,
                   'joint7': 1., 'joint8': 1.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7/urdf/panda7_nleg_arm_1008.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales( LeggedRobotCfg.rewards.scales ):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -100.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 1
            test = 0
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 1.
            track_root_rot = 1.
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 5
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 5
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0


    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda/panda_beat.txt"
        frame_duration = 1/50
        RSI = 1  # 参考状态初始化
        num_actions = 20
        num_observations = 141
        # debug = True


class panda7BeatCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = ('panda7_beat')
        # resume_path = 'legged_gym/logs/panda7_beat/Dec01_20-31-14_/model_1500.pt'


class panda7TrotCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
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

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 150., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.,
                     'joint7': 10., 'joint8': 10.}  # [N*m/rad]#
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 12., 'joint2': 12., 'joint3': 12.,
                   'joint4': 0.8, 'joint5': 1., 'joint6': 1.,
                   'joint7': 1., 'joint8': 1.}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7/urdf/panda7_nleg_arm_1008.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -10.0
            action_rate = -0.1
            collision = -10.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 1
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_rot = 0
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 3
            track_dof_vel = 3
            track_toe_pos = 8
            # 机械臂
            track_arm_dof_pos = 0
            track_griper_dof_pos = 5
            track_arm_dof_vel = 5
            track_arm_pos = 5
            track_arm_rot = 0



    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda/panda_trot.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 20
        num_observations = 141
        # debug = True


class panda7TrotCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7_trot')


class panda7SwingCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
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

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 150., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.,
                     'joint7': 10., 'joint8': 10.}  # [N*m/rad]#
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 12., 'joint2': 12., 'joint3': 12.,
                   'joint4': 0.8, 'joint5': 1., 'joint6': 1.,
                   'joint7': 1., 'joint8': 1.}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7/urdf/panda7_nleg_arm_1008.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -100.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 1
            test = 0
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 0.5
            track_root_rot = 1.
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 1
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 5
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda/panda_swing.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 20
        num_observations = 141
        # debug = True


class panda7SwingCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7_swing')
        # resume_path = 'legged_gym/logs/panda7_beat/Dec01_20-31-14_/model_1500.pt'


class panda7Turn_and_jumpCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
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

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 150., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.,
                     'joint7': 10., 'joint8': 10.}  # [N*m/rad]#
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 12., 'joint2': 12., 'joint3': 12.,
                   'joint4': 0.8, 'joint5': 1., 'joint6': 1.,
                   'joint7': 1., 'joint8': 1.}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7/urdf/panda7_nleg_arm_1008.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -100.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 1
            test = 0
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 1
            track_root_height = 0
            track_root_rot = 1.2
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 1.5
            # jump reward
            jump = 1
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 5
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda/panda_turn_and_jump.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 20
        num_observations = 141
        # debug = True


class panda7Turn_and_jumpCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7_turn_and_jump')
        # resume_path = 'legged_gym/logs/panda7_beat/Dec01_20-31-14_/model_1500.pt'

class panda7WaveCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
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

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 150., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.,
                     'joint7': 10., 'joint8': 10.}  # [N*m/rad]#
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 12., 'joint2': 12., 'joint3': 12.,
                   'joint4': 0.8, 'joint5': 1., 'joint6': 1.,
                   'joint7': 1., 'joint8': 1.}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7/urdf/panda7_nleg_arm_1008.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -100.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 1
            test = 0
            orientation = -0.0
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 1.
            track_root_rot = 1.2
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 8
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 5
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda/panda_wave.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 20
        num_observations = 141
        # debug = True


class panda7WaveCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7_wave')

class panda7PaceCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
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

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 150., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.,
                     'joint7': 10., 'joint8': 10.}  # [N*m/rad]#
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 12., 'joint2': 12., 'joint3': 12.,
                   'joint4': 0.8, 'joint5': 1., 'joint6': 1.,
                   'joint7': 1., 'joint8': 1.}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7/urdf/panda7_nleg_arm_1008.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -100.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 1
            test = 0
            orientation = -0.0
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 0
            track_root_rot = 0
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 3
            track_dof_vel = 3
            track_toe_pos = 8
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 5
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda/panda_pace.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 20
        num_observations = 141
        # debug = True


class panda7PaceCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7_pace')

