from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Panda7DnaceCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]  [0.0, 0.0, 0.6]  x为机器人的前进正方向
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
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 150., 'joint3': 150,  # 150 150 150  150 600 150
                     'joint4': 20., 'joint5': 15., 'joint6': 10.,
                     'joint7': 10., 'joint8': 10.}  # [N*m/rad]
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 12., 'joint2': 12., 'joint3': 12.,  # 12. 12. 12.  2 2 2
                   'joint4': 0.8, 'joint5': 1., 'joint6': 1.,  # 0.8 1. 1.   0.1 1. 1.
                   'joint7': 1., 'joint8': 1.}  # [N*m*s/rad]
        action_scale = 0.25  # action scale: target angle = actionScale * action + defaultAngle

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_arm/urdf/panda7_arm.urdf'
        name = "panda7_arm"
        foot_name = "FOOT"
        arm_name = "arm_link"
        arm_link6_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        only_positive_rewards = False
        class scales:
            # regularization reward
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            lin_vel_z = -0.  # -1.0
            ang_vel_xy = -0.
            torques = -0.00001
            action_rate = -0.1
            collision = -5.
            dof_pos_limits = -10.0
            feet_air_time = 0.
            
            survival = 0.
            test = 0.  

    class env(LeggedRobotCfg.env):
        motion_files = None
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_observations = 60  # 94 63
        num_privileged_obs = 132
        num_actions = 18  # 12
        # debug = True


class Panda7DanceCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 10000  # number of policy updates
        experiment_name = 'panda7_arm_dance'


# *********************** Panda7_arm fixed gripper Dance Beat *******************************************
class Panda7DanceBeatCfg(Panda7DnaceCfg):
    class asset(Panda7DnaceCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_arm/urdf/panda7_arm_fixed_gripper.urdf'

    class rewards(Panda7DnaceCfg.rewards):
        class scales(Panda7DnaceCfg.rewards.scales):
            # 模仿奖励
            track_root_pos = 0.
            track_root_height = 0.5
            track_root_rot = 2.0
            track_lin_vel_ref = 1.0
            track_ang_vel_ref = 1.0
            track_toe_pos = 10.
            track_dof_pos = 1
            track_dof_vel = 1
            
            # 机械臂
            track_arm_dof_pos = 1
            track_arm_dof_vel = 1
            track_arm_end_pos = 0
            track_arm_end_rot = 0
            track_griper_dof_pos = 0

    class env(Panda7DnaceCfg.env):
        motion_files = "opti_traj/output_panda_fixed_gripper/panda_beat.txt"


class Panda7DanceBeatCfgPPO(Panda7DanceCfgPPO):
    class runner(Panda7DanceCfgPPO.runner):
        experiment_name = ('panda7_arm_fixed_gripper_dance_beat')


# *********************** Panda7_arm fixed gripper Dance Swing *******************************************
class Panda7DanceSwingCfg(Panda7DnaceCfg):
    class asset(Panda7DnaceCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_arm/urdf/panda7_arm_fixed_gripper.urdf'

    class rewards(Panda7DnaceCfg.rewards):
        class scales(Panda7DnaceCfg.rewards.scales):
            # 模仿奖励
            track_root_pos = 0.
            track_root_height = 0.5
            track_root_rot = 2.0
            track_lin_vel_ref = 1.0
            track_ang_vel_ref = 1.0
            track_toe_pos = 10.
            track_dof_pos = 1
            track_dof_vel = 1
            
            # 机械臂
            track_arm_dof_pos = 1
            track_arm_dof_vel = 2
            track_arm_end_pos = 10.  # 0
            track_arm_end_rot = 10.  # 0
            track_griper_dof_pos = 0

    class env(Panda7DnaceCfg.env):
        motion_files = "opti_traj/output_panda_fixed_gripper/panda_swing.txt"


class Panda7DanceSwingCfgPPO(Panda7DanceCfgPPO):
    class runner(Panda7DanceCfgPPO.runner):
        experiment_name = ('panda7_arm_fixed_gripper_dance_swing')
        

class Panda7DanceTurnJumpCfg(Panda7DnaceCfg):
    class rewards(Panda7DnaceCfg.rewards):
        class scales(Panda7DnaceCfg.rewards.scales):
            lin_vel_z = -0
            survival = 1
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 1
            track_root_height = 0
            track_root_rot = 1
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 10
            # jump reward
            jump = 5
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 0
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(Panda7DnaceCfg.env):
        motion_files = "opti_traj/output_panda_fixed_gripper/panda_turn_and_jump.txt"


class Panda7DanceTurnJumpCfgPPO(Panda7DanceCfgPPO):
    class runner(Panda7DanceCfgPPO.runner):
        experiment_name = 'panda7_fixed_gripper_turn_and_jump'