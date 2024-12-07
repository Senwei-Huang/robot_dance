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
                     'joint1': 150., 'joint2': 600., 'joint3': 150,  # 150 150 150
                     'joint4': 20., 'joint5': 15., 'joint6': 10.,
                     'joint7': 10., 'joint8': 10.}  # [N*m/rad]
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 2., 'joint2': 2., 'joint3': 2.,  # 12. 12. 12.
                   'joint4': 0.1, 'joint5': 1., 'joint6': 1.,  # 0.8 1. 1.
                   'joint7': 1., 'joint8': 1.}  # [N*m*s/rad]
        action_scale = 0.25  # action scale: target angle = actionScale * action + defaultAngle
        decimation = 4  # decimation: Number of control action updates @ sim DT per policy DT

    class domain_rand(LeggedRobotCfg.domain_rand):
        friction_range = [0.4, 2.0]
        restitution_range = [0.0, 0.4]

        max_push_ang_vel = 0.6
        randomize_com = True
        com_displacement_range = [-0.2, 0.2]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]    # Factor

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = True
        added_link_mass_range = [0.9, 1.1]

        randomize_motor_offset = True
        # Offset to add to the motor angles
        motor_offset_range = [-0.035, 0.035]

        randomize_joint_friction = False
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.3]
        joint_5_friction_range = [0.5, 1.3]
        joint_6_friction_range = [0.01, 1.15]
        joint_7_friction_range = [0.01, 1.15]
        joint_8_friction_range = [0.01, 1.15]
        joint_9_friction_range = [0.5, 1.3]
        joint_10_friction_range = [0.5, 1.3]

        randomize_joint_damping = False
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.3, 1.5]
        joint_1_damping_range = [0.3, 1.5]
        joint_2_damping_range = [0.3, 1.5]
        joint_3_damping_range = [0.3, 1.5]
        joint_4_damping_range = [0.9, 1.5]
        joint_5_damping_range = [0.9, 1.5]
        joint_6_damping_range = [0.3, 1.5]
        joint_7_damping_range = [0.3, 1.5]
        joint_8_damping_range = [0.3, 1.5]
        joint_9_damping_range = [0.9, 1.5]
        joint_10_damping_range = [0.9, 1.5]

        randomize_joint_armature = True
        randomize_joint_armature_each_joint = False
        joint_armature_range = [0.0001, 0.05]     # Factor
        joint_1_armature_range = [0.0001, 0.05]
        joint_2_armature_range = [0.0001, 0.05]
        joint_3_armature_range = [0.0001, 0.05]
        joint_4_armature_range = [0.0001, 0.05]
        joint_5_armature_range = [0.0001, 0.05]
        joint_6_armature_range = [0.0001, 0.05]
        joint_7_armature_range = [0.0001, 0.05]
        joint_8_armature_range = [0.0001, 0.05]
        joint_9_armature_range = [0.0001, 0.05]
        joint_10_armature_range = [0.0001, 0.05]

        add_lag = False
        randomize_lag_timesteps = False
        lag_timesteps_range = [0, 1]

        randomize_coulomb_friction = True
        joint_coulomb_range = [0.1, 1.0]
        joint_viscous_range = [0.10, 0.90]

        delay_update_global_steps = 24 * 2000
        action_delay = False
        action_curr_step = [1, 2]
        action_curr_step_scratch = [0, 1]
        action_delay_view = 1
        action_buf_len = 8

    class noise(LeggedRobotCfg.noise):
        add_noise = True  # False
        add_height_noise = False
        noise_level = 1.0  # scales other values
        quantize_height = True

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            rotation = 0.05
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.05
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_arm/urdf/panda7_arm.urdf'
        name = "panda7_arm"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.55
        target_feet_height = 0.05

    class env(LeggedRobotCfg.env):
        motion_files = None
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 18
        num_observations = 131
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
            # regularization reward
            torques = -0.00001  #
            dof_pos_limits = -10.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 2  #

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
            track_toe_pos = 5  #
            # 机械臂
            track_arm_dof_pos = 1
            track_arm_dof_vel = 1
            track_arm_pos = 0
            track_arm_rot = 0

    class env(Panda7DanceCfgPPO.env):
        motion_files = "opti_traj/output_panda_fixed_gripper/panda_beat.txt"


class Panda7DanceBeatCfgPPO(Panda7DanceCfgPPO):
    class runner(Panda7DanceCfgPPO.runner):
        experiment_name = ('panda7_arm_fixed_gripper_dance_beat')
        # resume_path = 'legged_gym/logs/panda7_beat/Dec01_20-31-14_/model_1500.pt'
