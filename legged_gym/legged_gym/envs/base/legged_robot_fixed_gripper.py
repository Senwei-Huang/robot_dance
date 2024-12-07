from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.go2.go2_dance_config import GO2DanceCfg_beat
from legged_gym.motion_loader.motion_loader_panda_fixed_gripper import motionLoaderPandaFixedGripper
from .legged_robot import LeggedRobot


class LeggedRobotPandaFixedGripper(LeggedRobot):
    def __init__(self, cfg: GO2DanceCfg_beat, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # 重新加载动作数据
        self.motion_loader = motionLoaderPandaFixedGripper(motion_files=self.cfg.env.motion_files, device=self.device,
                                          time_between_frames=self.dt, frame_duration=self.cfg.env.frame_duration)
        self.max_episode_length_s = self.motion_loader.trajectory_lens[0]
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # get arm_indices
        arm_name = self.cfg.asset.arm_name
        self.arm_indices = torch.zeros(len(arm_name), dtype=torch.long, device=self.device, requires_grad=False)
        self.arm_indices = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], arm_name)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        if not self.cfg.env.RSI:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)
        else:
            # 用于初始化
            traj_idxs = self.motion_loader.weighted_traj_idx_sample_batch(len(env_ids))
            init_times = np.zeros(len(env_ids), dtype=int)
            frames = self.motion_loader.get_full_frame_at_time_batch(traj_idxs, init_times)
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)


        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf


    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        包含amp的函数在这里就是初始化的时候把机器人的位置和关节角度设置成参考动作的状态
        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = torch.cat((self.motion_loader.get_joint_pose_batch(frames),
                                          self.motion_loader.get_arm_joint_pos_batch(frames)), dim=1)
        self.dof_vel[env_ids] = torch.cat((self.motion_loader.get_joint_vel_batch(frames),
                                           self.motion_loader.get_arm_joint_vel_batch(frames)), dim=1)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _reward_track_arm_dof_pos(self):
        if 0:
            print(torch.sum(torch.square(self.frames[:, 56:64] - self.dof_pos[:, 12:20]), dim=1))
            print(torch.exp(-0.1 * torch.sum(torch.square(self.frames[:, 56:62] - self.dof_pos[:, 12:18]), dim=1)))
            print(50*'#')
        return torch.exp(-0.1 * torch.sum(torch.square(self.frames[:, self.motion_loader.ARM_JOINT_POS_START_IDX:
                                                                      self.motion_loader.ARM_JOINT_POS_END_IDX] -
                                                       self.dof_pos[:, 12:18]), dim=1))

    def _reward_track_griper_dof_pos(self):
        if 0:
            # print(self.dof_pos[:, 18:20])
            # print(torch.sum(torch.square(self.frames[:, 62:64] - self.dof_pos[:, 18:20]), dim=1))
            print(torch.exp(-1000 * torch.sum(torch.square(self.frames[:, 62:64] - self.dof_pos[:, 18:20]), dim=1)))
            print(50*'_')
        return torch.exp(-1000 * torch.sum(torch.square(self.frames[:, 62:64] - self.dof_pos[:, 18:20]), dim=1))

    def _reward_track_arm_dof_vel(self):
        return torch.exp(-1 * torch.sum(torch.square(self.frames[:, self.motion_loader.ARM_JOINT_VEL_START_IDX:
                                                                    self.motion_loader.ARM_JOINT_VEL_END_IDX] -
                                                     self.dof_vel[:, 12:18]), dim=1))

    def _reward_track_arm_pos(self):
        arm_pos = self.rb_states[:, self.arm_indices, 0:3].view(self.num_envs, -1) - self.base_pos
        if 0:
            print(f"arm pos ref is: {self.frames[:, 49:52]}")
            print(f"arm pos is: {arm_pos}")
        temp = torch.exp(-100 * torch.sum(torch.square(self.frames[:, 49:52] - arm_pos), dim=1))
        return temp

    def _reward_track_arm_rot(self):
        arm_rot = self.rb_states[:, self.arm_indices, 3:7].view(self.num_envs, -1)
        if self.cfg.env.debug:
            print(f"arm rot ref is : {self.frames[:, 52:56]}")
            print(f"arm rot is : {arm_rot}")
        return torch.exp(-10 * torch.sum(torch.square(self.frames[:, 52:56] - arm_rot), dim=1))

    def _reward_survival(self):
        "如果这个回合存活，给一个奖励"
        survival_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) < 1.,
                                 dim=1)
        # print(survival_buf)
        return survival_buf

    def _reward_test(self):
        test_buf = torch.ones_like(self.episode_length_buf)
        return test_buf
