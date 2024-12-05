

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
from legged_gym.motion_loader.motion_loader import motionLoader
from .legged_robot import LeggedRobot


class LeggedRobotTrans(LeggedRobot):
    def __init__(self, cfg: GO2DanceCfg_beat, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if self.cfg.env.dance_sequence is not None:
            self.temp = 0
            self.traj_idxs = np.full(self.num_envs, self.cfg.env.dance_sequence[0])
        else:
            # 随机选择num_env个从0到5的6个数字，分别代表self.dance_task_name_list中的六个舞蹈动作
            self.traj_idxs = np.random.randint(0, 5, (self.num_envs,))
            # self.traj_idxs = np.ones((self.num_envs), dtype=int)*5

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.max_length = np.ceil(self.motion_loader.trajectory_lens / self.dt)
        self.max_primitive_length = self.max_length[self.traj_idxs]


    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.max_primitive_length = self.max_length[self.traj_idxs]
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False),
                                                 torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions, dance_actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.dance_actions = dance_actions
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 在这里计算对应时刻的参考位置，放在这里self.episode_length_buf的范围是0-69，放到下面就是1-70，越界了
        # self.traj_idxs = self.motion_loader.weighted_traj_idx_sample_batch(self.num_envs)
        self.check_primitive_termination()
        a = self.episode_length_buf.cpu().numpy() - (self.max_primitive_length - self.max_length[self.traj_idxs])
        # print(self.episode_length_buf[54])
        self.episode_time = (self.episode_length_buf.cpu().numpy() -
                             (self.max_primitive_length -
                              self.max_length[self.traj_idxs])) * self.dt

        self.frames = self.motion_loader.get_full_frame_at_time_batch(self.traj_idxs, self.episode_time)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

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
            reset_traj_idxs = self.traj_idxs[env_ids.cpu().numpy()]
            # init_times = self.motion_loader.traj_time_sample_batch(reset_traj_idxs)#ndarray
            init_times = np.zeros(len(reset_traj_idxs), dtype=int)
            frames = self.motion_loader.get_full_frame_at_time_batch(reset_traj_idxs, init_times)
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.max_primitive_length[env_ids.cpu().numpy()] = 0
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

    def check_primitive_termination(self):
        """检查一个舞蹈动作是否结束"""
        self.primitive_time_out = self.episode_length_buf.cpu() >= torch.from_numpy(self.max_primitive_length)
        # 需要更新self.traj_idxs的环境
        env_ids = self.primitive_time_out.nonzero(as_tuple=False).flatten()
        if self.cfg.env.dance_sequence is not None:
            if len(env_ids) != 0:
                self.temp += 1
            self.traj_idxs[self.primitive_time_out] = np.full(len(env_ids), self.cfg.env.dance_sequence[self.temp])
        else:
            # 这里是如果有需要重新选择动作的环境，那就采样对应环境的数量这么多的随机数，赋值给self.traj_idxs对应的位置
            self.traj_idxs[self.primitive_time_out] = np.random.randint(0, 5, (len(env_ids)))
        # 如果一个动作结束了，那么在self.max_primitive_length加上新的动作的轨迹时长，以便下一个动作对比是否结束
        self.max_primitive_length[self.primitive_time_out] += self.max_length[self.traj_idxs[self.primitive_time_out]]

    def _reward_survival(self):
        "如果这个回合存活，给一个奖励"
        survival_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) < 1.,
                                 dim=1)
        return survival_buf

    def _reward_no_action(self):
        "鼓励policy输出动作为0的actions"
        return torch.exp(-0.1 * torch.sum(torch.square(self.actions - self.dance_actions), dim=1))
