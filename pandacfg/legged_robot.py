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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import random
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from scipy.spatial.transform import Rotation as R
from .legged_robot_config import LeggedRobotCfg

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.leg_num = 4
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.n_proprio, device=self.device)  # 53
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0:  3] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[3:  5] = noise_scales.rotation                            # imu obs
        noise_vec[5: 13] = 0.                                               # commands
        noise_vec[13:25] = noise_scales.dof_pos * self.obs_scales.dof_pos   # joint pos obs
        noise_vec[25:37] = noise_scales.dof_vel * self.obs_scales.dof_vel   # joint vel obs
        noise_vec[37:49] = 0.                                               # last action
        noise_vec[49:53] = 0.                                               # contact
        return noise_vec
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self.reindex(actions)

        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        # print("self.cfg.domain_rand.action_delay: ", self.cfg.domain_rand.action_delay)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            indices = -self.delay -1
            # print("indices: ", indices)  # -2
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6
        # print("self.privileged_obs_buf: ", self.privileged_obs_buf)
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_history_observations(self):
        return self.obs_history_buf
    
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        # depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image += self.cfg.depth.dis_noise * torch.rand_like(depth_image) - self.cfg.depth.dis_noise/2
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
                                                                
            
            if i==-1:
              print("depth_image_: ", depth_image_)
              print("depth_image_.shape: ", depth_image_.shape)  # (60, 106)
              print("type(depth_image_): ", type(depth_image_))
              
              depth_image = gymtorch.wrap_tensor(depth_image_)
              print("depth_image_: ", depth_image_)
              print("depth_image_.shape: ", depth_image_.shape)  # (60, 106)
              print("type(depth_image_): ", type(depth_image_))

              depth_image = self.process_depth_image(depth_image, i)
              print("depth_image: ", depth_image)
              print("depth_image.shape: ", depth_image.shape)  # torch.Size([58, 87])
              print("type(depth_image): ", type(depth_image))

              depth_buffer = depth_image.to(self.device).unsqueeze(0)
              print("depth_buffer: ", depth_buffer)
              print("depth_buffer.shape: ", depth_buffer.shape)  # torch.Size([1, 58, 87])
              print("type(depth_buffer): ", type(depth_buffer))
            
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)

        self.gym.end_access_image_tensors(self.sim)

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        # print("self.cur_goals: ", self.cur_goals)
        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        
        # self._update_jump_schedule()
        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            # self._draw_height_samples()
            self._draw_goals()
            self._draw_feet()
            if self.cfg.depth.use_camera:
                # window_name = "Depth Image Up"
                # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                # cv2.imshow("Depth Image Up", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                # cv2.waitKey(1)
                window_name = "Depth Image Down"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image Down", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)

    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]  # LF LH RF RH --> LH LF RH RF

    def reindex(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        roll_cutoff = torch.abs(self.roll) > 1.5
        pitch_cutoff = torch.abs(self.pitch) > 1.5
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        height_cutoff = (self.root_states[:, 2] < -0.25) & (self.env_class != 14)  # -0.25 -10.0

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf |= reach_goal_cutoff

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff

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
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # Randomize joint parameters:
        self.randomize_motor_props(env_ids)
        self.randomize_dof_props(env_ids)
        self._refresh_actor_dof_props(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.

        if self.cfg.domain_rand.obs_delay:
            self.base_ang_vel_buf[env_ids, :, :] = 0.
            self.imu_obs_buf[env_ids, :, :] = 0.
            self.delta_yaw_buf[env_ids, :, :] = 0.
            self.delta_next_yaw_buf[env_ids, :, :] = 0.
            self.commands_buf[env_ids, :, :] = 0.
            self.dof_pos_buf[env_ids, :, :] = 0.
            self.dof_vel_buf[env_ids] = 0

        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ 
        Computes observations
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        if self.global_counter % 5 == 0:
            self.delta_yaw = self.target_yaw - self.yaw
            self.delta_next_yaw = self.next_target_yaw - self.yaw
        # print("self.global_counter: ", self.global_counter)
        # print("self.delta_yaw: ", self.delta_yaw)
        # print("self.env_class: ", self.env_class)
        # print("self.env_class.shape: ", self.env_class.shape)
        # print("(self.env_class != 17).float()[:, None]: ", (self.env_class != 17).float()[:, None])  # tensor([[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]], device='cuda:0')
        # print("(self.env_class != 17).float()[:, None]: ", (self.env_class != 17).float()[:, None].shape)
        # print("self.contact_filt: ", self.contact_filt)
        if self.cfg.domain_rand.obs_delay:
            obs_buf = torch.cat((#skill_vector, 
                              self.base_ang_vel_buf  * self.obs_scales.ang_vel,             # [1,3] 0 1 2 
                              imu_obs,                                                  # [1,2] 3 4 roll pitch
                              0*self.delta_yaw[:, None],                                # 5
                              self.delta_yaw[:, None],                                  # 6
                              self.delta_next_yaw[:, None],                             # 7
                              0*self.commands[:, 0:2],                                  # 8 9
                              self.commands[:, 0:1],                                    # [1,1] 10
                              (self.env_class != 17).float()[:, None],                  # 11
                              (self.env_class == 17).float()[:, None],                  # 12
                              self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),  # 13 14 ... 24
                              self.reindex(self.dof_vel * self.obs_scales.dof_vel),     # 25 ... 36
                              self.reindex(self.action_history_buf[:, -1]),             # 37 ... 48
                              0*self.reindex_feet(self.contact_filt.float()-0.5),       # 49 ... 52
                              ), dim=-1)
        else:
            obs_buf = torch.cat((#skill_vector, 
                                self.base_ang_vel  * self.obs_scales.ang_vel,             # [1,3] 0 1 2 
                                imu_obs,                                                  # [1,2] 3 4 roll pitch
                                0*self.delta_yaw[:, None],                                # 5
                                self.delta_yaw[:, None],                                  # 6
                                self.delta_next_yaw[:, None],                             # 7
                                0*self.commands[:, 0:2],                                  # 8 9
                                self.commands[:, 0:1],                                    # [1,1] 10
                                # *** 控方向 ***
                                # self.delta_yaw[:, None] * (self.env_class != 17).float()[:, None],#1
                                # self.delta_next_yaw[:, None] * (self.env_class != 17).float()[:, None],#1
                                # 0*self.commands[:, 0:1], #1
                                # self.commands[:, 2:3] * (self.env_class == 17).float()[:, None],#1
                                # self.commands[:, 0:1],  #[1,1]
                                # *************
                                (self.env_class != 17).float()[:, None],                  # 11
                                (self.env_class == 17).float()[:, None],                  # 12
                                self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),  # 13 14 ... 24
                                self.reindex(self.dof_vel * self.obs_scales.dof_vel),     # 25 ... 36
                                self.reindex(self.action_history_buf[:, -1]),             # 37 ... 48
                                0*self.reindex_feet(self.contact_filt.float()-0.5),       # 49 ... 52
                                ), dim=-1)
        # *** 控方向 ***               
        # obs_buf[self.env_class == 17, 6:7] = (self.commands[self.env_class == 17, 3:4] - self.yaw[self.env_class == 17,None]).clip(min=-0.5,max=0.5)
        # obs_buf[self.env_class == 17, 7:8] = (self.commands[self.env_class == 17, 3:4] - self.yaw[self.env_class == 17,None]).clip(min=-0.5,max=0.5)
        # *************
        priv_explicit = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                                   0 * self.base_lin_vel,  # 3
                                   0 * self.base_lin_vel  # 3
                                   ), dim=-1)  # 9
        priv_latent = torch.cat((self.mass_params_tensor, self.friction_coeffs_tensor, self.motor_strength[0]-1, self.motor_strength[1]-1), dim=-1)
        
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) -1) * self.noise_scale_vec * self.cfg.noise.noise_level
        # print("self.cfg.terrain.measure_heights: ", self.cfg.terrain.measure_heights)
        # print("self.measured_heights: ", self.measured_heights)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            # print("heights.shape: ", heights.shape)  # torch.Size([16, 132])
            if self.cfg.noise.add_height_noise:
                heights += (2 * torch.rand_like(heights) -1) * self.cfg.noise.noise_scales.height_measurements * self.cfg.noise.noise_level
            self.obs_buf = torch.cat([obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        obs_buf[:, 6:8] = 0  # mask yaw in proprioceptive history
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([self.obs_history_buf[:, 1:],obs_buf.unsqueeze(1)], dim=1))

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([self.contact_buf[:, 1:], self.contact_filt.float().unsqueeze(1)], dim=1))
        
    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x += (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x

    def create_sim(self):
        """ Creates simulation, terrain and evironments 创建模拟环境
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)  # sim包含物理与图形信息，用于加载资源、创建环境以及与仿真交互
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*"*80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def randomize_motor_props(self, env_ids):
        # Randomise the motor strength:
        if self.cfg.domain_rand.randomize_motor:
            motor_strength_range = self.cfg.domain_rand.motor_strength_range
            self.torque_multi[env_ids] = torch_rand_float(motor_strength_range[0], motor_strength_range[1], (len(env_ids),self.num_actions), device=self.device)

        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch_rand_float(min_offset, max_offset, (len(env_ids),self.num_actions), device=self.device)
        
        if self.cfg.domain_rand.randomize_gains:
            p_gains_range = self.cfg.domain_rand.stiffness_multiplier_range
            d_gains_range = self.cfg.domain_rand.damping_multiplier_range
            self.randomized_p_gains[env_ids] = torch_rand_float(p_gains_range[0], p_gains_range[1], (len(env_ids),self.num_actions), device=self.device) * self.p_gains
            self.randomized_d_gains[env_ids] = torch_rand_float(d_gains_range[0], d_gains_range[1], (len(env_ids), self.num_actions), device=self.device) * self.d_gains

        if self.cfg.domain_rand.randomize_coulomb_friction:
            joint_coulomb_range = self.cfg.domain_rand.joint_coulomb_range
            joint_viscous_range = self.cfg.domain_rand.joint_viscous_range
            self.randomized_joint_coulomb[env_ids] = torch_rand_float(joint_coulomb_range[0], joint_coulomb_range[1], (len(env_ids),self.num_actions), device=self.device)
            self.randomized_joint_viscous[env_ids] = torch_rand_float(joint_viscous_range[0], joint_viscous_range[1], (len(env_ids),self.num_actions), device=self.device)

        if self.cfg.domain_rand.add_lag:   
            self.lag_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_lag_timestep:
                self.lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.lag_timestep_range[0], self.cfg.domain_rand.lag_timestep_range[2], (len(env_ids),),device=self.device) 
            else:
                self.lag_timestep[env_ids] = self.cfg.domain_rand.lag_timestep_range[1]
    
    def randomize_dof_props(self, env_ids):
        if self.cfg.domain_rand.randomize_joint_friction:
            if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                for i in range(self.num_dofs):
                    range_key = f'joint_{i+1}_friction_range'
                    friction_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_friction_coeffs[env_ids, i] = torch_rand_float(friction_range[0], friction_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:                      
                joint_friction_range = self.cfg.domain_rand.joint_friction_range
                self.joint_friction_coeffs[env_ids] = torch_rand_float(joint_friction_range[0], joint_friction_range[1], (len(env_ids), 1), device=self.device)

        if self.cfg.domain_rand.randomize_joint_damping:
            if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                for i in range(self.num_dofs):
                    range_key = f'joint_{i+1}_damping_range'
                    damping_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_damping_coeffs[env_ids, i] = torch_rand_float(damping_range[0], damping_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:
                joint_damping_range = self.cfg.domain_rand.joint_damping_range
                self.joint_damping_coeffs[env_ids] = torch_rand_float(joint_damping_range[0], joint_damping_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_joint_armature:
            if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                for i in range(self.num_dofs):
                    range_key = f'joint_{i+1}_armature_range'
                    armature_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_armatures[env_ids, i] = torch_rand_float(armature_range[0], armature_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:
                joint_armature_range = self.cfg.domain_rand.joint_armature_range
                self.joint_armatures[env_ids] = torch_rand_float(joint_armature_range[0], joint_armature_range[1], (len(env_ids), 1), device=self.device)

    def _refresh_actor_dof_props(self, env_ids):
        for env_id in env_ids:
            dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], 0)
            for i in range(self.num_dof):
                if self.cfg.domain_rand.randomize_joint_friction:
                    if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                        dof_props["friction"][i] *= self.joint_friction_coeffs[env_id, i]
                    else:    
                        dof_props["friction"][i] *= self.joint_friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_joint_damping:
                    if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                        dof_props["damping"][i] *= self.joint_damping_coeffs[env_id, i]
                    else:
                        dof_props["damping"][i] *= self.joint_damping_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_joint_armature:
                    if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                        dof_props["armature"][i] = self.joint_armatures[env_id, i]
                    else:
                        dof_props["armature"][i] = self.joint_armatures[env_id, 0]
            self.gym.set_actor_dof_properties(self.envs[env_id], 0, dof_props)

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 256  # 64 256
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
                # print("bucket_ids: ", bucket_ids)
                # print("friction_buckets: ", friction_buckets)
                # print("self.friction_coeffs: ", self.friction_coeffs)
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 256  # 64 256
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets,1), device='cpu')
                self.restitution_coeffs = restitution_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            # prepare friction damping armature randomization
            friction_range = self.cfg.domain_rand.joint_friction_range
            self.joint_friction = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs, self.num_dof), device=self.device)
            joint_damping_range = self.cfg.domain_rand.joint_damping_range
            self.joint_damping = torch_rand_float(joint_damping_range[0], joint_damping_range[1], (self.num_envs, self.num_dof), device=self.device)
            armature_range = self.cfg.domain_rand.armature_range
            self.armature_add = torch_rand_float(armature_range[0], armature_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_joint_friction:
            for i in range(self.num_dof):
                if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                    props["friction"][i] = self.joint_friction[env_id, i]
                else:    
                    props["friction"][i] = self.joint_friction[env_id, 0]
        if self.cfg.domain_rand.randomize_joint_damping:
            for i in range(self.num_dof):
                if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                    props["damping"][i] = self.joint_damping[env_id, i]
                else:
                    props["damping"][i] = self.joint_damping[env_id, 0]
        if not self.cfg.domain_rand.randomize_armature:
            self.armature_add = self.armature_add*0
        armature_value = self.cfg.domain_rand.armature_value
        for s in range(self.leg_num):
            props["armature"][0+s*3] = armature_value[0] + self.armature_add[env_id, 0+s*3]
            props["armature"][1+s*3] = armature_value[1] + self.armature_add[env_id, 1+s*3]
            props["armature"][2+s*3] = armature_value[2] + self.armature_add[env_id, 2+s*3]
        return props

    def randomize_rigid_body_props(self, env_ids):
        ''' Randomise some of the rigid body properties of the actor in the given environments, i.e.
            sample the mass, centre of mass position, friction and restitution.'''
        # From Walk These Ways:
        if self.cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = self.cfg.domain_rand.added_mass_range
            self.payload_masses[env_ids] = torch_rand_float(min_payload, max_payload, (len(env_ids), 1), device=self.device)

        if self.cfg.domain_rand.randomize_base_com:
            min_com_displacement, max_com_displacement = self.cfg.domain_rand.added_com_range
            self.com_displacements[env_ids, :] = torch_rand_float(min_com_displacement, max_com_displacement, (len(env_ids), 3), device=self.device)
            
        if self.cfg.domain_rand.randomize_link_mass:
            min_link_mass, max_link_mass = self.cfg.domain_rand.added_link_mass_range
            self.link_masses[env_ids] = torch_rand_float(min_link_mass, max_link_mass, (len(env_ids), self.num_bodies-1), device=self.device)
            
    def _process_rigid_body_props(self, props, env_id):
        """No need to use tensors as only called upon env creation"""
        # randomize base mass
        # if self.cfg.domain_rand.randomize_base_mass:
        #     props[0].mass += self.payload_masses[env_id]
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        
        # randomize base com
        # if self.cfg.domain_rand.randomize_base_com:
        #     props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1], self.com_displacements[env_id, 2])
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)

        # randomize links mass
        if self.cfg.domain_rand.randomize_link_mass:
            for i in range(1, len(props)):
                props[i].mass *= self.link_masses[env_id, i-1]
        # if self.cfg.domain_rand.randomize_link_mass:
        #   for i in range(1, len(props)):
        #     link_mass_range = self.cfg.domain_rand.added_link_mass_range
        #     links_mass = np.random.uniform(link_mass_range[0], link_mass_range[1], size=(1, ))
        #     props[i].mass += links_mass
        # else:
        #     links_mass = np.zeros((1, ))
        # rand_mass += links_mass

        if env_id == self.num_envs-1:
          total_mass = 0
          for i in range(0, len(props)):
              total_mass += props[i].mass
          print("total_mass: ", total_mass)

        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations在计算终止、奖励和观察之前调用的回调
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
            默认行为：根据目标和航向计算角速度命令，计算测量的地形高度，并随机推动机器人
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0)
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)  # 基于四元素的旋转矩阵: Body to World
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            heading_yaw = self.cfg.commands.max_ranges.heading_yaw
            self.commands[:, 2] = torch.clip(0.8*wrap_to_pi(self.commands[:, 3] - heading), heading_yaw[0], heading_yaw[1])
            self.commands[:, 2] *= torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip  # set small commands to zero
        if self.cfg.terrain.measure_heights:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        
    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        flat_terrain_ids = [env_id for env_id in env_ids if self.env_class[env_id]==17]
        if flat_terrain_ids:
            flat_terrain_indices = torch.tensor(flat_terrain_ids, device=self.device)
            lin_vel_x_flat = self.cfg.commands.max_ranges.lin_vel_x_flat
            self.commands[flat_terrain_indices, 0] = torch_rand_float(lin_vel_x_flat[0], lin_vel_x_flat[1], (len(flat_terrain_ids), 1), device=self.device).squeeze(1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip

        # set small commands to zero
        self.commands[env_ids, :2] *= torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip
        rand1 = random.uniform(0,1)
        if rand1<0.1:
            self.commands[env_ids,:2] *= self.env_class[env_ids].unsqueeze(-1)!=17  # 非17地形时，随机将水平速度置0

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # PD controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            # if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
            #     torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel
            # else:
            #     torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel
            # add delay
            if self.cfg.domain_rand.add_lag:
                self.lag_buffer[:,:,1:] = self.lag_buffer[:,:,:self.cfg.domain_rand.lag_timestep_range[1]].clone()
                self.lag_buffer[:,:,0] = actions_scaled.clone()
                actions_scaled = self.lag_buffer[torch.arange(self.num_envs), :, self.lag_timestep.int()]
            if self.cfg.domain_rand.randomize_gains:
                p_gains = self.randomized_p_gains
                d_gains = self.randomized_d_gains
            else:
                p_gains = self.p_gains
                d_gains = self.d_gains
            if self.cfg.domain_rand.randomize_coulomb_friction:
                torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets) -d_gains * self.dof_vel - self.randomized_joint_coulomb  *  self.dof_vel - self.randomized_joint_viscous
            else: 
                torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets) - d_gains * self.dof_vel
            if self.cfg.domain_rand.randomize_motor:
                motor_strength_range = self.cfg.domain_rand.motor_strength_range
                self.torque_multi = torch_rand_float(motor_strength_range[0], motor_strength_range[1], (self.num_envs,self.num_actions), device=self.device)
                torques *= self.torque_multi

        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(self.cfg.env.rand_pos_range[0], self.cfg.env.rand_pos_range[1], (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.env.randomize_start_x:
                self.root_states[env_ids, 0] += self.cfg.env.rand_x_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_ori:
                if self.cfg.env.randomize_start_roll:
                    rand_roll = self.cfg.env.rand_roll_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_roll = torch.zeros(len(env_ids), device=self.device)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                if self.cfg.env.randomize_start_yaw:
                    rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_yaw = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(rand_roll, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if self.cfg.domain_rand.push_vel:
            max_vel = self.cfg.domain_rand.max_push_vel_xy
            self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        if self.cfg.domain_rand.push_ang:
            max_angular = self.cfg.domain_rand.max_push_ang_vel
            self.root_states[:, 10:13] = torch_rand_float(-max_angular, max_angular, (self.num_envs, 3), device=self.device) # ang vel
        if self.cfg.domain_rand.push_vel or self.cfg.domain_rand.push_ang:
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        
        dis_to_origin = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s
        move_up =dis_to_origin > 0.8*threshold
        move_down = dis_to_origin < 0.4*threshold

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    def update_dof_state(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # shallow_copy_dof_state = dof_state_tensor.clone()
        # self.dof_state = gymtorch.wrap_tensor(shallow_copy_dof_state)
        # print("dof_state_tensor: ", dof_state_tensor)
        # print("dof_state_tensor: ", dof_state_tensor.shape)

    def update_robot_state(self):
        # get gym GPU state tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # print("actor_root_state: ", actor_root_state)
        # print("actor_root_state: ", actor_root_state.shape)
        # print("net_contact_force: ", net_contact_force)
        # print("net_contact_force: ", net_contact_force.shape)
        # print("force_sensor_tensor: ", force_sensor_tensor)
        # print("force_sensor_tensor: ", force_sensor_tensor.shape)
        # 包装成PyTorch张量 create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_force).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6) # for feet only, see create_env()
        # # 浅拷贝
        # shallow_copy_root_states = actor_root_state.clone()
        # shallow_copy_rigid_body_states = rigid_body_state_tensor.clone()
        # shallow_copy_contact_forces = net_contact_force.clone()
        # shallow_copy_force_sensor_tensor = force_sensor_tensor.clone()
        # # 包装浅拷贝成PyTorch张量
        # self.root_states = gymtorch.wrap_tensor(shallow_copy_root_states)
        # self.rigid_body_states = gymtorch.wrap_tensor(shallow_copy_rigid_body_states).view(self.num_envs, -1, 13)
        # self.contact_forces = gymtorch.wrap_tensor(shallow_copy_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        # self.force_sensor_tensor = gymtorch.wrap_tensor(shallow_copy_force_sensor_tensor).view(self.num_envs, 4, 6) # for feet only, see create_env()

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.update_robot_state()
        self.update_dof_state()
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # self.height_noise_scale_vec = self._get_height_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # body系下的躯干角速度
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        if self.cfg.domain_rand.obs_delay:
            self.base_ang_vel_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.obs_buf_len, self.base_ang_vel.shape[1], device=self.device, dtype=torch.float)
            self.imu_obs_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.obs_buf_len, self.base_ang_vel.shape[1], device=self.device, dtype=torch.float)
            self.delta_yaw_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.obs_buf_len, self.base_ang_vel.shape[1], device=self.device, dtype=torch.float)
            self.delta_next_yaw_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.obs_buf_len, self.base_ang_vel.shape[1], device=self.device, dtype=torch.float)
            self.commands_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.obs_buf_len, self.base_ang_vel.shape[1], device=self.device, dtype=torch.float)
            self.dof_pos_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.obs_buf_len, self.base_ang_vel.shape[1], device=self.device, dtype=torch.float)
            self.commands_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.obs_buf_len, self.base_ang_vel.shape[1], device=self.device, dtype=torch.float)
            self.dof_vel_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.obs_buf_len, self.base_ang_vel.shape[1], device=self.device, dtype=torch.float)
        # self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs): # 12
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        # print("self.p_gains: ", self.p_gains)
        # print("self.d_gains: ", self.d_gains)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.torque_multi = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.randomized_p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) * self.p_gains  # ones
        self.randomized_d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) * self.d_gains
        self.joint_coulomb = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_viscous = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.randomized_joint_coulomb = torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) * self.joint_coulomb
        self.randomized_joint_viscous = torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) * self.joint_viscous
        if self.cfg.domain_rand.add_lag:
            self.lag_buffer = torch.zeros(self.num_envs, self.num_actions, self.cfg.domain_rand.lag_timestep_range[2], device=self.device)
            if self.cfg.domain_rand.randomize_lag_timestep:
                self.lag_timestep = torch.randint(self.cfg.domain_rand.lag_timestep_range[0], self.cfg.domain_rand.lag_timestep_range[2], (self.num_envs,), device=self.device) 
            else:
                self.lag_timestep = torch.ones(self.num_envs, device=self.device) * self.cfg.domain_rand.lag_timestep_range[1]
        self.randomize_motor_props(torch.arange(self.num_envs, device=self.device))
        
        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs, self.cfg.depth.buffer_len, self.cfg.depth.resized[1], self.cfg.depth.resized[0]).to(self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
                将高度场地形添加到模拟中, 并基于cfg设置参数。
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border
        hf_params.transform.p.y = -self.terrain.border
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order='C'), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
                将三角形网格地形添加到模拟中, 并基于cfg设置参数。水平刻度很小时非常慢
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def attach_camera(self, i, env_handle, actor_handle, camera_config):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            # 创建相机属性对象
            camera_props = gymapi.CameraProperties()
            # camera_props.width = self.cfg.depth.original[0]
            # camera_props.height = self.cfg.depth.original[1]
            # camera_props.enable_tensors = True
            # camera_horizontal_fov = self.cfg.depth.horizontal_fov 
            # camera_props.horizontal_fov = camera_horizontal_fov
            camera_props.width = camera_config['width']
            camera_props.height = camera_config['height']
            camera_props.enable_tensors = camera_config['enable_tensors']
            camera_props.horizontal_fov = camera_config['horizontal_fov']
            # 创建相机传感器
            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            # 设置相机的本地变换
            local_transform = gymapi.Transform()
            # camera_position = np.copy(config.position)
            # camera_angle = np.random.uniform(config.angle[0], config.angle[1])
            # local_transform.p = gymapi.Vec3(*camera_position)
            # local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            camera_angle = np.radians(np.random.uniform(camera_config['angle']+config.angle_error[0], camera_config['angle']+config.angle_error[1]))
            local_transform.p = gymapi.Vec3(*camera_config['position'])
            local_transform.r = gymapi.Quat.from_euler_zyx(0, camera_angle, 0)
            # 获取演员的根刚体句柄，并将相机附加到演员上
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

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
        # print("dof_props_asset: ", dof_props_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        hip_dof_names = [s for s in self.dof_names if self.cfg.asset.hip_dof_name in s]
        thigh_dof_names = [s for s in self.dof_names if self.cfg.asset.thigh_dof_name in s]
        calf_dof_names = [s for s in self.dof_names if self.cfg.asset.calf_dof_name in s]
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        body_indices = {}  # 创建一个空字典
        for s in body_names:
            body_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)  # 获取索引
            body_indices[s] = body_idx  # 将名称和索引作为键值对存入字典
        print("body_names: ", body_names)  # ['base', 'LF_hip', 'LF_thigh', 'LF_calf', 'LF_FOOT', 'LH_hip', 'LH_thigh', 'LH_calf', 'LH_FOOT', 'RF_hip', 'RF_thigh', 'RF_calf', 'RF_FOOT', 'RH_hip', 'RH_thigh', 'RH_calf', 'RH_FOOT']
        print("self.dof_names: ", self.dof_names)  # ['LF_HAA', 'LF_HFE', 'LF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
        print("num_bodies: ", self.num_bodies)  # 17
        print("num_dofs: ", self.num_dofs)  # 12
        print("hip_dof_names: ", hip_dof_names)  # ["LF_HAA", "RF_HAA", "LH_HAA", "RH_HAA"]
        print("thigh_dof_names: ", thigh_dof_names)  # ["LF_HFE", "RF_HFE", "LH_HFE", "RH_HFE"]
        print("calf_dof_names: ",calf_dof_names)  # ["LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE"]
        print("feet_names: ", feet_names)  # ['LF_FOOT', 'LH_FOOT', 'RF_FOOT', 'RH_FOOT']
        print("body_indices: ", body_indices)   # {'base': 0, 'LF_hip': 1, 'LF_thigh': 2, 'LF_calf': 3, 'LF_FOOT': 4, 'LH_hip': 5, 'LH_thigh': 6, 'LH_calf': 7, 'LH_FOOT': 8, 'RF_hip': 9, 'RF_thigh': 10, 'RF_calf': 11, 'RF_FOOT': 12, 'RH_hip': 13, 'RH_thigh': 14, 'RH_calf': 15, 'RH_FOOT': 16}

        # hip_dof_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        # thigh_dof_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]
        # calf_dof_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
        # foot_dof_names=  ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        # hip_dof_names = ["LF_HAA", "RF_HAA", "LH_HAA", "RH_HAA"]
        # thigh_dof_names = ["LF_HFE", "RF_HFE", "LH_HFE", "RH_HFE"]
        # calf_dof_names = ["LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE"]
        # foot_names= ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        for s in feet_names:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            # print("feet_idx: ", feet_idx)  # 4  12  8  16
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
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
        
        # self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # self.randomized_p_gains = torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) * self.p_gains
        # self.randomized_d_gains = torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) * self.d_gains
        

        self.payload_masses = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.link_masses = torch.zeros(self.num_envs, self.num_bodies-1, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_joint_friction_each_joint:
            self.joint_friction_coeffs = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self.joint_friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_joint_damping_each_joint:
            self.joint_damping_coeffs = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self.joint_damping_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_joint_armature_each_joint:
            self.joint_armatures = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)  
        else:
            self.joint_armatures = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.randomize_dof_props(torch.arange(self.num_envs, device=self.device))
        self.randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device))

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(self.cfg.env.rand_pos_range[0], self.cfg.env.rand_pos_range[1], (2,1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.env.rand_yaw_range*np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "panda", i, self.cfg.asset.self_collisions, 0)
            
            dof_props = self._process_dof_props(dof_props_asset, i)  # stores position, velocity and torques limits defined in the URDF
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            # if (i==0):
            #   for dof in range(self.num_dof):
            #     print("armature{dof}: ", dof_props_asset["armature"][dof].item())
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)
            # print("mass_params: ", mass_params)
            # print("self.mass_params_tensor[i, :]: ", self.mass_params_tensor[i, :])

            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
            
            # self.attach_camera(i, env_handle, anymal_handle, self.cfg.depth.camera1_config)
            self.attach_camera(i, env_handle, anymal_handle, self.cfg.depth.camera2_config)

        # self.mass_params_tensor = torch.cat((self.payload_masses, self.com_displacements),dim = -1)
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
            # print("self.friction_coeffs_tensor: ", self.friction_coeffs_tensor)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        self.hip_indices = torch.zeros(len(hip_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_dof_names):
            self.hip_indices[i] = self.dof_names.index(name)

        self.thigh_indices = torch.zeros(len(thigh_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_dof_names):
            self.thigh_indices[i] = self.dof_names.index(name)

        self.calf_indices = torch.zeros(len(calf_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(calf_dof_names):
            self.calf_indices[i] = self.dof_names.index(name)
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            # print("self.num_envs: ", self.num_envs)
            # print("self.terrain_types: ", self.terrain_types)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]
            # print("self.terrain_class: ", self.terrain_class)
            # print("self.env_class: ", self.env_class)

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        print("*"*60)
        print("self.sim_params.dt: ", self.sim_params.dt)
        self.dt = self.cfg.control.decimation * self.sim_params.dt  # 4*0.005
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        reward_norm_factor = 1  # np.sum(list(self.reward_scales.values()))
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        print("self.cfg.commands.curriculum: ", self.cfg.commands.curriculum)
        if self.cfg.commands.curriculum:
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        else:
            self.command_ranges = class_to_dict(self.cfg.commands.max_ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s  # 20 s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_height_samples(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        i = self.lookat_id
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def _draw_goals(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))
        sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))
        sphere_geom_reached = gymutil.WireframeSphereGeometry(self.cfg.env.next_goal_threshold, 32, 32, None, color=(0, 1, 0))
        goals = self.terrain_goals[self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()
        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + self.terrain.cfg.border_size
            pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)
            goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
            pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
            if i == self.cur_goal_idx[self.lookat_id].cpu().item():
                gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                if self.reached_goal_ids[self.lookat_id]:
                    gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            else:
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
            for i in range(5):
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
            for i in range(5):
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.2*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
    def _draw_feet(self):
        if hasattr(self, 'feet_at_edge'):
            non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
            edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            for i in range(4):
                pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)
                if self.feet_at_edge[self.lookat_id, i]:
                    gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[i], pose)
                else:
                    gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[i], pose)
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)
        对每个机器人周围所需点的地形高度进行采样。这些点由基准的位置偏移，并由基准的偏航旋转
        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def get_foot_contacts(self):
        foot_contacts_bool = self.contact_forces[:, self.feet_indices, 2] > 10
        if self.cfg.env.include_foot_contacts:
            return foot_contacts_bool
        else:
            return torch.zeros_like(foot_contacts_bool).to(self.device)

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_heights_points(self, coords, env_ids=None):
        if env_ids:
            points = coords[env_ids]
        else:
            points = coords

        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    ################## parkour rewards ##################

    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        # rew[self.env_class == 17] = torch.exp(-5 * torch.sum(torch.square(self.commands[self.env_class == 17, :2] - self.base_lin_vel[self.env_class == 17, :2]), dim=1))
        # rew[self.env_class == 17] = 1.0
        return rew

    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        # rew[self.env_class == 17] = torch.exp(-5 * torch.abs(self.commands[self.env_class == 17, 3] - self.yaw[self.env_class == 17]))
        return rew
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        rew[self.env_class != 17] *= 0.5
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
     
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        rew[self.env_class != 17] = 0.
        return rew

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew
