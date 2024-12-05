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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Panda7ParkourCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.55] # x,y,z [m]  [0.0, 0.0, 0.6]  x为机器人的前进正方向
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 'LF_HAA': 0.0,   # [rad]
            # 'RF_HAA': 0.0,   # [rad]
            # 'LH_HAA': -0.0,  # [rad]
            # 'RH_HAA': -0.0,   # [rad]

            # 'LF_HFE': 0.785,     # [rad]
            # 'RF_HFE': 0.785,   # [rad]
            # 'LH_HFE': 0.785,     # [rad]
            # 'RH_HFE': 0.785,   # [rad]

            # 'LF_KFE': -1.5708,   # [rad]
            # 'RF_KFE': -1.5708,    # [rad]
            # 'LH_KFE': -1.5708,  # [rad]
            # 'RH_KFE': -1.5708,    # [rad]
            'LH_HAA': -0.1,  # [rad]
            'LF_HAA': -0.1,   # [rad]
            'RH_HAA': 0.1,   # [rad]
            'RF_HAA': 0.1,   # [rad]
            
            'LH_HFE': 1.0,     # [rad]
            'LF_HFE': 0.8,     # [rad]
            'RH_HFE': 1.0,   # [rad]
            'RF_HFE': 0.8,   # [rad]
            
            'LH_KFE': -1.5,  # [rad]
            'LF_KFE': -1.5,   # [rad]
            'RH_KFE': -1.5,    # [rad]
            'RF_KFE': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'HAA': 150., 'HFE':150., 'KFE': 150.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE':2., 'KFE': 2.}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7/urdf/panda7.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/XT70/urdf/XT70.urdf'
        foot_name = "FOOT"
        hip_dof_name = "HAA"
        thigh_dof_name = "HFE"
        calf_dof_name = "KFE"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]#, "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.55
        

class Panda7ParkourCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_panda7'


