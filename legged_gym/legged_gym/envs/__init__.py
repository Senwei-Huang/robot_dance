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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .go2.go2_dance_config import GO2DanceCfg_swing, GO2DanceCfg_swingPPO

from .go2.go2_dance_config import GO2DanceCfg_beat, GO2DanceCfg_beatPPO
from .go2.go2_dance_config import GO2DanceCfg_turn_and_jump, GO2DanceCfg_turn_and_jumpPPO
from .go2.go2_dance_config import GO2DanceCfg_wave, GO2DanceCfg_wavePPO
from .go2.go2_dance_config import GO2DanceCfg_pace, GO2DanceCfg_pacePPO
from .go2.go2_dance_config import GO2DanceCfg_trot, GO2DanceCfg_trotPPO
from .base.legged_robot_transition import LeggedRobotTrans
from .go2.go2_dance_trans_config import GO2DanceCfg_trans, GO2DanceCfg_trans_PPO
from .base.legged_robot_panda import LeggedRobotPanda
from .base.panda_fixed_gripper import LeggedRobotPandaFixedGripper

from .panda7.panda7_config import panda7BeatCfg, panda7BeatCfgPPO
from .panda7.panda7_config import panda7TrotCfg, panda7TrotCfgPPO

from .panda7_fixed_arm.panda7_fixed_arm_config import panda7_fixed_arm_BeatCfg, panda7_fixed_arm_BeatCfgPPO
from .panda7_fixed_arm.panda7_fixed_arm_config import panda7_fixed_arm_TrotCfg, panda7_fixed_arm_TrotCfgPPO
from .panda7_fixed_arm.panda7_fixed_arm_config import panda7_fixed_arm_PaceCfg, panda7_fixed_arm_PaceCfgPPO
from .panda7_fixed_arm.panda7_fixed_arm_config import panda7_fixed_arm_SwingCfg, panda7_fixed_arm_SwingCfgPPO
from .panda7_fixed_arm.panda7_fixed_arm_config import panda7_fixed_arm_Turn_and_jumpCfg, panda7_fixed_arm_Turn_and_jumpCfgPPO
from .panda7_fixed_arm.panda7_fixed_arm_config import panda7_fixed_arm_WaveCfg, panda7_fixed_arm_WaveCfgPPO

from .panda7_fixed_gripper.panda7_fixed_gripper import panda7_fixed_gripper_BeatCfg, panda7_fixed_gripper_BeatCfgPPO
import os
from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
# GO2
task_registry.register( "go2_dance_swing", LeggedRobot, GO2DanceCfg_swing(), GO2DanceCfg_swingPPO() )
task_registry.register( "go2_dance_beat", LeggedRobot, GO2DanceCfg_beat(), GO2DanceCfg_beatPPO() )
task_registry.register( "go2_dance_turn_and_jump", LeggedRobot, GO2DanceCfg_turn_and_jump(),
                        GO2DanceCfg_turn_and_jumpPPO() )
task_registry.register( "go2_dance_wave", LeggedRobot, GO2DanceCfg_wave(), GO2DanceCfg_wavePPO() )
task_registry.register( "go2_dance_pace", LeggedRobot, GO2DanceCfg_pace(), GO2DanceCfg_pacePPO() )
task_registry.register( "go2_dance_trot", LeggedRobot, GO2DanceCfg_trot(), GO2DanceCfg_trotPPO() )
task_registry.register("go2_dance_trans", LeggedRobotTrans, GO2DanceCfg_trans(), GO2DanceCfg_trans_PPO())
# panda7
task_registry.register( "panda7_beat", LeggedRobotPanda, panda7BeatCfg(), panda7BeatCfgPPO() )
task_registry.register( "panda7_trot", LeggedRobotPanda, panda7TrotCfg(), panda7TrotCfgPPO() )
# panda7 fixed arm
task_registry.register( "panda7_fixed_arm_beat", LeggedRobot, panda7_fixed_arm_BeatCfg(), panda7_fixed_arm_BeatCfgPPO() )
task_registry.register( "panda7_fixed_arm_trot", LeggedRobot, panda7_fixed_arm_TrotCfg(), panda7_fixed_arm_TrotCfgPPO() )
task_registry.register( "panda7_fixed_arm_pace", LeggedRobot, panda7_fixed_arm_PaceCfg(), panda7_fixed_arm_PaceCfgPPO() )
task_registry.register( "panda7_fixed_arm_swing", LeggedRobot, panda7_fixed_arm_SwingCfg(), panda7_fixed_arm_SwingCfgPPO() )
task_registry.register( "panda7_fixed_arm_turn_and_jump", LeggedRobot, panda7_fixed_arm_Turn_and_jumpCfg(),
                        panda7_fixed_arm_Turn_and_jumpCfgPPO() )
task_registry.register( "panda7_fixed_arm_wave", LeggedRobot, panda7_fixed_arm_WaveCfg(), panda7_fixed_arm_WaveCfgPPO() )
# panda7 fixed gripper
task_registry.register( "panda7_fixed_gripper_beat", LeggedRobotPandaFixedGripper, panda7_fixed_gripper_BeatCfg(), panda7_fixed_gripper_BeatCfgPPO())


