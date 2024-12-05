import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

from rsl_rl.modules import ActorCritic



def load_policy() -> dict:
    '''
    这个函数是按照task_list中的顺序提取对应的动作policy函数
    '''
    dance_task_policy = {}
    # path_list = ['/home/pcpc/robot_dance/legged_gym/log/GO2/keep_the_beat/model_1500.pt',
    #             '/home/pcpc/robot_dance/legged_gym/log/GO2/pace/model_1500.pt',
    #             '/home/pcpc/robot_dance/legged_gym/log/GO2/swing/model_1500.pt',
    #             '/home/pcpc/robot_dance/legged_gym/log/GO2/trot/model_1500.pt',
    #             '/home/pcpc/robot_dance/legged_gym/log/GO2/turn_and_jump/model_1500.pt',
    #             '/home/pcpc/robot_dance/legged_gym/log/GO2/wave/model_1500.pt']
    path_list = ["legged_gym/log/GO2_new/keep_the_beat/model_1500.pt",
                 "legged_gym/log/GO2_new/pace/model_1500.pt",
                 "legged_gym/log/GO2_new/swing/model_1500.pt",
                 "legged_gym/log/GO2_new/trot/model_1500.pt",
                 "legged_gym/log/GO2_new/turn_and_jump/model_1500.pt",
                 "legged_gym/log/GO2_new/wave/model_1500.pt"]
    task_list = ['go2_dance_beat', 'go2_dance_pace', 'go2_dance_swing', 'go2_dance_trot',
                 'go2_dance_turn_and_jump', 'go2_dance_wave']

    for i, load_path in enumerate(path_list):
        env_cfg = task_registry.env_cfgs[task_list[i]]
        train_cfg = task_registry.train_cfgs[task_list[i]]
        policy:torch.nn.Module = ActorCritic(env_cfg.env.num_observations, env_cfg.env.num_observations,
                                             env_cfg.env.num_actions, train_cfg.policy.actor_hidden_dims,
                                             train_cfg.policy.critic_hidden_dims, train_cfg.policy.activation,
                                             train_cfg.policy.init_noise_std).to('cuda:0')
        loaded_dict = torch.load(load_path)
        policy.load_state_dict(loaded_dict['model_state_dict'])
        policy.eval()
        dance_task_policy[task_list[i]] = policy.act_inference

    return dance_task_policy



def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    dance_task_policy = load_policy()
    ppo_runner.learn_trans( dance_task_policy, num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)

if __name__ == '__main__':
    args = get_args()
    train(args)
