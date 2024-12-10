# train
python legged_gym/legged_gym/scripts/train.py --task=panda7_arm_swing
python legged_gym/legged_gym/scripts/train.py --task=panda7_arm_swing --num_envs=1

python legged_gym/legged_gym/scripts/train.py --task=panda7_arm_beat --sim_device cuda:1 --rl_device cuda:1 --headless


# play
python legged_gym/legged_gym/scripts/play_panda.py --task=panda7_arm_swing

python legged_gym/legged_gym/scripts/play_panda.py --task=panda7_arm_beat --num_envs=1

# log
~/robot_dance/legged_gym/logs/panda7_fixed_gripper_beat$ tensorboard --logdir Dec04_22-34-54_/
tensorboard --logdir Dec04_22-34-54_/
