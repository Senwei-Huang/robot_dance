import os

print("go2_dance_beat")
os.system("python legged_gym/legged_gym/scripts/train.py --task=go2_dance_beat --num_envs=53248 --headless")
print("go2_dance_turn_and_jump")
os.system("python legged_gym/legged_gym/scripts/train.py --task=go2_dance_turn_and_jump --num_envs=53248 --headless")
print("go2_dance_wave")
os.system("python legged_gym/legged_gym/scripts/train.py --task=go2_dance_wave --num_envs=53248 --headless")
print("go2_dance_pace")
os.system("python legged_gym/legged_gym/scripts/train.py --task=go2_dance_pace --num_envs=53248 --headless")
print("go2_dance_trot")
os.system("python legged_gym/legged_gym/scripts/train.py --task=go2_dance_trot --num_envs=53248 --headless")
# "python legged_gym/legged_gym/scripts/train_trans.py --task=go2_dance_trans --num_envs=53248 --headless"


