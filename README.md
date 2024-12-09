# 训练机器人跳舞

### 参考：
- https://github.com/leggedrobotics/legged_gym
- https://github.com/inspirai/MetalHead
- https://github.com/Alescontrela/AMP_for_hardware

### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended). i.e. with conda:
    - `conda create -n dance python==3.8`
    - `conda activate dance`
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 tensorboard==2.8.0 pybullet==3.2.1 opencv-python==4.5.5.64 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   - Clone this repository
   -  `cd robot_dance/rsl_rl && pip install -e .` 
5. Install legged_gym
   - `cd ../legged_gym && pip install -e .`

### 训练补充安装：

```bash
conda install numpy=1.23
pip install setuptools==59.5.0
pip install protobuf==3.20.*
```
```bash
pip install attrs cloudpickle decorator ml_dtypes packaging psutil scipy tornado
pip install scipy
```
Sim2Sim安装
```bash
pip install sysv_ipc
pip install keyboard
 ```
Sim2Sim运行
```bash
sudo /home/ubuntn/anaconda3/envs/dance/bin/python bjtu_dance_twodogs.py
```
