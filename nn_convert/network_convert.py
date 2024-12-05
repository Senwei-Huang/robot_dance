"""
************************************************************
** Detail: Convert the PyTorch neural network model to ONNX.
           Compile, run, and test the ONNX model using TVM.
** Author: Senwei Huang
** Update: 2024-06-09
** Version: 1.0
************************************************************
"""
import torch
import os, os.path
import argparse
from convert_utils import *
from neural_network import DanceNN
import warnings 
warnings.filterwarnings('ignore')


def play(args):
    load_run = "../legged_gym/logs/panda7_fixed_arm_swing/Dec03_18-08-47_"
    checkpoint = args.checkpoint
    num_obs = 94
    num_actions = 12
    actor_hidden_dims = [512, 256, 128]
    num_envs = 1
    device = torch.device('cpu')  # cuda:0
    
    # 创建Pytorch模型
    # policy = HardwareVisionNN(n_proprio, num_scan, n_priv_latent, n_priv_explicit, history_len, num_actions, args.tanh).to(device)
    actor_critic = DanceNN(num_actor_obs=num_obs, num_critic_obs=num_obs, num_actions=num_actions,
                     actor_hidden_dims=actor_hidden_dims, critic_hidden_dims=actor_hidden_dims)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)  # 预训练模型参数的路径
    load_run = os.path.dirname(load_path)
    print(f"Loading PyTorch model parameters from: {load_path}")
    # print(f"load_run: {load_run}")  # load_run: ../../logs/parkour_new/202-29-WHATEVER
    
    # 加载预训练模型参数
    ac_state_dict = torch.load(load_path, map_location=device)
    print(f"ac_state_dict: {ac_state_dict.keys()}")
    actor_critic.load_state_dict(ac_state_dict['model_state_dict']) # 从字典中取出键对应的模型参数, strict=True: 严格模式，确保所有预训练权重的键都与模型架构兼容
    torch_model = actor_critic.to(device)
    torch_model.eval()  # 将模型设置为评估模式，以便稍后进行推理
    policy = actor_critic.act_inference

    with torch.no_grad():  # 屏蔽梯度计算
        torch.manual_seed(0)  # 设置随机种子，确保每次运行产生相同随机输入数据
        # 创建测试输入张量，需要根据实际模型输入调整尺寸
        obs_input = torch.zeros(1, num_obs, device=device)
        
        # Save the traced actor
        if not os.path.exists(os.path.join(load_run, "traced")):
            os.mkdir(os.path.join(load_run, "traced"))
        state_dict = {'model_state_dict': ac_state_dict['model_state_dict']}
        torch.save(state_dict, os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-vision_weight.pt"))  # 保存深度编码网络模型的参数
        jit_save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-base_jit.pt")
        pytorch_to_jit(torch_model, obs_input, jit_save_path)
        
        # Save the onnx model
        onnx_save_path = os.path.join(load_run, "onnx_models", args.exptid + "-" + str(checkpoint) + "-panda_dance.onnx")  # 设置ONNX模型的保存路径和名称格式
        if not os.path.exists(os.path.dirname(onnx_save_path)):
            os.makedirs(os.path.dirname(onnx_save_path))  # 检查onnx_models文件夹是否存在，如果不存在则创建该文件夹
        onnx_out = pytorch_to_onnx(torch_model, obs_input, onnx_save_path)

        # 转TVM
        onnx_load_run = os.path.join(load_run, "onnx_models")
        # print("onnx_load_run: ", onnx_load_run)
        onnx_load_path, _ = get_load_path(root=onnx_load_run, checkpoint=-1, model_name_include="onnx")  # ONNX模型路径
        tvm_output_path = os.path.join(load_run, "tvm_output", args.exptid + "-" + str(checkpoint) + "-panda_dance.tar")  # TVM模型保存路径和名称格式
        if not os.path.exists(os.path.dirname(tvm_output_path)):
            os.makedirs(os.path.dirname(tvm_output_path))  # 检查tvm_output文件夹是否存在，如果不存在则创建该文件夹
        tvmc_compile(onnx_load_path, tvm_output_path)
    

    for i in range(1):
        torch.manual_seed(i)
        # obs_input = torch.ones(1, 753, device=device)
        # depth_input = torch.ones(1, 58, 87, device=device)
        # hidden_input = torch.ones(1, 1, 512, device=device)
        obs_input = torch.randn(1, 753, device=device)
        depth_input = torch.randn(1, 58, 87, device=device)
        hidden_input = torch.randn(1, 1, 512, device=device)
        
        torch_output = get_torch_output(obs_input, depth_input, hidden_input, torch_model)
        jit_output = get_jit_output(obs_input, depth_input, hidden_input, jit_save_path)
        onnx_output = get_onnx_output(obs_input, depth_input, hidden_input, onnx_save_path)
        tvm_output = relay_compile(obs_input, depth_input, hidden_input, onnx_load_path)
        
        print("torch_output: ", torch_output, sep="\n")
        print("jit_output: ", jit_output, sep="\n")
        print("onnx_output: ", onnx_output, sep="\n")
        print("tvm_output: ", tvm_output, sep="\n")
        
        test_result(torch_output, jit_output)  # 测试导出的JIT模型
        test_result(torch_output, onnx_output)  # 测试导出的ONNX模型
        test_result(torch_output, tvm_output)  # 测试导出的TVM模型

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--tanh', action='store_true')
    args = parser.parse_args()
    play(args)
    