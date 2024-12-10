"""
************************************************************
** Detail: Convert the PyTorch neural network model to ONNX.
           Compile, run, and test the ONNX model using TVM.
** Author: Senwei Huang and Chong Pi
** Update: 2024-12-05
** Version: 1.0
************************************************************
"""
import torch
from actor_critic import ActorCritic
from convert_utils import *


def nn_convert():
    device = torch.device('cpu')  # cpu cuda 在cuda设备上测试时PyTorch模型推理结果的精度受设备和库版本的影响较大
    num_obs = 63  # 94 63
    num_actions = 18  # 12 18
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    policy: ActorCritic = ActorCritic(num_obs, num_obs, num_actions, actor_hidden_dims, critic_hidden_dims)
    policy.to(device)
    policy.eval()  # 模型置为评估模式
    
    # load_path = "../model/swing/model_1500"
    # load_path = "../model/wave/model_6000"
    
    load_path = "../model/swing/model_15000"
    # load_path = "../model/wave/model_6900"
    # load_path = "../model/turnjump/model_7450"
    
    # 加载模型参数
    pt_path =  load_path + ".pt"  # 模型路径 "../legged_gym/logs/panda7_fixed_arm_swing/Dec03_18-08-47_"
    loaded_dict = torch.load(pt_path)
    print("loaded_dict: ", loaded_dict.keys())
    policy.actor.load_state_dict(loaded_dict['actor_state_dict']) 
    # policy.load_state_dict(loaded_dict['model_state_dict'])
    actor_model = policy.actor
    
    # 模型转换
    with torch.no_grad():  # 屏蔽梯度计算
      torch.manual_seed(0)  # 设置随机种子，确保每次运行产生相同随机输入数据
      # 创建输入张量，需要根据实际模型输入调整尺寸
      inputs = torch.zeros(num_obs, device=device)
      outputs = torch.zeros(size=(num_actions,), device=device)
      
      # 转JIT
      jit_save_path = load_path + ".jit"  # JIT模型保存路径
      pytorch_to_jit(actor_model, inputs, jit_save_path)
      
      # 转ONNX
      onnx_save_path = load_path + ".onnx"  # ONNX模型保存路径
      pytorch_to_onnx(actor_model, inputs, onnx_save_path)
      
      # 转TVM
      tvm_save_path = (load_path + ".tar")  # TVM模型保存路径和名称格式
      tvmc_compile(onnx_save_path, tvm_save_path)
      
      # 转换结果测试
      N_torch_jit = 0
      N_torch_onnx = 0
      N_torch_tvm = 0
      N_onnx_tvm = 0
      for i in range(1):
        torch.manual_seed(i)
        inputs = torch.randn(num_obs, device=device)
        # inputs = torch.zeros(num_obs, device=device)
        inputs = torch.ones(num_obs, device=device)
        
        torch_output = get_torch_output(inputs, actor_model)
        jit_output = get_jit_output(inputs, jit_save_path)
        onnx_output = get_onnx_output(inputs, onnx_save_path)
        tvm_output = relay_compile(inputs, onnx_save_path)
        
        print("torch_output: ", torch_output, sep="\n")
        print("jit_output: ", jit_output, sep="\n")
        print("onnx_output: ", onnx_output, sep="\n")
        print("tvm_output: ", tvm_output, sep="\n")
        
        N_torch_jit += test_result(torch_output, jit_output, decimal=4)  # 测试导出的JIT模型
        N_torch_onnx += test_result(torch_output, onnx_output, decimal=4)  # 测试导出的ONNX模型
        N_torch_tvm += test_result(torch_output, tvm_output, decimal=4)  # 测试导出的TVM模型
        N_onnx_tvm += test_result(onnx_output, tvm_output, decimal=4)  # 测试导出的TVM模型
        
      print("The number of successful matches between PyTorch and JIT: ", N_torch_jit)
      print("The number of successful matches between PyTorch and ONNX: ", N_torch_onnx)
      print("The number of successful matches between PyTorch and TVM: ", N_torch_tvm)
      print("The number of successful matches between ONNX and TVM: ", N_onnx_tvm)
      

if __name__ == '__main__':
    nn_convert()
