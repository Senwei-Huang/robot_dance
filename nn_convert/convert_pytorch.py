"""
************************************************************
** Detail: Convert the PyTorch neural network model to ONNX.
           Compile, run, and test the ONNX model using TVM.
** Author: Senwei Huang, Chong Tian
** Update: 2024-06-09
** Version: 1.0
************************************************************
"""
import torch
import onnx
import onnxruntime
import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.contrib import graph_executor
import os, os.path
import numpy as np
from actorNN_58 import ModelLoad

print("PyTorch Version: ", torch.__version__)  # PyTorch Version:  1.10.0+cu113  2.1.1+cu118
print("ONNX Version: ", onnx.__version__)  # ONNX Version:  1.17.0  ONNX Version:  1.16.1
print("onnxruntime Version: ", onnxruntime.__version__)  # onnxruntime Version:  1.19.2  onnxruntime Version:  1.18.0


def pytorch_to_onnx(torch_model: torch.nn.Module, inputs, onnx_save_path):
    """
    torch_model: PyTorch模型实例, 定义了模型的结构并加载预训练权重文件， PyTorch版本2.0以下
    """
    print("########################## Exporting model to ONNX format ##########################")
    # 导出模型到 ONNX 格式
    torch.onnx.export(torch_model,
                      inputs,
                      # {"actor_desc": actor_desc, "sense_desc": sense_desc},
                      onnx_save_path,
                      export_params=True,  # 保存训练参数
                      verbose=False,
                      opset_version=12,  # 导出模型使用的 ONNX opset 版本
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      # input_names=['inputs'],  # 模型输入名
                      input_names=['actor_desc', 'sense_desc'],
                      output_names=['output'],  # 模型输出名
                      # example_outputs=outputs,
                      )
    print("Saved ONNX model at: ", os.path.abspath(onnx_save_path))

def tvmc_compile(model_path, tvm_output_path):
  # Step 1: Load
  model = tvmc.load(model_path) 

  # Step 1.5: Optional Tune
  # print("################# Logging File #################") 
  # log_file = "merged_net_tune_record.json"
  # tvmc.tune(model, target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu", enable_autoscheduler = True, tuning_records=log_file) 

  # Step 2: Compile
  print("########################## Network Converted ##########################") 
  # package = tvmc.compile(model, target="llvm", tuning_records=log_file, package_path=tvm_output_path)
  # package = tvmc.compile(model, target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu", cross='aarch64-linux-gnu-gcc',package_path=tvm_output_path)
  # package_tuned = tvmc.compile(model, target="llvm -device=arm_cpu -mtriple=aarch64-none-linux-gnu",cross='aarch64-none-linux-gnu-gcc', tuning_records=log_file, package_path=tvm_output_path)
  # package = tvmc.compile(model, target="cuda", package_path=tvm_output_path)
  package_tuned = tvmc.compile(model, target="cuda -arch=sm_87", target_host='llvm -mtriple=aarch64-linux-gnu',cross='aarch64-linux-gnu-gcc',package_path=tvm_output_path)  # sm_80 sm_87

  # package_tuned = tvmc.TVMCPackage(package_path=tvm_output_path)
  # print(tvm.target.Target.list_kinds())


  # Step 3: Run
  # result = tvmc.run(package, device="cuda")
  # print(result)
  # result_tuned = tvmc.run(package_tuned, device="cpu") 
  # print(result_tuned)

  # o_ex = np.zeros((1,1,208)).astype(np.float32)
  # o_pt = np.zeros((1, 1, 154)).astype(np.float32)
  # h_t = np.zeros((2, 1, 50)).astype(np.float32)
  # shape_dict = {'robot_state':o_pt.shape,'vision_input':o_ex.shape,'hidden_state':h_t.shape}

def relay_compile(inputs, actor_desc, sense_desc, model_path):
  # 步骤1: 加载ONNX模型
  onnx_model = onnx.load(model_path)

  # 步骤2: 设置目标和目标主机
  # target = 'llvm'
  # target = "llvm -mtriple=aarch64-linux-gnu"
  target = "cuda"
  # target = 'cuda -arch=sm_87'
  # target_host = 'llvm -mtriple=aarch64-linux-gnu'
  # dev = tvm.cpu(0)
  dev = tvm.cuda(0)

  # 步骤3: 通过relay将ONNX模型转换为TVM中间表示
  input_name = ['actor_desc', 'sense_desc']
  # input_name = ['inputs']
  shape_dict = {input_name[0]: actor_desc.shape, input_name[1]: sense_desc.shape}
  # shape_dict = {input_name[0]: inputs.shape}
  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
  
  # 步骤4: 构建配置 优化模型 模型编译
  with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod, target=target, params=params)
    # graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

  # 步骤5: 导出编译结果
  # lib.export_library("relay_compiled_model.tar")
  # lib.export_library("compiled_model_agx_orin.so")
  # lib.export_library('oghr_controller.tar', cross_compile='aarch64-linux-gnu-gcc')
  # lib.export_library('oghr_controller.tar', fcompile=cross_compiler_toolchain)

  # 输出张量的形状以供调试
  print(f"actor_desc shape: {actor_desc.shape}, dtype: {actor_desc.dtype}")
  print(f"sense_desc shape: {sense_desc.shape}, dtype: {sense_desc.dtype}")
  # 检查张量大小
  actor_desc_size = actor_desc.numel() * actor_desc.element_size()
  sense_desc_size = sense_desc.numel() * sense_desc.element_size()
  expected_actor_size = 512 * 4  # 512 * sizeof(float32)
  expected_sense_size = 348 * 4    # 348 * sizeof(float32)
  if actor_desc_size != expected_actor_size:
      raise ValueError(f"actor_desc size mismatch: expected {expected_actor_size}, got {actor_desc_size}")
  if sense_desc_size != expected_sense_size:
      raise ValueError(f"sense_desc size mismatch: expected {expected_sense_size}, got {sense_desc_size}")

  executor = graph_executor.create(graph, lib, dev)
  executor.set_input('actor_desc', tvm.nd.array(actor_desc.cpu().detach().numpy(), dev))
  executor.set_input('sense_desc', tvm.nd.array(sense_desc.cpu().detach().numpy(), dev))
  executor.set_input(**params)
  executor.run()
  tvm_output = executor.get_output(0)
  tvm_output = tvm_output.asnumpy().flatten()
  return tvm_output

def get_torch_output(inputs, torch_model):
    torch_output = torch_model(inputs)
    print("torch_output :", torch_output)  # GPU if use cuda
    torch_output = torch_output.detach().cpu().numpy().flatten()
    return torch_output

def get_onnx_output(actor_desc, sense_desc, onnx_save_path):
    onnx_model = onnx.load(onnx_save_path)
    onnx.checker.check_model(onnx_model)
    
    input_names = [input.name for input in onnx_model.graph.input]
    output_names = [output.name for output in onnx_model.graph.output]

    input_data = {'actor_desc': actor_desc.detach().cpu().numpy(), 'sense_desc': sense_desc.detach().cpu().numpy()}
    session = onnxruntime.InferenceSession(onnx_save_path)  # 加载 ONNX 模型
    input_info = session.get_inputs()[0]  # 获取输入信息
    input_name = input_info.name
    input_shape = input_info.shape
    input_type = input_info.type
    
    onnx_output = session.run(None, input_data)  # 运行ONNX模型
    onnx_output_info = session.get_outputs()[0]  # 获取输出信息
    onnx_output_name = onnx_output_info.name
    onnx_output_shape = onnx_output_info.shape
    
    onnx_output_data = onnx_output[0]
    onnx_output_data = onnx_output_data.flatten()
    
    # print("ONNX模型的输入名称: ", input_names)
    # print("ONNX模型的输出名称: ", output_names)
    # print("onnx_output_info :", onnx_output_info )
    # print("onnx_output_name :", onnx_output_name )
    # print("onnx_output_shape :", onnx_output_shape )
    # print("onnx_output_data.shape: ", onnx_output_data.shape)
    # print("onnx_output:", onnx_output)
    print("onnx_output_data :", onnx_output_data )
    # print(onnx.helper.printable_graph(onnx_model.graph))
    
    return onnx_output_data

def test_result(torch_out, onnx_out, decimal=4):
    """
    加载、检查和测试导出的ONNX模型
    """
    print("########################## Testing Convert Result ##########################")
    try: 
      np.testing.assert_almost_equal(torch_out, onnx_out, decimal=decimal) 
      print("Result: Model outputs are closely matched ! Decimal =", decimal)
      return 1
    except AssertionError as e: 
      print("Mismatch in outputs:", e)
      return 0


if __name__ == '__main__':
    device = torch.device('cpu')  # cpu cuda 在cuda设备上测试时PyTorch模型推理结果的精度受设备和库版本的影响较大
    torch_model_path = "./model/gap_Nov04_10-13-03_.pt"  # 模型路径
    remodel_0 = ModelLoad(device=device)#实例
    remodel_0.cfg.model.path = torch_model_path
    remodel_0.load_model()
    torch_model = remodel_0.net
    # torch_model = torch.load("model.pt")  # 或 torch.jit.load("model.jit")
    torch_model.eval()  # 设置模型为评估模式
    
    # 模型转换
    with torch.no_grad():  # 屏蔽梯度计算
      torch.manual_seed(0)  # 设置随机种子，确保每次运行产生相同随机输入数据
      # 创建输入张量，需要根据实际模型输入调整尺寸
      actor_desc = torch.zeros(512, device=device) 
      sense_desc = torch.zeros(348, device=device)
      outputs = torch.zeros(size=(12,), device=device)
      inputs = {}
      inputs['actor_desc'] = actor_desc
      inputs['sense_desc'] = sense_desc
      
      # 转ONNX
      onnx_save_path = "./model/gap_Nov04_10-13-03_.onnx"  # ONNX模型保存路径
      pytorch_to_onnx(torch_model, inputs, onnx_save_path)
      
      # 转TVM
      tvm_output_path = ("./model/gap_Nov04_10-13-03_.tar")  # TVM模型保存路径和名称格式
      tvmc_compile(onnx_save_path, tvm_output_path)
      
      # 转换结果测试
      N_torch_onnx = 0
      N_torch_tvm = 0
      N_onnx_tvm = 0
      for i in range(10):
        torch.manual_seed(i)
        actor_desc = torch.randn(512, device=device) 
        sense_desc = torch.randn(348, device=device)
        # actor_desc = torch.ones(512, device=device) 
        # sense_desc = torch.ones(348, device=device)
        inputs = {}
        inputs['actor_desc'] = actor_desc
        inputs['sense_desc'] = sense_desc 
        
        # torch_output = remodel_0.inference(actor_desc, sense_desc)
        torch_output = get_torch_output(inputs, torch_model)
        onnx_output = get_onnx_output(actor_desc, sense_desc, onnx_save_path)
        tvm_output = relay_compile(inputs, actor_desc, sense_desc, onnx_save_path)
        
        print("torch_output: ", torch_output, sep="\n")
        print("onnx_output: ", onnx_output, sep="\n")
        print("tvm_output: ", tvm_output, sep="\n")
        N_torch_onnx += test_result(torch_output, onnx_output, decimal=5)  # 测试导出的ONNX模型
        N_torch_tvm += test_result(torch_output, tvm_output, decimal=5)  # 测试导出的TVM模型
        N_onnx_tvm += test_result(onnx_output, tvm_output, decimal=5)  # 测试导出的TVM模型
      print("The number of successful matches between PyTorch and ONNX: ", N_torch_onnx)
      print("The number of successful matches between PyTorch and TVM: ", N_torch_tvm)
      print("The number of successful matches between ONNX and TVM: ", N_onnx_tvm)
    