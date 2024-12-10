"""
************************************************************
** Detail: Neural network conversion and compilation tool.
** Author: Senwei Huang, Chong Tian
** Update: 2024-06-09
** Version: 1.0
************************************************************
"""
import os, os.path
import numpy as np
import torch
import onnx
import onnxruntime
import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.contrib import graph_executor


print("PyTorch Version: ", torch.__version__)  # PyTorch Version:  1.10.0+cu113  2.1.1+cu118
print("ONNX Version: ", onnx.__version__)  # ONNX Version:  1.17.0  ONNX Version:  1.16.1
print("onnxruntime Version: ", onnxruntime.__version__)  # onnxruntime Version:  1.19.2  onnxruntime Version:  1.18.0


def pytorch_to_jit(torch_model, inputs, jit_save_path):
    """
    torch_model: PyTorch模型实例, 定义了模型的结构并加载预训练权重文件
    """
    print("******************* Exporting model to JIT format *******************")
    traced_policy = torch.jit.trace(torch_model, inputs)
    # script_policy = torch.jit.script(torch_model)
    # print("traced_policy: ", traced_policy)
    # print("script_policy: ", script_policy)

    # 调用模型
    traced_output = traced_policy(inputs)
    # script_output = script_policy(inputs)
    # print("traced_output: ", traced_output)
    # print("script_output: ", script_output)

    # 保存模型
    traced_policy.save(jit_save_path)
    # script_policy.save(jit_save_path)
    print("Saved JIT model at: ", os.path.abspath(jit_save_path))


def get_jit_output(inputs, jit_save_path):
    # 加载模型
    model = torch.jit.load(jit_save_path)
    # print("model: ", model)
    model.eval()
    jit_output = model(inputs)
    jit_output = jit_output.detach().numpy().flatten()
    return jit_output
  
  
def pytorch_to_onnx(torch_model: torch.nn.Module, inputs, onnx_save_path):
    """
    torch_model: PyTorch模型实例, 定义了模型的结构并加载预训练权重文件， PyTorch版本2.0以下
    """
    print("########################## Exporting model to ONNX format ##########################")
    # 导出模型到 ONNX 格式
    torch.onnx.export(torch_model,
                      inputs,
                      onnx_save_path,
                      export_params=True,  # 保存训练参数
                      verbose=False,
                      opset_version=12,  # 导出模型使用的 ONNX opset 版本
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=['inputs'],  # 模型输入名
                      output_names=['outputs'],  # 模型输出名
                      # example_outputs=outputs,
                      )
    print("Saved ONNX model at: ", os.path.abspath(onnx_save_path))


def get_onnx_output(inputs, onnx_save_path):
    onnx_model = onnx.load(onnx_save_path)
    onnx.checker.check_model(onnx_model)
    
    input_names = [input.name for input in onnx_model.graph.input]
    output_names = [output.name for output in onnx_model.graph.output]

    input_data = {'inputs': inputs.detach().cpu().numpy()}
    session = onnxruntime.InferenceSession(onnx_save_path)  # 加载 ONNX 模型
    input_info = session.get_inputs()[0]  # 获取输入信息
    input_name = input_info.name
    input_shape = input_info.shape
    input_type = input_info.type
    print("ONNX模型的输入名称: ", input_names)
    print("ONNX模型的输出名称: ", output_names)
    
    onnx_output = session.run(None, input_data)  # 运行ONNX模型
    onnx_output_info = session.get_outputs()[0]  # 获取输出信息
    onnx_output_name = onnx_output_info.name
    onnx_output_shape = onnx_output_info.shape
    
    onnx_output_data = onnx_output[0]
    onnx_output_data = onnx_output_data.flatten()
    

    # print("onnx_output_info :", onnx_output_info )
    # print("onnx_output_name :", onnx_output_name )
    # print("onnx_output_shape :", onnx_output_shape )
    # print("onnx_output_data.shape: ", onnx_output_data.shape)
    # print("onnx_output:", onnx_output)
    print("onnx_output_data :", onnx_output_data )
    # print(onnx.helper.printable_graph(onnx_model.graph))
    
    return onnx_output_data
  
  
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

def relay_compile(inputs, model_path):
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
  input_name = ['inputs']
  shape_dict = {input_name[0]: inputs.shape}
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

  executor = graph_executor.create(graph, lib, dev)
  executor.set_input('inputs', tvm.nd.array(inputs.cpu().detach().numpy(), dev))
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