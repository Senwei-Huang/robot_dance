"""
************************************************************
** Detail: Neural network conversion and compilation tool.
** Author: Senwei Huang, Chong Tian
** Update: 2024-06-09
** Version: 1.0
************************************************************
"""
import os, os.path
import torch
import numpy as np
import onnx
import onnxruntime
import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.contrib import graph_executor
from tvm.contrib import cc


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        # print("models: ", models)
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        # print("models: ", models)
        model = models[-1]
        # print("models: ", model)
        checkpoint = model.split("_")[-1].split(".")[0]
        # print("checkpoint: ", checkpoint)
    else:
        if model_name_include=="model":
          model = "model_{}.pt".format(checkpoint) 
        else:
          model = "{}-panda_parkour.onnx".format(checkpoint)
    load_path = os.path.join(root, model)
    # print("load_path: ", load_path)
    return load_path, checkpoint


def pytorch_to_jit(torch_model, obs_input, jit_save_path):
    """
    torch_model: PyTorch模型实例, 定义了模型的结构并加载预训练权重文件
    """
    print("******************* Exporting model to JIT format *******************")
    traced_policy = torch.jit.trace(torch_model, obs_input)
    # script_policy = torch.jit.script(torch_model)
    # print("traced_policy: ", traced_policy)
    # print("script_policy: ", script_policy)

    # 调用模型
    traced_output = traced_policy(obs_input)
    # script_output = script_policy(obs_input, depth_input, hidden_input)
    # print("traced_output: ", traced_output)
    # print("script_output: ", script_output)

    # 保存模型
    traced_policy.save(jit_save_path)
    # script_policy.save(jit_save_path)
    print("Saved JIT model at: ", os.path.abspath(jit_save_path))


def get_jit_output(obs_input, depth_input, hidden_input, jit_save_path):
    # 加载模型
    model = torch.jit.load(jit_save_path)
    # print("model: ", model)
    model.eval()
    jit_output, jit_hidden = model(obs_input, depth_input, hidden_input)
    jit_output = jit_output.detach().numpy().flatten()
    return jit_output


def pytorch_to_onnx(torch_model: torch.nn.Module, obs_input, onnx_save_path: str):
    """
    torch_model: PyTorch模型实例, 定义了模型的结构并加载预训练权重文件
    """
    print("******************* Exporting model to ONNX format *******************")
    torch.onnx.export(
        torch_model,      
        obs_input,
        onnx_save_path,       
        export_params=True,  
        verbose=False,      
        opset_version=12,   
        do_constant_folding=True, 
        input_names=['obs_input'],
        output_names=['act_output'],)
    print("Saved ONNX model at: ", os.path.abspath(onnx_save_path))


def get_onnx_output(obs_input, depth_input, hidden_input, onnx_save_path):
    onnx_model = onnx.load(onnx_save_path)
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))
    input_data = {'obs_input': obs_input.detach().numpy(), 'depth_input': depth_input.detach().numpy(), 'hidden_input': hidden_input.detach().numpy()}
    session = onnxruntime.InferenceSession(onnx_save_path)
    onnx_out, onnx_hidden = session.run(None, input_data)
    print("onnx_out.shape: ", onnx_out.shape)
    onnx_out = onnx_out.flatten()
    return onnx_out


def get_torch_output(obs_input, depth_input, hidden_input, torch_model):
    torch_out, torch_hidden = torch_model(obs_input, depth_input, hidden_input)
    torch_out = torch_out.detach().numpy().flatten()
    return torch_out


def test_result(torch_out, onnx_out):
    """
    加载、检查和测试导出的ONNX模型
    """
    print("******************* Testing Convert Result *******************")
    try: 
      np.testing.assert_almost_equal(torch_out, onnx_out, decimal=4) 
      print("Result: Model outputs are closely matched!!!")
    except AssertionError as e: 
      print("Mismatch in outputs:", e)


def tvmc_compile(model_path, tvm_output_path):
  # Step 1: Load
  model = tvmc.load(model_path) 

  # Step 1.5: Optional Tune
  # print("################# Logging File #################") 
  # log_file = "merged_net_tune_record.json"
  # tvmc.tune(model, target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu", enable_autoscheduler = True, tuning_records=log_file) 

  # Step 2: Compile
  print("################# Network Converted #################") 
  # package = tvmc.compile(model, target="llvm", tuning_records=log_file, package_path=tvm_output_path)
  package = tvmc.compile(model, target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu", cross='aarch64-linux-gnu-gcc',package_path=tvm_output_path)
  # package_tuned = tvmc.compile(model, target="llvm -device=arm_cpu -mtriple=aarch64-none-linux-gnu",cross='aarch64-none-linux-gnu-gcc', tuning_records=log_file, package_path=tvm_output_path)
  # package = tvmc.compile(model, target="cuda", package_path=tvm_output_path)
  # package_tuned = tvmc.compile(model, target="cuda -arch=sm_87", target_host='llvm -mtriple=aarch64-linux-gnu',cross='aarch64-linux-gnu-gcc',package_path=tvm_output_path)

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


def cross_compiler_toolchain(file_name, files, **kwargs):
  # 调用交叉编译器
  return cc.cross_compiler(file_name, files, **kwargs)


def relay_compile(obs_input, depth_input, hidden_input, model_path):
  # 步骤1: 加载ONNX模型
  onnx_model = onnx.load(model_path)

  # 步骤2: 设置目标和目标主机
  target = 'llvm'
  # target = "llvm -mtriple=aarch64-linux-gnu"
  # target = "cuda"
  # target = 'cuda -arch=sm_87'
  # target_host = 'llvm -mtriple=aarch64-linux-gnu'
  dev = tvm.cpu(0)
  # dev = tvm.cuda(0)

  # 步骤3: 通过relay将ONNX模型转换为TVM中间表示
  input_name = ['obs_input', 'depth_input', 'hidden_input']
  shape_dict = {input_name[0]: obs_input.shape, input_name[1]: depth_input.shape, input_name[2]: hidden_input.shape}
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
  executor.set_input('obs_input', tvm.nd.array(obs_input, dev))
  executor.set_input('depth_input', tvm.nd.array(depth_input, dev))
  executor.set_input('hidden_input', tvm.nd.array(hidden_input, dev))
  executor.set_input(**params)
  executor.run()
  tvm_output = executor.get_output(0)
  tvm_hidden = executor.get_output(1)
  tvm_output = tvm_output.asnumpy().flatten()
  return tvm_output