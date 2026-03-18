import torch
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
    print("GPU名称:", torch.cuda.get_device_name(0))
    print("GPU计算能力:", torch.cuda.get_device_capability(0))