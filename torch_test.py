import torch
print(torch.cuda.is_available())  # Should print True
print(torch.version.cuda)         # Should print the CUDA version
print(torch.cuda.get_device_name(0))  # Should print your GPU's name