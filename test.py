import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.backends.cudnn.enabled)
print(torch.backends.cudnn.deterministic)
