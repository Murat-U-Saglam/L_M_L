import json
import torch

data = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "Number of devices": torch.cuda.device_count(),
    "Name of device": torch.cuda.get_device_name(torch.cuda.current_device())
}

json_data = json.dumps(data, indent=4)
print(json_data)


