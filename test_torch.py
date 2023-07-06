import json
import torch

data = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda
}

json_data = json.dumps(data, indent=4)
print(json_data)