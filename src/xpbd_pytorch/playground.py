import torch

t = torch.tensor([1.0, 0.0, 0.0])

larger_tensor = torch.full((2, 3, len(t)), float('nan'))

larger_tensor[0] = t

print(larger_tensor)