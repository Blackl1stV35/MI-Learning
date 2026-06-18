import torch
ckpt = torch.load('models/dsac_pretrained.pt', map_location='cpu')
for k, v in ckpt['actor'].items():
    print(f"{k:40s} {tuple(v.shape)}")