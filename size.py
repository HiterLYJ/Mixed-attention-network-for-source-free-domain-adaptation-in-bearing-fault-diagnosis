import torch
import torch.nn as nn
from torch.nn import functional as F
import network

if __name__ == "__main__":
    a = torch.linspace(1,128,128)
    a = torch.unsqueeze(a, 0)
    a = torch.unsqueeze(a, 0)
    b, c, H, W = a.size()
    print(W)