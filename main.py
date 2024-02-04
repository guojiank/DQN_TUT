import collections

import torch

if __name__ == '__main__':
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    b = a.max(1)[0]
    print(b)
    c = torch.zeros(10)
    print(c)
