import torch

if __name__ == '__main__':
    a = torch.tensor([[1], [2]])
    b = torch.tensor([[0.1], [0.2]])
    c = (1 - b) * a
    print(c)
