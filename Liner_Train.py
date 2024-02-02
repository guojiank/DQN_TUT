import random

import torch.nn
from tqdm import tqdm, trange

# Get cpu, gpu or mps device for training. cuda
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def f(x):
    return 2 * x + 11


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, 4),
            torch.nn.Linear(4, 1),
        )

    def forward(self, x):
        return self.model(x)


def train():
    net = Net().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    with trange(100000) as bar:
        for _ in bar:
            lst = [[i] for i in range(1, 1000)]
            inputs = random.sample(lst, k=100)
            outputs = net(torch.tensor(inputs, dtype=torch.float).to(device))
            labels = torch.tensor([[f(i[0])] for i in inputs], dtype=torch.float)
            optimizer.zero_grad()
            loss = loss_fn(outputs.to(device), labels.to(device))
            loss.backward()
            optimizer.step()
            bar.set_description(f'Loss: {loss}, size: {len(str(int(loss.item())))}')
    r = net(torch.tensor([[1], [2]], dtype=torch.float).to(device))
    print(r)


if __name__ == '__main__':
    train()
