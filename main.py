import random

import torch.nn


def f(x):
    return 2 * x + 3


def f_(x):
    return 2 * x + 3 + random.uniform(-1, 1) * 2


def dataset():
    lst = [i for i in range(1000)]
    inputs = random.sample(lst, k=100)
    outputs = [f_(i) for i in inputs]
    labels = [f(i) for i in inputs]
    return inputs, outputs, labels


class N(torch.nn.Module):
    def __init__(self):
        super(N, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    data = dataset()
    a = torch.nn.MSELoss()
    print(a)
    pass
