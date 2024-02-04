import collections

import torch

if __name__ == '__main__':
    q = collections.deque(maxlen=10)
    q.append([1, 2, 3])
    q.append([4, 5, 6])
    a = list(zip(*q))
    print(a)
