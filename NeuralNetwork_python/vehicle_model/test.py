import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd

cmd = np.arange(0, 10, 5)
speed = np.arange(0, 0.6, 0.2)
cmd, speed = np.meshgrid(cmd, speed)
input_data = np.column_stack((cmd.ravel(), speed.ravel()))

a = [1, 2, 3, 4, 5]
print(f"ori: {a}\n")
a = np.vstack(a)
print(f"v: {a}\n")
a = np.hstack(a)
print(f"h: {a}\n")
a = np.stack(a)
print(f"stack: {a}\n")
