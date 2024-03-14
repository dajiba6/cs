import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


data = pd.read_csv("./record.csv")

X_train = data[
    [
        "throttle_percentage",
        "brake_percentage",
        "steering_percentage",
        "speed_mps",
        "acceleration_current_point",
    ]
].head()

y_train = data[["acceleration_next_point", "angular_velocity_vrf"]].head()

print(f"{X_train}")

X_tensor = torch.tensor(X_train.values)
print(f"{X_tensor}")
