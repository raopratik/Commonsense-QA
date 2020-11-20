import sys

sys.path.append(".")

from torch.utils import data
import numpy as np


class DownstreamDataset(data.Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __len__(self):
        return len(self.input_dict["label"])

    def __getitem__(self, index):
        ans = np.array(self.input_dict["ans"][index])
        label = np.array(self.input_dict['label'][index])
        return ans, label
