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
        sol1 = np.array(self.input_dict["sol1"][index])
        sol2 = np.array(self.input_dict["sol2"][index])
        sol1_att = np.array(self.input_dict["sol1_att"][index])
        sol2_att = np.array(self.input_dict["sol2_att"][index])
        sol1_token = np.array(self.input_dict["sol1_token"][index])
        sol2_token = np.array(self.input_dict["sol2_token"][index])
        label = np.array(self.input_dict['label'][index])
        return sol1, sol2, sol1_att, sol2_att, \
               sol1_token, sol2_token, label
