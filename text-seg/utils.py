import torch
from torch.utils import data

class TextSegDS(data.Dataset):
    def __init__(self, inputs, labels):
        """
        :inputs - list of outputs (dicts) from transformer's tokenizer.encode_plus
        :labels - list of int: 1 if from same section; 0 otherwise
        """
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]