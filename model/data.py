import pickle
import torch
from torch.utils.data import Dataset


class Corpus(Dataset):
    """
    Corpus dataset
    Args:
        filename (string): pickle format
        token_fn (function): function that makes tokens to indices
        label_fn (function): function that makes labels to indices
    """
    def __init__(self, filename, token_fn, label_fn):
        with open(filename, 'rb') as f:
            self.corpus = pickle.load(f)
        self.token_fn = token_fn
        self.label_fn = label_fn

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        tokens, labels = map(lambda elm: elm, self.corpus[idx])
        tokens2indices = torch.tensor(self.token_fn(tokens))
        labels2indices = torch.tensor(self.label_fn(labels))
        length = torch.tensor(len(tokens2indices))
        return tokens2indices, labels2indices, length