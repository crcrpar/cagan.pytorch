import os
from PIL import Image


import torch


def _default_loader(path):
    return Image.open(path).convert('RGB')


class TripletDataset(torch.utils.data.Dataset):

    def __init__(self, root, triplet_file_path, loader=_default_loader,
                 transform=None):
        self.root = root
        with open(triplet_file_path) as f:
            self.triplets = [line.strip('\n') for line in f]
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        images = self.triplets[index]
        images = [os.path.join(self.root, filepath) for filepath in images]
        images = [self.loader(path) for path in images]
        if self.transform is not None:
            images = [self.transform(image) for image in images]
        return images


class PairDataset(torch.utils.data.Dataset):

    def __init__(self, root, pair_file_path, loader=_default_loader,
                 transform=None):
        self.root = root
        with open(pair_file_path) as f:
            self.pairs = [line.strip('\n') for line in f]
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        images = self.pairs[index]
        images = [os.path.join(self.root, filepath) for filepath in images]
        images = [self.loader(path) for path in images]
        if self.transform is not None:
            images = [self.transform(image) for image in images]
        return images
