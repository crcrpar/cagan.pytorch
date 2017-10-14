from datetime import datetime as dt
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from tensorboardX import SummaryWriter


def patch_loss(y_in, y_out, positive=True):
    batch_size, _, w, h = y_in.size()
    denom = batch_size * w * h
    if positive:
        l1 = F.softplus(y_in).sum()
    else:
        l1 = F.softplus(-y_in).sum()
    l2 = F.softplus(y_out).sum()
    return (l1 + l2) / denom


class Trainer(object):

    def __init__(self, root, use_tensorboard=True, use_plateau_decay=True):
        time = dt.now().strftime("%m%d_%H%M")
        root = os.path.join(root, time)
        if not os.path.isdir(root):
            os.makedirs(root)
        self.root = root

    def run(self, gen, opt_gen, dis, opt_dis, loader,
