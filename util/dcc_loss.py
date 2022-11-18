import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

city_cls_num_list = [
    2036071935,
    336037810,
    1259785820,
    36211593,
    48485660,
    67767822,
    11509768,
    30521277,
    878719065,
    63964556,
    221461664,
    67201112,
    7444743,
    386482742,
    14774826,
    12995272,
    12863901,
    5445705,
    22848390
]
syn_cls_num_list = [
    1185290631,
    1238150643,
    1877694687,
    17349953,
    17092879,
    66859757,
    2471516,
    6790000,
    654951069,
    1e-5,
    438904043,
    270470344,
    29551148,
    260325897,
    1e-5,
    95771398,
    1e-5,
    13280308,
    13896624
]
gtav_cls_num_list = [
    8237943280,
    1931023594,
    3694301950,
    474412231,
    196878425,
    284853377,
    31996530,
    23590587,
    1741486285,
    569004561,
    3693099178,
    84462036,
    7145664,
    610411201,
    286816852,
    115366961,
    28849798,
    7545186,
    1567045
]


class BalancedSoftmaxCE(nn.Module):
    r"""
    References:
    Ren et al., Balanced Meta-Softmax for Long-Tailed Visual Recognition, NeurIPS 2020.
    Equation: Loss(x, c) = -log(\frac{n_c*exp(x)}{sum_i(n_i*exp(i)})
    """

    def __init__(self, ignore_index=None, reduction='mean', dataset='gtav'):
        super(BalancedSoftmaxCE, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)
        if dataset == 'gtav':
            self.num_class_list = gtav_cls_num_list
        elif dataset == 'cityscapes':
            self.num_class_list = city_cls_num_list
        elif dataset == 'syn':
            self.num_class_list = syn_cls_num_list
        elif dataset == 'gtav_syn':
            self.num_class_list = np.sum([gtav_cls_num_list, syn_cls_num_list],axis=0).tolist()
        self.bsce_weight = torch.FloatTensor(self.num_class_list)

    def forward(self, pred, targets):
        pred = pred + self.bsce_weight.reshape(1, pred.shape[1], 1, 1).log().to(pred.device)
        return self.criterion(pred, targets)