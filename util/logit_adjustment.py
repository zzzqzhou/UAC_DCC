import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def compute_adjustment(tro=1.0, dataset_name='gtav'):
    """compute the base probabilities"""
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
        0,
        438904043,
        270470344,
        29551148,
        260325897,
        0,
        95771398,
        0,
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

    label_freq = None
    if dataset_name == 'gtav':
        label_freq = gtav_cls_num_list
    elif dataset_name == 'cityscapes':
        label_freq = city_cls_num_list
    elif dataset_name == 'syn':
        label_freq = syn_cls_num_list
    elif dataset_name == 'gtav_syn':
        label_freq = np.sum([gtav_cls_num_list, syn_cls_num_list],axis=0).tolist()
    else:
        raise ValueError('Not support dataset {}.'.format(dataset_name))
    
    label_freq_array = np.array(label_freq)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    return adjustments