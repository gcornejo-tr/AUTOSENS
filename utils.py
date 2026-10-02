import os
import torch.nn as nn
from pathlib import Path
import math


def get_sequences(folder_path):

    sequences_list = []

    for root, _, _ in os.walk(folder_path):
        if root[-1] == "n" or root[-1]=="p":
            sequences_list.append(root)

    sequences_list.sort()

    return sequences_list

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def init_dual_head_resnet(model, prior_prob=0.5):
    model.apply(init_weights)

    cls_output_layer = model.fc_class[-2]
    nn.init.kaiming_normal_(
        cls_output_layer.weight, mode="fan_out", nonlinearity="relu"
    )
    if cls_output_layer.bias is not None:
        cls_bias_init = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(cls_output_layer.bias, cls_bias_init)

    dist_output_layer = model.fc_distance[-2]
    nn.init.kaiming_normal_(
        dist_output_layer.weight, mode="fan_out", nonlinearity="relu"
    )
    if dist_output_layer.bias is not None:
        nn.init.constant_(dist_output_layer.bias, 0.0)

