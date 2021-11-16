#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from collections import OrderedDict

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        #print(len(w))
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def weighted_FedAvg(w,coefficients):
    averaged_weights = OrderedDict()

    for it, idx in enumerate(w):
        for key in idx.keys():
            if it == 0:
                averaged_weights[key] = coefficients[it] * idx[key]
            else:
                averaged_weights[key] += coefficients[it] * idx[key]
    return averaged_weights


def weight_divergence(iid_w, local_w,local_num):

    client_wd = []
    server_wd = []

    conv1_w = torch.flatten(local_w['conv1.weight'])
    conv1_b = torch.flatten(local_w['conv1.bias'])
    conv2_w = torch.flatten(local_w['conv2.weight'])
    conv2_b = torch.flatten(local_w['conv2.bias'])
    fc1_w = torch.flatten(local_w['fc1.weight'])
    fc1_b = torch.flatten(local_w['fc1.bias'])
    fc2_w = torch.flatten(local_w['fc2.weight'])
    fc2_b = torch.flatten(local_w['fc2.bias'])
    fc3_w = torch.flatten(local_w['fc3.weight'])
    fc3_b = torch.flatten(local_w['fc3.bias'])

    client_wd = torch.cat([conv1_w,conv1_b,conv2_w,conv2_b,fc1_w,fc1_b,fc2_w,fc2_b,fc3_w,fc3_b])

    conv1_w = torch.flatten(iid_w['conv1.weight'])
    conv1_b = torch.flatten(iid_w['conv1.bias'])
    conv2_w = torch.flatten(iid_w['conv2.weight'])
    conv2_b = torch.flatten(iid_w['conv2.bias'])
    fc1_w = torch.flatten(iid_w['fc1.weight'])
    fc1_b = torch.flatten(iid_w['fc1.bias'])
    fc2_w = torch.flatten(iid_w['fc2.weight'])
    fc2_b = torch.flatten(iid_w['fc2.bias'])
    fc3_w = torch.flatten(iid_w['fc3.weight'])
    fc3_b = torch.flatten(iid_w['fc3.bias'])

    server_wd = torch.cat([conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b,fc3_w,fc3_b])

    client_wd = torch.div(torch.norm(client_wd-server_wd),torch.norm(server_wd))

    print("client {} : {}".format(local_num,client_wd))

    return client_wd