#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    labels = dataset.train_labels.numpy()

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()
    #labels = dataset.train_labels
    print(type(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        print(labels[dict_users[i]])
        print(dict_users[i])
    return dict_users

def mnist_size_imbalance(dataset, num_users):
    '''
    :param dataset:
    :param num_users:
    :return:
    '''
    num_items = int(len(dataset) / num_users)
    print("할당 가능한 최대 data 수 : ", num_items)
    list_size =[]

    sum = 0
    #num_max = len(dataset) - (num_users*100}
    num_max = len(dataset)

    #client 별 dataset size 결정
    for i in range(num_users):
        size = np.random.randint(100,900)
        num_max -= size
        sum += size
        list_size.append(size)

    print("total data size : ", sum)
    print(list_size)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    labels = dataset.train_labels.numpy()
    temp = []
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, list_size[i], replace=False))
        #print("user ", i, " : ", sorted(dict_users[i]))
        #print(dict_users[i])
        all_idxs = list(set(all_idxs) - dict_users[i])
        for j in dict_users[i]:
            temp.append(labels[j])
    #plt.hist(temp)
    plt.bar(range(len(list_size)),list_size)
    plt.title('size_imbalance, client dataset size')
    plt.show()
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    print(len(dataset))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
    #mnist_iid(dataset_train,num)
    #mnist_size_imbalance(dataset_train,100)
