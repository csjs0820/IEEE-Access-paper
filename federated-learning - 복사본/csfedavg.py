#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid,mnist_size_imbalance, cifar_iid
from utils.setting import hybrid2, uniform_iid, server_data, hybrid1
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, weight_divergence, weighted_FedAvg
from models.test import test_img

# *추가 excel로 내보내기 위한 라이브러리
import pandas as pd
import openpyxl
import os
import datetime

# * 네트워크 시각화 *
from torchinfo import summary

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        #server iid dataset 할당
        num_server = args.server_data

        idxs = np.arange(len(dataset_train))
        labels = dataset_train.train_labels.numpy()
        temp =[[0] for _ in range(10)]
        print(temp)
        for label in labels:
            temp[label][0] += 1
        print(temp)

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        server_data = []

        start = 0
        server_data = set(np.random.choice(idxs, 500, replace=False))
        #for i in range(0,10):
        #    server_data.append((np.random.choice(idxs[start:start+5001], 10, replace=False)))
        #    start += 6000

        #print(labels[server_dataset])

        temp = []
        for i in server_data:
            temp.append(labels[i])
        plt.hist(temp)
        plt.show()

        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
        #else:
            #dict_users = mnist_size_imbalance(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar/', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar/', train=False, download=True, transform=trans_cifar)

        # server iid dataset 할당
        server_dataset = server_data(dataset_train,args.server_data)

        if args.iid == 'iid':
            dict_users = cifar_iid(dataset_train, args.num_users)
            print("iid distribution complete")

        elif args.iid == 'hybrid1':
            dict_users = hybrid1(dataset_train, args.num_users, 0)
            print("hybrid1 distribution complete")

        elif args.iid == 'hybrid2':
            dict_users = hybrid2(dataset_train, args.num_users, 0.06)
            print("hybrid2 distribution complete")

        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        net_glob2 = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    #print(net_glob)
    #summary(net_glob)

    net_glob.train()
    net_glob2.train()

    # copy weights
    w_glob = net_glob.state_dict()
    net_glob2.load_state_dict(w_glob)


    # training
    loss_train = []
    test_acc = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    alpha = 0.1

    random_client = set()
    final_client = set()
    candidate_client = set()

    dk = [[ 0 for i in range(args.num_users)] for i in range(2)]
    dj = [[0 for i in range(int(args.frac*args.num_users))] for i in range(2)]
    w_locals = [w_glob for i in range(args.num_users)]
    ww_locals = [w_glob for i in range(30)]

    wd = [[0 for i in range(args.num_users)] for i in range(args.epochs)]
    avg_wd = [0 for i in range(args.epochs)]

    for iter in range(args.epochs):
        loss_locals = []
        w_locals = [0 for i in range(args.num_users)]
        ww_local = []

        m = max(int(args.frac * args.num_users), 1)
        random_client = set(np.random.choice(range(args.num_users), m, replace=False))

        idxs_users = random_client | candidate_client | final_client

        print("client trainig")
        for idx in idxs_users:

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            #w_locals.append(copy.deepcopy(w))
            w_locals[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

        if len(final_client) < int(alpha*args.num_users):
            iid_model = LocalUpdate(args=args, dataset=dataset_train, idxs=server_dataset)
            iid_w, loss_w = iid_model.train(net=copy.deepcopy(net_glob2).to(args.device))
            #print("server weight : ",iid_w)

            # 알고리즘 2
            if len(candidate_client) != 0:
                candidate_client =list(candidate_client)
                random_client = list(random_client)

                for i in range(len(candidate_client)):
                    dk[0][i] = candidate_client[i]
                    dk[1][i] = weight_divergence(iid_w, w_locals[candidate_client[i]], candidate_client[i])
                for j in range(len(random_client)):
                    dj[0][j] = random_client[j]
                    dj[1][j] = weight_divergence(iid_w, w_locals[random_client[j]], random_client[j])

                dj = np.array(dj)
                dj = dj[:,dj[1,:].argsort()]
                random_client = dj[0]

                candidate_client = set(candidate_client)
                for q in range(len(candidate_client)):
                    i = int(alpha*len(random_client))
                    if dk[1][q] <= dj[1][i]:
                        final_client = final_client | {dk[0][q]}
                        candidate_client = candidate_client - set([dk[0][q]])

            random_client =list(random_client)
            candidate_client = candidate_client | set(random_client[0:int(alpha*len(random_client))])
            random_client = set(random_client)


            last_client = {}
            last_client = random_client | final_client

            print("========final client < alpha * K==========")
            print("candidate client : ", candidate_client)
            print("final client : ", final_client)
            print("last client : ",last_client)
            ww_locals = []
            for client in last_client:
                ww_locals.append(w_locals[client])

            selected_total_size = sum(len(dict_users[idx]) for idx in last_client)
            mixing_coefficients = [len(dict_users[idx]) / selected_total_size for idx in last_client]

            w_glob = weighted_FedAvg(ww_locals,mixing_coefficients)
            #w_glob = FedAvg(ww_locals)

        elif len(final_client) == int(alpha*args.num_users):
            iid_model = LocalUpdate(args=args, dataset=dataset_train, idxs=server_dataset)
            iid_w, loss_w = iid_model.train(net=copy.deepcopy(net_glob).to(args.device))

            last_client = {}
            last_client = random_client | final_client

            print("========final client >= alpha * K==========")
            print("candidate client : ", candidate_client)
            print("final client : ", final_client)
            print("last client : ", last_client)

            ww_locals = []
            for client in last_client:
                ww_locals.append(w_locals[client])

            selected_total_size = sum(len(dict_users[idx]) for idx in last_client) + args.server_data
            mixing_coefficients = [len(dict_users[idx]) / selected_total_size for idx in last_client]

            ag_w = weighted_FedAvg(ww_locals, mixing_coefficients)

            temp = copy.deepcopy(iid_w)
            for k in temp.keys():
                temp[k] = (args.server_data/selected_total_size)*temp[k]
                w_glob[k] = temp[k] + ag_w[k]

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob2.load_state_dict(iid_w)

        net_glob.eval()
        net_glob2.eval()
        acc_test, loss_test1 = test_img(net_glob, dataset_test, args)
        acc_test2, loss_test2 = test_img(net_glob2, dataset_test, args)

        print('idd_model Round {:3d}, Average loss {:.3f} Test accuarcy {:.3f}'.format(iter,loss_test2, acc_test2))
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f} Test accuarcy {:.3f}'.format(iter, loss_avg,acc_test))
        loss_train.append(loss_avg)
        test_acc.append(float(acc_test))


    print("Round 당 평균 weight divergence : ", avg_wd)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('hybrid1')
    plt.savefig('./save/{}/loss_{}_{}_{}_C{}_iid{}.png'.format(args.iid,args.dataset, args.model, args.epochs, args.frac, args.iid))

    # plot acc curve
    plt.figure()
    plt.plot(range(len(test_acc)), test_acc)
    plt.title('hybrid1')
    plt.ylabel('test_acc')
    plt.savefig('./save/{}/acc_fed_{}_{}_{}_C{}_iid{}.png'.format(args.iid,args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    #save excel

    file = 'D:/federated-learning/save/export_acc.xlsx'

    if not os.path.isfile(file):
        wb = openpyxl.Workbook()
        wb.active.title = 'acc'
        new_filename = 'D:/federated-learning/save/export_acc.xlsx'

        wb.save(new_filename)
        count = 0

    '''
    else:
        pd.read_excel('D:/federated-learning/save/export_acc.xlsx',engine='openpyxl' )
        ws = pd.read_excel('D:/federated-learning/save/export_acc.xlsx',sheet_name = 'acc',engine='openpyxl')
    '''

    count = 0
    df = pd.DataFrame([[float(acc_train)],
                       [float(acc_test)],
                       ['fed_{}_{}_{}_C{}_iid{}_num{}'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.num_users)],
                       [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]],
                      index=['train accuracy', 'test_accuracy','inform', 'date'], columns=['test_{}'.format(count)])

    #df.to_excel('D:/federated-learning/save/export_acc.xlsx', header =True, index = True, sheet_name='acc')
    with pd.ExcelWriter('D:/federated-learning/save/export_acc.xlsx', mode = 'a',engine='openpyxl')as writer:
        df.to_excel(writer,sheet_name='acc')