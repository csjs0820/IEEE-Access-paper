import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def server_data(dataset,num_data):

    server_dataset = np.array([], dtype='int64')
    all_idxs = [i for i in range(len(dataset))]
    all_labels = np.array(dataset.targets)

    #num_items = int(len(dataset) / num_users)
    num_items= num_data

    idxs_labels = np.vstack((all_idxs, all_labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    allocate = [0 for i in range(len(dataset))]

    class_size = int(len(dataset) / 10)

    temp = np.array([], dtype='int64')
    temp3 = []

    for k in range(10):
        temp2 = []
        while len(temp2) < int(num_items/10):
            if k == 0:
                pick = int(np.random.choice(idxs[0:5000],1, replace=False))
                if allocate[pick] == 0:
                    temp2.append(pick)
                    allocate[pick] = 1
            else:
                pick = int(np.random.choice(idxs[(class_size * k):((k + 1) * class_size)],1, replace=False))
                if allocate[pick] == 0:
                    temp2.append(pick)
                    allocate[pick] = 1
        temp = np.append(temp,temp2)

    server_dataset = np.append(server_dataset,temp)
    return server_dataset

def uniform_iid(dataset,num_users):

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    all_idxs = [i for i in range(len(dataset))]
    all_labels = np.array(dataset.targets)

    #num_items = int(len(dataset) / num_users)
    num_items= 500

    idxs_labels = np.vstack((all_idxs, all_labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    allocate = [0 for i in range(len(dataset))]

    class_size = int(len(dataset) / 10)

    temp = np.array([], dtype='int64')
    temp3 = []

    for q in range(num_users):
        for k in range(10):
            temp2 = []
            while len(temp2) < int(num_items/10):
                if k == 0:
                    pick = int(np.random.choice(idxs[0:5000],1, replace=False))
                    if allocate[pick] == 0:
                        temp2.append(pick)
                        allocate[pick] = 1
                else:
                    pick = int(np.random.choice(idxs[(class_size * k):((k + 1) * class_size)],1, replace=False))
                    if allocate[pick] == 0:
                        temp2.append(pick)
                        allocate[pick] = 1
            temp = np.append(temp,temp2)

        dict_users[q] = np.append(dict_users[q],temp)
        for m in dict_users[q]:
            temp3 = np.append(temp3,idxs_labels[1,np.where(idxs==m)])

        temp = np.array([],dtype='int64')
        #plt.hist(temp3)
        #plt.show()
        temp3 = []

    return dict_users

def hybrid1(dataset, num_users, gamma):

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    all_idxs = [i for i in range(len(dataset))]
    all_labels = np.array(dataset.targets)

    num_items = int(len(dataset) / num_users)

    iid_client = int(num_users * gamma)

    idxs_labels = np.vstack((all_idxs, all_labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    allocate = [0 for i in range(len(dataset))]

    class_size = int(len(dataset)/10)

    temp = np.array([],dtype='int64')
    temp3 = []

    # uniform iid 클라이언트
    for q in range(iid_client):
        for k in range(10):
            temp2 = []
            while len(temp2) < 50:
                if k == 0:
                    pick = int(np.random.choice(idxs[0:5000],1, replace=False))
                    if allocate[pick] == 0:
                        temp2.append(pick)
                        allocate[pick] = 1
                else:
                    pick = int(np.random.choice(idxs[(class_size * k):((k + 1) * class_size)],1, replace=False))
                    if allocate[pick] == 0:
                        temp2.append(pick)
                        allocate[pick] = 1
            temp = np.append(temp,temp2)
        #print(len(idxs[(class_size * k)-50*q:((k + 1) * class_size)-50*q]))
        dict_users[q] = np.append(dict_users[q], temp)
        temp = np.array([], dtype='int64')

    num_imgs = num_items
    num_shards = int(len(idxs)/num_imgs)

    idx_shard = [i for i in range(num_shards)]

    if iid_client == 0:
        q = 0

    for j in range(q,num_users):
        rand_set = np.random.choice(idx_shard, 1, replace=False)
        idx_shard = list(set(idx_shard) - set(rand_set))
        dict_users[j] = idxs[int(rand_set) * num_imgs:(int(rand_set) + 1) * num_imgs]

    return dict_users


def hybrid2(dataset,num_users,gamma):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    all_idxs = [i for i in range(len(dataset))]
    all_labels = np.array(dataset.targets)

    num_items = int(len(dataset) / num_users)

    iid_client = int(num_users * gamma)

    idxs_labels = np.vstack((all_idxs, all_labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    allocate = [0 for i in range(len(dataset))]

    class_size = int(len(dataset) / 10)

    temp = np.array([], dtype='int64')
    temp3 = []

    # uniform iid 클라이언트
    for q in range(iid_client):
        for k in range(10):
            temp2 = []
            while len(temp2) < 50:
                if k == 0:
                    pick = int(np.random.choice(idxs[0:5000], 1, replace=False))
                    if allocate[pick] == 0:
                        temp2.append(pick)
                        allocate[pick] = 1
                else:
                    pick = int(np.random.choice(idxs[(class_size * k):((k + 1) * class_size)], 1, replace=False))
                    if allocate[pick] == 0:
                        temp2.append(pick)
                        allocate[pick] = 1
            temp = np.append(temp, temp2)
        # print(len(idxs[(class_size * k)-50*q:((k + 1) * class_size)-50*q]))

        dict_users[q] = np.append(dict_users[q], temp)
        # print(dict_users[q])
        for m in dict_users[q]:
            temp3 = np.append(temp3, idxs_labels[1, np.where(idxs == m)])
            # idxs_labels[1, np.where(idxs == m)] = -1

        temp = np.array([], dtype='int64')
        # print(temp3)
        # plt.hist(temp3)
        # plt.show()
        temp3 = []

    num_imgs = int(num_items / 2)
    num_shards = int(len(all_idxs) / num_imgs)

    idx_shard = [i for i in range(num_shards)]
    # idxs = np.arange(num_shards * num_imgs)
    check = [0 for i in range(len(dataset))]

    for j in range(q, num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[j] = np.concatenate((dict_users[j], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        # print([dict_users[j]])

    return dict_users

if __name__ == '__main__':
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar/', train=True, download=True, transform=trans_cifar)

    num = 100
    gamma = 0.06
    d = hybrid1(dataset_train,num,gamma)
