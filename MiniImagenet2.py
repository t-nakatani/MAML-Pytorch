import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import collections
from PIL import Image
import csv
import random
import pandas as pd
import re

class MiniImagenet2(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0, idx = None):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
#         print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
#         mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(root, 'images')  # image path
        self.cls_num = 2
        if mode == 'pred':
            self.create_batch_for_predict(idx)
        else:
            self.create_batch(self.batchsz, mode)

    def create_batch(self, batchsz, mode):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        df = pd.read_csv('flower.csv')
        pres = df['pre'].unique()
        from sklearn.model_selection import train_test_split
        pres_tr, pres_ts = train_test_split(pres, random_state=0)
        dic = df.groupby('pre').count()['fname'].to_dict()
        if mode == 'train': pres = pres_tr
        else: pres = pres_ts
        print('===train_test_split===')
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            support_x = []
            query_x = []
            selected_flower = np.random.choice(pres)
                # 2. select k_shot + k_query for each class
            selected_imgs_idx = np.random.choice(dic[selected_flower]//2, self.k_shot + self.k_query, False)
            np.random.shuffle(selected_imgs_idx)
            indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
            indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
            support_x.append(list(df[df['pre'] == selected_flower].iloc[[indexDtrain[0], indexDtrain[0]+dic[selected_flower]//2], 0]))
            random.shuffle(support_x[-1])
            query_x.append(list(df[df['pre'] == selected_flower].iloc[[indexDtest[0], indexDtest[0]+dic[selected_flower]//2], 0]))
            random.shuffle(query_x[-1])
#             break

            # shuffle the correponding relation between support set and query set
#             random.shuffle(support_x[0])
#             random.shuffle(query_x[0])

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets
            
    def create_batch_for_predict(self, idx):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        df = pd.read_csv('flower.csv')
        df_ = pd.read_csv('flower_natural_with_label.csv')
        df_ = df_[df_['label'] != '-']
        pres = list(df.groupby('pre').count()['fname'].index)
        selected_flower = pres[idx]
        dic = df.groupby('pre').count()['fname'].to_dict()
        dic_ = df_.groupby('pre').count()['fname'].to_dict()

        support_x = []
        query_x = []

            # 2. select 1_shot + (filesize)_query for each class
        selected_imgs_idx = np.random.choice(dic[selected_flower]//2, 1, False)
        selected_imgs_idx_ = np.arange(dic_[selected_flower])
        
        indexDtrain = np.array(selected_imgs_idx[0])  # idx for Dtrain
        indexDtest = np.array(selected_imgs_idx_[:])  # idx for Dtest
        support_x.append(list(df[df['pre'] == selected_flower].iloc[[indexDtrain, indexDtrain+dic[selected_flower]//2], 0]))
        random.shuffle(support_x[-1])
#         print(selected_flower, dic_[selected_flower])
        for i in range(dic_[selected_flower]):
            query_x.append((df_[df_['pre'] == selected_flower].iloc[indexDtest[i], 0]))
#         random.shuffle(query_x)

        self.support_x_batch.append(support_x)  # append set to current sets
        self.query_x_batch.append([query_x])  # append sets to current sets
#         with open('data/log_query.txt', mode='a') as f:
#             f.write(','.join(query_x + ['\n']))
    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        df_ = pd.read_csv('flower_natural_with_label.csv')
        df_ = df_[df_['label'] != '-']
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        if self.k_query > 1:
            query_x = torch.FloatTensor(self.querysz//self.n_way, 3, self.resize, self.resize)
        else:
            query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)
        
        path = './flower/images/'
        flatten_support_x = [os.path.join(path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array([int(re.findall('.+_(\d+).png', item)[0]) % 2 for sublist in self.support_x_batch[index] for item in sublist])

        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([0 if str(list(df_[df_['fname'] == item]['label'])[0]) == 'r' else 1 for sublist in self.query_x_batch[index] for item in sublist])
        
#         with open('data/log_qry_order.txt', mode='a') as f:
#             f.write(re.findall('images/(.+)_', flatten_query_x[0])[0] + ',')
#         print(flatten_query_x[0])
        
#         print(self.query_x_batch[index], query_y)

        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(len(query_y)) #fit size of query_y_relative to one of query_y
#         query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        # print('relative:', support_y_relative, query_y_relative)
#         print((flatten_query_x))
        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)
        # print(support_set_y)
        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

#         return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=5, k_shot=1, k_query=1, batchsz=1000, resize=168)

    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_

        support_x = make_grid(support_x, nrow=2)
        query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())
        plt.pause(0.5)

        tb.add_image('support_x', support_x)
        tb.add_image('query_x', query_x)

        time.sleep(5)

    tb.close()
