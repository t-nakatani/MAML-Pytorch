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
import random

class MiniImagenet(Dataset):
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

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, cross_val_idx, startidx=0):
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
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batchsz, n_way, k_shot, k_query, resize))

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
        self.mode = mode
        self.cross_val_idx = cross_val_idx
        self.df_spt = pd.read_csv('support_cv2.csv')
        self.df_qry = pd.read_csv('query_cv2.csv')
        self.dic_spt = dict([(fname, label) for fname, label in zip(self.df_spt['fname'], self.df_spt['label'])])
        self.dic_qry = dict([(fname, label) for fname, label in zip(self.df_qry['fname'], self.df_qry['label'])])
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
        test_idx = (self.cross_val_idx+1) % 10
        
#         print(self.df_spt)
        
        if mode == 'train':
            df_spt = self.df_spt[self.df_spt['cv'] != 'test'] # drop val later
            df_qry = self.df_qry[self.df_qry['cv'] != 'test'] # drop val later
            df_spt = df_spt[df_spt['cv'] != self.cross_val_idx]
            df_spt = df_spt[df_spt['cv'] != self.cross_val_idx]
        elif mode == 'train_ts':
            df_spt = self.df_spt[self.df_spt['cv'] != 'test']
            df_qry = self.df_qry[self.df_qry['cv'] != 'test']
        elif mode == 'test':
            df_spt = self.df_spt[self.df_spt['cv'] == 'test']
            df_qry = self.df_qry[self.df_qry['cv'] == 'test']
        elif mode == 'val':
            df_spt = self.df_spt[self.df_spt['cv'] == str(self.cross_val_idx)]
            df_qry = self.df_qry[self.df_qry['cv'] == str(self.cross_val_idx)]
        else: 
            print('===invalid mode===', mode)
#         print(df_spt, type(self.cross_val_idx))
        pres = df_spt['pre'].unique()
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            support_x = [] # ex) [['0_10.png', '0_12.png'], ['0_0.png', '0_2.png']]
            query_x = []
            while(True):
                selected_flower = np.random.choice(pres)
                if selected_flower != '182':
                    break
            df_tmp_s = df_spt[df_spt['pre'] == selected_flower]
            df_tmp_q = df_qry[df_qry['pre'] == selected_flower]
            candi_s0 = list(df_tmp_s[df_tmp_s['label'] == 0]['fname'])
            candi_q0 = list(df_tmp_q[df_tmp_q['label'] == 0]['fname'])
            candi_s1 = list(df_tmp_s[df_tmp_s['label'] == 1]['fname'])
            candi_q1 = list(df_tmp_q[df_tmp_q['label'] == 1]['fname'])

                # 2. select k_shot + k_query for each class
            try:

                selected_imgs_qry0 = random.sample(candi_q0, self.k_query)
                selected_imgs_spt0 = random.sample(candi_s0, self.k_shot)
                selected_imgs_qry1 = random.sample(candi_q1, self.k_query)
                selected_imgs_spt1 = random.sample(candi_s1, self.k_shot)
            except:
                print(selected_flower)
            
            np.random.shuffle(selected_imgs_spt0) # いらん気がする
            np.random.shuffle(selected_imgs_qry0)
            np.random.shuffle(selected_imgs_spt1)
            np.random.shuffle(selected_imgs_qry1)
            for k in range(self.k_shot):
                support_x.append([selected_imgs_spt0[k], selected_imgs_spt1[k]])
                random.shuffle(support_x[-1])
            for k in range(self.k_query):
                query_x.append([selected_imgs_qry0[k], selected_imgs_qry1[k]])
                random.shuffle(query_x[-1])
#             break

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
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
        
        path1 = '../flower_drop/images/'
        path2 = '../flower_natural2_drop/images/'
        flatten_support_x = [os.path.join(path1, item)
                             for sublist in self.support_x_batch[index] for item in sublist]

        support_y = np.array([self.dic_spt[fname] for sublist in self.support_x_batch[index] for fname in sublist])

        flatten_query_x = [os.path.join(path2, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.dic_qry[fname] for sublist in self.query_x_batch[index] for fname in sublist])

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
