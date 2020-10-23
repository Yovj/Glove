# @Author:Yifx
# @Contact: Xxuyifan1999@163.com
# @Time:2020/10/23 15:51
# @Software: PyCharm

"""
文件说明：
"""
from torch.utils import data
import os
import numpy as np
import pickle

# 自定义Dataset 继承自data.DataLoader
# 必须重写 __len__ 以及 __getitem__ 两个方法，否则会报错
class Wiki_Dataset(data.DataLoader):
    def __init__(self,min_count,window_size):
        self.min_count = min_count        # 用于剔除低频词
        self.window_size = window_size    # 滑动窗口大小，用于计算共现矩阵
        self.path = os.path.abspath('.')  # 当前文件路径
        self.word2id = {}
        self.word_freq = {}
        self.datas,self.labels = self.get_data() # 作者提出，不需要使用整个共现矩阵进行计算（稀疏），因此我们只提取矩阵中的非零部分。datas为索引,labels为值

    def read_data(self):
        #print(self.path)
        data = open(self.path+"/data/text8.txt").read()
        data = data.split()

        # 计算词频
        print("get word2freq..........")
        for word in data:
            if self.word_freq.get(word) == None:
                self.word_freq[word] = 1
            else:
                self.word_freq[word] += 1
        print(len(self.word_freq))
        # word2id
        print("get word2id..........")
        for word in list(self.word_freq.keys()):
            if self.word_freq.get(word) == None or self.word_freq.get(word) < self.min_count:
                continue
            else:
                self.word2id[word] = len(self.word2id)
        print(len(self.word2id))
        return data



    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index],self.labels[index]

    def get_data(self):
        if not os.path.exists(self.path+"/data/label.npy"):
            print("Processing data ... ")
            data = self.read_data()
            print("Generating co-occurrences ...")
            # 计算共现矩阵
            vocab_size = len(self.word2id)
            comat = np.zeros((vocab_size,vocab_size))

            for i in range(len(data)):
                if self.word2id.get(data[i])==None:
                    continue
                i_index = self.word2id[data[i]]
                for j in range(max(0,i-self.window_size),min(i+self.window_size,len(data))):
                    if i==j or self.word2id.get(data[j]) == None:
                        continue
                    j_index = self.word2id[data[j]]
                    comat[i_index][j_index] += 1
            non_zero_coo = np.transpose(np.nonzero(comat))
            labels = []
            for i in range(len(non_zero_coo)):
                labels.append(comat[non_zero_coo[i][0]][non_zero_coo[i][1]])
            labels = np.array(labels)
            np.save(self.path+"/data.npy",non_zero_coo)
            np.save(self.path+"/label.npy",labels)
            pickle.dump(self.word2id,open(self.path+"/word2id","wb"))
            return non_zero_coo,labels


        else:
            coocs = np.load(self.path+"/data/data.npy")
            labels = np.load(self.path+"/data/label.npy")
            self.word2id = pickle.load(open(self.path+"/data/word2id","rb"))
        return coocs,labels


if __name__ == '__main__':
    wiki_dataset = Wiki_Dataset(min_count=50,window_size=2)
    print(wiki_dataset.datas.shape)
    print(wiki_dataset.labels.shape)
    print (wiki_dataset.labels[0:100])