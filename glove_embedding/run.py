# @Author:Yifx
# @Contact: Xxuyifan1999@163.com
# @Time:2020/10/23 15:52
# @Software: PyCharm

"""
文件说明：
"""
from data.Wiki_Dataset import Wiki_Dataset
from model.glove_model import glove_model
import config as argumentparser
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm




if __name__ == '__main__':
    config = argumentparser.ArgumentParser()
    # 设置gpu
    if config.cuda and torch.cuda.is_available():
        print("using gpu ..... ")
        torch.cuda.set_device(config.gpu)
    # 导入数据集
    min_count = config.min_count
    window_size = config.window_size
    wiki_dataset = Wiki_Dataset(min_count=min_count,window_size=window_size)
    trainning_iter = torch.utils.data.DataLoader(dataset=wiki_dataset,
                                                 batch_size=config.batch_size,
                                                 shuffle=True,
                                                 num_workers=2)
    # 模型
    model = glove_model(vocab_size=len(wiki_dataset.word2id),embed_size=config.embed_size,alpha=config.alpha,x_max=config.x_max)
    if config.cuda and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
        model.cuda()

    optimizer = optim.Adam(model.parameters(),lr=config.learning_rate)
    loss=-1

    # 训练
    for epoch in range(config.epoch):
        process_bar = tqdm(trainning_iter)
        for data,label in process_bar:
            w_data = torch.Tensor(np.array([sample[0] for sample in data])).long()
            v_data = torch.Tensor(np.array([sample[1] for sample in data])).long()
            if config.cuda and torch.cuda.is_available():
                torch.cuda.set_device(config.gpu)
                w_data = w_data.cuda()
                v_data = v_data.cuda()
                label = label.cuda()
            loss_now = model(w_data,v_data,label)
            if loss == -1:
                loss = loss_now.data.item()
            else:
                loss = 0.95 * loss + 0.05 * loss_now.data.item()

            process_bar.set_postfix(loss=loss)
            process_bar.update()
            optimizer.zero_grad()
            loss_now.backward()
            optimizer.step()
    # 模型保存
    model.save_embedding(wiki_dataset.word2id,"./embeddings/result.txt")

