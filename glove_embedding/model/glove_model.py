# @Author:Yifx
# @Contact: Xxuyifan1999@163.com
# @Time:2020/10/23 15:52
# @Software: PyCharm

"""
文件说明：
"""
import torch
import torch.nn as nn

class glove_model(nn.Module):
    def __init__(self,vocab_size,embed_size,alpha,x_max):
        super(glove_model, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.alpha = alpha
        self.x_max = x_max

        self.w_embed = nn.Embedding(self.vocab_size,embed_size).type(torch.float64)
        self.v_embed = nn.Embedding(self.vocab_size,embed_size).type(torch.float64)

        self.w_bias = nn.Embedding(self.vocab_size,1).type(torch.float64)
        self.v_bias = nn.Embedding(self.vocab_size,1).type(torch.float64)

    def forward(self, wdata,vdata,labels):
        w_data_embed = self.w_embed(wdata)
        v_data_embed = self.v_embed(vdata)

        w_data_bias = self.w_bias(wdata)
        v_data_bias = self.v_bias(vdata)

        weights = torch.pow(labels/self.x_max,self.alpha)
        weights[weights>1] = 1

        loss = torch.mean(weights.double()*torch.pow(torch.sum(w_data_embed.double()*v_data_embed.double(),1)+w_data_bias.double()+v_data_bias.double()-
                                 torch.log(labels).double(),2))
        return loss

    def save_embedding(self,word2id,file_name):
        embedding_1 = self.w_embed.weight.data.cpu().numpy()
        embedding_2 = self.v_embed.weight.data.cpu().numpy()
        embedding = (embedding_1 + embedding_2) / 2

        fout = open(file_name,'w')
        fout.write('%d %d\n' % (len(word2id),self.embed_size))
        for w,wid in word2id.items():
            e = embedding[wid]
            e = ''.join(map(lambda x :str(x),e))
            fout.write('%s %s\n' % (w,e))

if __name__ == '__main__':
    model = glove_model(vocab_size=100,embed_size=300,alpha=0.7,x_max=100)

    w_data = torch.Tensor([0, 0, 1, 1, 1]).long()
    v_data =  torch.Tensor([1, 2, 0, 2, 3]).long()
    labels = torch.Tensor([1,2,3,4,5])
    model.forward(w_data, v_data, labels)
    print(model.w_embed.weight.data.numpy())
