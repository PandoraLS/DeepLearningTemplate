# -*- coding: utf-8 -*-
# @Time : 2021/4/21 下午9:07

"""
summary的一些用法，以rnn为例
"""



import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size

        # input_size是词典大小, hidden_size是词向量维度
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 这里nn.GRU(x, h)两个参数指明输入x和隐藏层状态的维度, 这里都用hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        :param input: 这里的input是每次一个词, 具体形式为: [word_idx]
        :param hidden:
        :return:
        """
        # input: [1]
        # embedding(input): [1, emb_dim]
        # embedded: [1, 1, 1 * emb_dim]
        embedded = self.embedding(input).view(1, 1, -1)

        # 关于gru的输入输出参数
        # [seq_len, batch_size, feture_size]
        # output: [1, 1, 1 * emb_dim]
        output = embedded
        # hidden: [1, 1, hidden_size]
        # 这里hidden_size == emb_dim
        output, hidden = self.gru(output, hidden)
        # output: [seq_len, batch, num_directions * hidden_size]
        # output: [1, 1, hidden_size]
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size

        # embedding层的结构，1. 有多少个词，2. 每个词多少维
        self.embedding = nn.Embedding(output_size, hidden_size)
        # GRU的参数: 1. 输入x的维度, 2. 隐藏层状态的维度; 这里都用了hidden_size
        # emb_dim == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        # [batch_size, hidden_size] -> [batch_size, output_size]
        # 这里output_size就是目标语言字典的大小V
        self.out = nn.Linear(hidden_size, output_size)
        # softmax层, 求每一个单词的概率
        self.softmax = nn.LogSoftmax(dim=1)  # ?

    def forward(self, input, hidden):
        # input: [1], 一个单词的下标
        # hidden: [1, 1, hidden_size]
        # embedding(input): [emb_dim]
        output = self.embedding(input).view(1, 1, -1)  # 展开
        # output: [1, 1, emb_dim]
        output = F.relu(output)
        # output: [1, 1, emb_dim]

        # 关于gru的输入输出参数
        # [seq_len, batch_size, input_size],  [num_layers * num_directions, batch_size, hidden_size]
        # output: [1, 1, emb_dim], hidden: [1, 1, hidden_size]
        output, hidden = self.gru(output, hidden)
        # output: [1, 1, hidden_size] # [seq_len, batch, num_directions * hidden_size] # 这里hidden_size == emb_dim
        # output[0]: [1, emb_dim]
        # self.out(output[0]): [1, V]
        # output: [1, V] 值为每个单词的概率
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


if __name__ == '__main__':
    from torchsummaryX import summary
    hidden_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.ones((1, 1), dtype=torch.long).to(device)
    hiddens = torch.ones(1, 1, hidden_size).to(device)
    print("encoder+++++++++++++++++++++++++++++++++++++++")
    encoder = EncoderGRU(input_size=300, hidden_size=128).to(device)
    print(encoder)
    summary(encoder, inputs, hiddens)
    print("decoder+++++++++++++++++++++++++++++++++++++++")
    decoder = DecoderGRU(hidden_size=128, output_size=256).to(device)
    print(decoder)
    summary(decoder, inputs, hiddens)
    
    # 如果仅仅想在显卡上使用上也可以对model和inputs添加.cuda(), 使用cpu则不需要.cuda()
    # inputs = torch.ones((1, 1), dtype=torch.long).cuda()
    # hiddens = torch.ones(1, 1, hidden_size).cuda()
    # print("encoder+++++++++++++++++++++++++++++++++++++++")
    # encoder = EncoderGRU(input_size=300, hidden_size=128).cuda()
    # print(encoder)
    # summary(encoder, inputs, hiddens)
    # print("decoder+++++++++++++++++++++++++++++++++++++++")
    # decoder = DecoderGRU(hidden_size=128, output_size=256).cuda()
    # print(decoder)
    # summary(decoder, inputs, hiddens)