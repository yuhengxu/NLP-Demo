import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def create_weight_matrix(vocab, glove):

    weights_matrix = np.zeros((len(vocab), 300))
    for key in vocab.keys():
        if key in glove.keys():
            weights_matrix[vocab[key]] = glove[key]
        else:
            # weights_matrix[vocab[key]] = np.random.normal(scale=0.6, size=300)
            weights_matrix[vocab[key]] = np.ones(300)

    weights_matrix = torch.from_numpy(weights_matrix)
    return weights_matrix


class MLP(nn.Module):

    def __init__(self, cfg, vocab, glove):
        super(MLP, self).__init__()
        self.glove = glove
        self.vocab = vocab

        self.embed = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        if glove is not None:
            self.weight_matrix = create_weight_matrix(self.vocab, self.glove)
            self.embed.load_state_dict({'weight': self.weight_matrix})

        self.fc1 = nn.Linear(cfg.embedding_dim, cfg.hidden_size)
        self.fc2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.fc3 = nn.Linear(cfg.hidden_size, 1)

    def forward(self, x):
        # x: batch_size * seq_len
        e = self.embed(x)  # batch_size * seq_len * hidden_size
        h = e.mean(dim=1)  # batch_size * hidden_size
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h.squeeze(1)


class RNN(nn.Module):

    def __init__(self, cfg, vocab, glove):
        super(RNN, self).__init__()
        self.glove = glove
        self.vocab = vocab

        self.embed = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        if glove is not None:
            self.weight_matrix = create_weight_matrix(self.vocab, self.glove)
            self.embed.load_state_dict({'weight': self.weight_matrix})
        self.rnn = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.fc1 = nn.Linear(cfg.hidden_size * 2, 500)
        self.fc2 = nn.Linear(500, 300)
        self.out = nn.Linear(300, 1)

    def forward(self, x):
        # x: batch_size * seq_len
        e = self.embed(x)   # e: batch_size * seq_len * hidden_size
        r, (h_n, c_n) = self.rnn(e)
        x = self.fc1(r).mean(dim=1)
        x = self.fc2(x)
        output = self.out(x).squeeze()
        return output


class myRNN(nn.Module):

    def __init__(self, cfg, vocab, glove):
        super(myRNN, self).__init__()
        # self.batch_size = cfg.batch_size
        self.seq_len = cfg.max_len
        self.hidden_size = cfg.hidden_size
        self.glove = glove
        self.vocab = vocab

        self.embed = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        if glove is not None:
            self.weight_matrix = create_weight_matrix(self.vocab, self.glove)
            self.embed.load_state_dict({'weight': self.weight_matrix})

        self.in2hidden = nn.Linear(cfg.embedding_dim+cfg.hidden_size, cfg.hidden_size)
        self.tanh = nn.Tanh()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size * 2),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size * 2, 1),
        )

    def forward(self, x):
        e = self.embed(x)
        self.batch_size = e.shape[0]
        a = Variable(torch.zeros(self.batch_size, self.seq_len, self.hidden_size)).cuda()
        pre_y = Variable(torch.zeros(self.batch_size, self.seq_len, self.hidden_size)).cuda()
        pre_state = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()

        for t in range(self.seq_len):
            tmp = torch.cat((e[:, t, :], pre_state), 1)
            a[:, t, :] = self.in2hidden(tmp)
            h = self.tanh(a[:, t, :])
            pre_state = h
            pre_y[:, t, :] = h

        output = self.mlp(pre_y[:, -1, :])
        return output.squeeze()


class CNN(nn.Module):

    def __init__(self, cfg, vocab, glove):
        super(CNN, self).__init__()
        self.glove = glove
        self.vocab = vocab

        self.embed = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        if glove is not None:
            self.weight_matrix = create_weight_matrix(self.vocab, self.glove)
            self.embed.load_state_dict({'weight': self.weight_matrix})
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=cfg.n_kernel, kernel_size=(cfg.filter_sizes[0], cfg.embedding_dim))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=cfg.n_kernel, kernel_size=(cfg.filter_sizes[1], cfg.embedding_dim))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=cfg.n_kernel, kernel_size=(cfg.filter_sizes[2], cfg.embedding_dim))
        self.fc0 = nn.Linear(len(cfg.filter_sizes)*cfg.n_kernel, 500)
        self.fc1 = nn.Linear(500, 1)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.embed(x).unsqueeze(1)

        conv0 = F.relu(self.conv0(x).squeeze(3))
        conv1 = F.relu(self.conv1(x).squeeze(3))
        conv2 = F.relu(self.conv2(x).squeeze(3))

        pool0 = F.max_pool1d(conv0, conv0.shape[2]).squeeze(2)
        pool1 = F.max_pool1d(conv1, conv1.shape[2]).squeeze(2)
        pool2 = F.max_pool1d(conv2, conv2.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pool0, pool1, pool2), dim=1))

        cat = self.fc0(cat)

        return self.fc1(cat).squeeze()
