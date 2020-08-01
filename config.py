import torch


class Config(object):

    def __init__(self, args):
        assert args.model in ['MLP', 'RNN', 'CNN', 'myRNN']
        assert args.choose_vocab in ['vocab', 'glove']
        self.model = args.model
        self.choose_vocab = args.choose_vocab
        self.pretrain = args.pretrain

        self.path = {
            'train_X': './Data/train_X.txt',
            'train_Y': './Data/train_Y.txt',
            'test_X': './Data/test_X.txt',
            'test_Y': './Data/test_Y.txt',
            'vocab': './Data/vocab.txt',
            'glove': './GloVe/glove.6B.300d.txt'
        }

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.max_len = 31
        self.vocab_size = 25467
        self.embedding_dim = 300

        if self.model == 'MLP':
            self.epoch = 1000
            self.batch_size = 100000
            self.learn_rate = 1e-4
            self.hidden_size = 500
        elif self.model == 'RNN' or self.model == 'myRNN':
            self.epoch = 20
            self.batch_size = 4096
            self.learn_rate = 1e-3
            self.hidden_size = 200
            self.num_layers = 2
            self.dropout = 0.5
        elif self.model == 'CNN':
            self.epoch = 100
            self.batch_size = 32768
            self.learn_rate = 1e-3
            self.dropout = 0.25
            self.n_kernel = 128
            self.filter_sizes = [3, 4, 5]
