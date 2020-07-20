import numpy as np

count_len = {}


def normalize(sens, max_len):
    temp = []
    for sen in sens:

        if len(sen) > max_len:
            sen = sen[:max_len]

        elif len(sen) < max_len:
            sen = [0] * (max_len - len(sen)) + sen

        temp.append(sen)
    return temp


def load_vocab(path):
    with open(path) as f:
        data = f.read().splitlines()
    data = [line.split() for line in data]
    words = [word for word, count in data if int(count) > 1]
    vocab = {word: index for index, word in enumerate(words, start=2)}
    vocab.update({'<pad>': 0, '<unk>': 1})
    return vocab


def load_glove(path, choose_vocab):

    with open(path['glove']) as f:
        datas = f.read().splitlines()
    datas = [line.split() for line in datas]

    if choose_vocab == 'vocab':
        vocab = load_vocab(path['vocab'])
    else:
        words = [data[0] for data in datas]
        vocab = {word: index for index, word in enumerate(words, start=2)}
        vocab.update({'<pad>': 0, '<unk>': 1})

    glove = {data[0]: np.array([float(i) for i in data[1:]]) for data in datas}
    glove.update({'<pad>': np.zeros(300), '<unk>': np.ones(300)})

    return vocab, glove


def load_sents(path, max_len, vocab):
    sens = []
    with open(path) as f:
        for line in f:
            sen = []
            sentence = line.strip().split()
            for word in sentence:
                if word in vocab.keys():
                    sen.append(vocab[word])
                else:
                    sen.append(vocab['<unk>'])
            if len(sen) in count_len.keys():
                count_len[len(sen)] += 1
            else:
                count_len[len(sen)] = 1

            sens.append(sen)

    sens = normalize(sens, max_len)

    # count every sentence len, found the longest sentence
    # print(count_len)
    return sens


def load_label(path):
    with open(path) as f:
        label = [int(line) for line in f]

    return label


def make_pair(x, y):

    pairs = []
    for i in range(len(x)):
        tempx = np.asarray(x[i])
        tempy = np.asarray(y[i])
        pairs.append([tempx, tempy])

    pairs = np.array(pairs)
    return pairs


def load_data(cfg):

    glove = None
    if cfg.pretrain:
        vocab, glove = load_glove(cfg.path, cfg.choose_vocab)
        cfg.vocab_size = len(vocab)

    else:
        vocab = load_vocab(cfg.path['vocab'])

    train_X = load_sents(cfg.path['train_X'], cfg.max_len, vocab)
    test_X = load_sents(cfg.path['test_X'], cfg.max_len, vocab)
    train_Y = load_label(cfg.path['train_Y'])
    test_Y = load_label(cfg.path['test_Y'])

    train_set = make_pair(train_X, train_Y)
    test_set = make_pair(test_X, test_Y)

    return train_set, test_set, vocab, glove
