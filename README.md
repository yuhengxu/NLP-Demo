# NLP-Demo

## 0 引言

利用Pytorch实现了MLP、CNN、RNN三种神经网络，并且实现了一个简单的RNN网络。分别在老师给定的vocab上对数据集进行训练，并载入预训练好的词向量Glove(需要自行下载)作为对照试验。

## 1 代码结构说明

本次实现的网络结构较多、数据集处理较复杂、设计对照试验因此代码结构较为复杂，代码结构包含3个文件夹和5个python文件，如下：

![代码结构](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\代码结构.png)

`Data`文件夹中存放数据集，包括`train_X.txt`、`train_Y.txt`、`test_X.txt`、`test_Y.txt`、`vocab.txt`五个文件，分别是训练集文本数据、训练集标签、测试集文本数据、测试集标签和词典。

`GloVe`文件夹中存放下载的预训练好的词向量`glove.6B.300d.txt`。

`runs`文件夹中存放tensorboard文件，用于实时查看训练效果。

`main.py`是代码的主干，包括可选参数、加载数据集到GPU、模型选择、模型训练、模型测试、结果展示几个部分。

`config.py`是配置文件，保存着文件路径、模型超参数等静态信息。

`dataset.py`中构建了一个Dataset class，用于加载数据集。

`models.py`保存MLP、CNN、RNN、myRNN四个网络模型。

`utils.py`是数据预处理部分，包括训练集、测试集、vocab、glove的处理。

## 2 前期配置

代码中用到了需要额外下载配置的库和文件。

### 2.1 tensorboard

`tensorboard`在模型训练过程中经常用于实时查看模型训练效果，能很好的反应出训练效果。

####2.1.1 install

```
pip install tensorboard
```

####2.1.2 run

在terminal中输入

```
tensorboard --logdir=runs --port=3389
```

在浏览器中输入网址http://localhost:3389/即可查看训练效果

####2.1.3 files introduce

在runs文件目录中已经保存了12个训练好的模型效果共查看

`MLP_with_glove_vocab`：代表采用MLP模型，使用预训练词向量glove与老师提供的vocab求交集作为词典，不在词典内的词均用<unk>表示。

`MLP_with_vocab`：代表采用MLP模型，仅使用老师提供的vocab作为词典，不在词典内的词均用<unk>表示。

`MLP_with_glove`：代表采用MLP模型，仅使用预训练词向量glove作为词典，不在词典内的词均用<unk>表示。

RNN、CNN、myRNN开头的文件以此类推

### 2.2 GLove

预训练词向量glove从[官网]( https://nlp.stanford.edu/projects/glove/ )下载，选用Wikipedia 2014，822MB压缩文件，选择其中将一个词表示为300维的词典`glove.6B.300d.txt`，若使用其他的词典，可以自行下载，并对代码相关部分进行修改。

## 3 可选参数＆超参数

###3.1 可选参数

在运行时设置了模型选择、词典选择、是否进行预训练三个可选参数，代码在main.py的主函数中，如下：

```python
parser = argparse.ArgumentParser(description='manual to this model')
parser.add_argument('--model', type=str, default='myRNN')
parser.add_argument('--choose_vocab', type=str, default='vocab')
parser.add_argument('--pretrain', type=bool, default=False)
args = parser.parse_args()
```

运行代码示例如下：

```python
python main.py
```

会调用默认参数训练，即：采用RNN模型、使用vocab作为词典、不进行预训练。需要注意的是，选择字典仅在进行预训练的情况下才会生效。`--pretrain=False`时，字典为老师给定的字典。

当然，也可以根据需要调整可选参数，比如采用CNN模型，使用glove作为字典，进行预训练。运行示例代码如下：

```python
python main.py --model=CNN --pretrain=True --choose_vocab=glove
```

###3.2 超参数

针对不同的网络模型，learning_rate、dropout、epoch、batch_size等超参数的设置也有不同。因此在选择对应模型后，会加载不同的超参数用于训练。代码在config.py的Config类中。如下：

```python
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
```

超参数命名遵信常规命名，不在赘述具体含义。调用代码在main.py中，代码如下：

```python
from config import Config
cfg = Config(args)
```

## 4 数据预处理

数据预处理较为复杂。包括对训练数据、标签、字典的预处理。实现代码在utils.py中。

### 4.1 字典

#### 4.1.1 vocab

老师给的字典共两列，第一列是单词，第二列是该词出现的次数，为了能更好的学到参数，我在处理时将出现次数少于2次的单词全部用<unk>代替。并将数据整理为{'word': index}的格式。代码如下：

```python
def load_vocab(path):
    with open(path) as f:
        data = f.read().splitlines()
    data = [line.split() for line in data]
    words = [word for word, count in data if int(count) > 1]
    vocab = {word: index for index, word in enumerate(words, start=2)}
    vocab.update({'<pad>': 0, '<unk>': 1})
    return vocab
```

#### 4.1.2 glove

glove每一行共301个项，第一项是单词，后面300项为表示该次的词向量。需要将数据整理成{‘word’: [tensor]}的形式。在`load_glove()`函数中除了转换glove格式外，还要生成同4.1.1中格式相同的vocab，因此还包含选择用老师给的词典，还是用glove作为词典。

```python
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
```

### 4.2 数据

#### 4.2.1 数据加载

`train_X.txt`、`test_X.txt`中每一行是一个完整的句子，因此要将句子切分成单词保存，实际训练中，单词不能作为输入，要改成索引输入，因此要对照4.1中生成的vocab，生成句子的索引序列。代码如下：

```python
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
```

其中count_len是用来统计文本长度的一个dict，通过观察发现，绝大多数句子的长度在31个单词以内，因此将31作为合适的序列长度。

####4.2.2 输入长度整理

 上述代码中count_len是用来统计文本长度的一个dict，通过观察发现，绝大多数句子的长度在31个单词以内，因此将31作为合适的序列长度对过长的文本进行裁剪，对过短的文本进行补齐 。补齐代码如下：

```python
def normalize(sens, max_len):
    temp = []
    for sen in sens:

        if len(sen) > max_len:
            sen = sen[:max_len]

        elif len(sen) < max_len:
            sen = [0] * (max_len - len(sen)) + sen

        temp.append(sen)
    return temp
```

### 4.3 标签

`train_Y.txt`、`test_Y.txt`中每一行只有一个元素，为对应的情感标签，因此按行读入，转换成浮点数即可，代码如下：

```python
def load_label(path):
    with open(path) as f:
        label = [int(line) for line in f]

    return label
```

### 4.4 配对

为了在接下来的方便调用pytorch的DataLoader，需要对数据和标签进行一对一捆绑，形成train_set和test_set，数据格式为[x, y]，代码如下：

```python
def make_pair(x, y):

    pairs = []
    for i in range(len(x)):
        tempx = np.asarray(x[i])
        tempy = np.asarray(y[i])
        pairs.append([tempx, tempy])

    pairs = np.array(pairs)
    return pairs
```

### 4.5 封装

将上述部件封装起来，方便调用，其中在选择字典时需要判断，代码如下：

```python
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
```

main.py调用方式如下：

```python
from utils import load_data
train_set, test_set, vocab, glove = load_data(cfg)
```

## 5 模型构建

模型均保存在models.py文件中，共四个模型。

### 5.1 MLP

多层感知机由一个embeding层和三个线性层构成。每层的大小维度见注释，代码如下：

```python
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
```

### 5.2 RNN

我选用的pytorch自带的LSTM作为RNN层，双层RNN，隐层维度为200，双向结构，代码如下：

```python
self.rnn = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=True,
            batch_first=True,
        )
```

以及全连接层双隐层，神经元数分别为500、300，代码如下：

```python
self.fc1 = nn.Linear(cfg.hidden_size * 2, 500)
self.fc2 = nn.Linear(500, 300)
self.out = nn.Linear(300, 1)
```

### 5.3 myRNN

自己实现的RNN较为简单，每次输入一句话，将一句话的每个单词放入RNN中更新参数，核心代码如下：

```python
for t in range(self.seq_len):
   tmp = torch.cat((e[:, t, :], pre_state), 1)
   a[:, t, :] = self.in2hidden(tmp)
   h = self.tanh(a[:, t, :])
   pre_state = h
   pre_y[:, t, :] = h
```

后面紧跟一个MLP，结构代码如下：

```python
self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size * 2),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size * 2, 1),
        )
```

### 5.4 CNN

CNN较为特殊，我采用的是pytorch中的`nn.Conv2d`函数，这样会从单词方向和embeding两个方向进行卷积，但其实embeding方向的卷积是没有意义的，后来得知pytorch中竟然已经提供了`nn.Conv1d`函数，但是效果对比并没有明显差别，因此没有进行修改。

CNN模型包括三个卷积层，每层包含128个卷积核，高度分别为3,4,5，全连接层单层隐层，神经元个数为500，采用max_pool1d即可。代码较长，不做展示，详见models.py中的CNN类。

### 5.5 加载Glove

为了在embeding是加入GLove，需要自己写load_glove函数，然后再调用embeding层时作为weight调用，生成weight矩阵代码如下：

```python
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
```

对于vocab中不存在的单词，统一用300维的全1矩阵代替。（注释掉的代码是用随机生成300维矩阵代替<unk>，实际效果没有差别）

在embeding时调用该矩阵方式如下：

```python
self.embed = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
if glove is not None:
     self.weight_matrix = create_weight_matrix(self.vocab, self.glove)
     self.embed.load_state_dict({'weight': self.weight_matrix})
```

## 6 模型训练

###6.1 数据加载

我采用pytorch自带的Dataloader进行数据加载，首先需要自己构建一个继承`torch.utils.data.Dataset`的类

```python
class MotionDataset(torch.utils.data.Dataset):
```

需要写明每个数据的大小

```python
def __len__(self):
    return len(self.sentence)
```

并写清如何提取数据

```python
def __getitem__(self, index):
    return self.sentence[index][0], self.sentence[index][1]
```

完整代码在dataset.py中，封装好后在main.py中调用，得到train_loader和test_loader，代码如下：

```python
from dataset import MotionDataset

train_loader = torch.utils.data.DataLoader(
        MotionDataset(train_set),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        MotionDataset(test_set),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
```

### 6.2 选择模型

在load数据后要加载模型，并将模型放到GPU上，代码如下：

```python
if cfg.model == 'MLP':
    model = MLP(cfg, vocab, glove).train().to(cfg.device)
elif cfg.model == 'RNN':
    model = RNN(cfg, vocab, glove).train().to(cfg.device)
elif cfg.model == 'myRNN':
    model = myRNN(cfg, vocab, glove).train().to(cfg.device)
elif cfg.model == 'CNN':
    model = CNN(cfg, vocab, glove).train().to(cfg.device)
```

### 6.3 激活函数

选用Adam作为激活函数，代码如下：

```python
optimizer = optim.Adam(model.parameters(), lr=cfg.learn_rate)
```

### 6.4 训练

首先将数据放到GPU上，再将数据送入模型，进一步计算loss，采用binary_cross_entropy_with_logits作为loss函数，该函数在二分类中表现优秀，且内置了sigmoid函数，因此在模型中不需要额外添加sigmoid层，剩下的就是神经网络的八股文，代码如下：

```python
data, label = data.to(cfg.device), label.to(cfg.device)
logits = model(data)
loss = F.binary_cross_entropy_with_logits(
    input=logits,
    target=label.double(),
    reduction='mean',
)
model.zero_grad()
loss.backward()
optimizer.step()
```

该部分还有很多额外的代码并没有解释，均为计算准确率，查看模型效果的辅助代码，详情可查看`tqdm`、`tensorboard`、`sklearn.metrics`文档查看详细的使用方法。

### 6.5 测试

测试部分代码与训练代码大同小异，不再赘述。

## 7 试验结果

 实验结果并不好，MLP训练1000轮，CNN训练100轮，RNN训练20轮，总体准确率在73%~75%之间

###7.1 precision、recall、F1

MLP的各项数据如下；

![MLP_train](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\MLP_train.png)

![MLP_test](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\MLP_test.png)

CNN各项数据如下：

![CNN_train](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\CNN_train.png)

![CNN_test](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\CNN_test.png)

RNN各项数据如下：

![RNN_train](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\RNN_train.png)

![RNN_test](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\RNN_test.png)

myRNN各项数据如下：

### 7.2 对比试验

####7.2.1 仅使用老师提供的vocab

即展示`MLP_with_vocab`、`RNN_with_vocab`、`CNN_with_vocab`、`myRNN_with_vocab`

全景展示：

![vocab](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\vocab.png)

由于MLP训练过多，仅取前100轮，结果展示：

![vocab_big](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\vocab_big.png)

在同一epoch时，各个模型准确率展示：

![vocab_data](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\vocab_data.png)

####7.2.1 仅使用glove

即展示`MLP_with_glove`、`CNN_with_glove`、`RNN_with_glove`、`myRNN_with_glove`

全景展示：

![glove](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\glove.png)

由于MLP训练过多，仅取前100轮，结果展示：

![glove_big](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\glove_big.png)

在同一epoch时，各个模型准确率展示：

![glove_data](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\glove_data.png)

###7.2.3 使用govle和vocab的交集

即展示`MLP_with_glove_vocab`、`CNN_with_glove_vocab`、`RNN_with_glove_vocab`、`myRNN_with_glove_vocab`

全景展示：

![vocab_glove](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\vocab_glove.png)

由于MLP训练过多，仅取前100轮，结果展示：

![vocab_glove_big](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\vocab_glove_big.png)

在同一epoch时，各个模型准确率展示：

![vocab_glove_data](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\vocab_glove_data.png)

#### 7.2.4 同一模型在不同词典中的表现

此处仅展示RNN模型的对比效果，即展示`RNN_with_glove_vocab`、`RNN_with_vocab`、`RNN_with_glove`若有兴趣可自行用tensorboard查看其他的对比效果.

![RNN](C:\Users\Yuhen\OneDrive - bit.edu.cn\Postgraduate\研一下\高级数据科学\Homework1\DY1906101_徐宇恒_Homework2\doc\RNN.png)

## 8 总结

本次实验较为复杂，涉及的知识较多，花了将近两周才完成了老师要求的包含两个附加目标在内的全部内容。对于代码能力有加强，对pytorch的使用有了更多的体会。对于实验不好的结果，也跟同学做了交流，发现大家的结果都不好，说明不是我调参的问题。对于数据集实在是无力吐槽，找搞NLP的大佬帮忙用bert跑了模型，准确率也才将将80%，说明数据集要么分布不均匀，要么就是脏数据太多。