# NLP-Demo

## 0 引言

利用Pytorch实现了MLP、CNN、RNN三种神经网络，并且实现了一个简单的RNN网络。分别在老师给定的vocab上对数据集进行训练，并载入预训练好的词向量Glove(需要自行下载)作为对照试验。

## 1 代码结构说明

本次实现的网络结构较多、数据集处理较复杂、设计对照试验因此代码结构较为复杂，代码结构包含3个文件夹和5个python文件，如下：

`Data`文件夹中存放数据集，包括`train_X.txt`、`train_Y.txt`、`test_X.txt`、`test_Y.txt`、`vocab.txt`五个文件，分别是训练集文本数据、训练集标签、测试集文本数据、测试集标签和词典。

`GloVe`文件夹中存放下载的预训练好的词向量`glove.6B.300d.txt`。

`runs`文件夹中存放tensorboard文件，用于实时查看训练效果。

`main.py`是代码的主干，包括可选参数、加载数据集到GPU、模型选择、模型训练、模型测试、结果展示几个部分。

`config.py`是配置文件，保存着文件路径、模型超参数等静态信息。

`dataset.py`中构建了一个Dataset class，用于加载数据集。

`models.py`保存MLP、CNN、RNN、myRNN四个网络模型。

`utils.py`是数据预处理部分，包括训练集、测试集、vocab、glove的处理。

