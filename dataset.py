import torch


class MotionDataset(torch.utils.data.Dataset):

    def __init__(self, sentence):
        self.sentence = sentence

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        return self.sentence[index][0], self.sentence[index][1]
