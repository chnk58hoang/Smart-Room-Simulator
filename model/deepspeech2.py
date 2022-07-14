import torch
import torch.nn as nn


class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CNN_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))


class Bidirectional_GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Bidirectional_GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x, _ = self.gru(x)
        return self.dropout(x)


class DeepSpeech2(nn.Module):
    def __init__(self, num_classes):
        super(DeepSpeech2, self).__init__()
        self.cnn1 = CNN_Block(1, 32, (11, 41), (2, 2))
        self.cnn2 = CNN_Block(32, 32, (11, 21), (1, 2))
        self.fc1 = nn.Linear(1568, 128)
        self.gru1 = Bidirectional_GRU(input_size=128, hidden_size=128)
        self.gru2 = Bidirectional_GRU(input_size=256, hidden_size=128)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.ctc_loss = nn.CTCLoss(blank=0)

    def forward(self, x, target=None, target_length=None):
        x = self.cnn1(x)
        x = self.cnn2(x)

        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)

        x = self.relu(self.fc1(x))
        x = self.gru2(self.gru1(x))
        x = self.fc2(x)

        if target != None and target_length != None:
            x = self.log_softmax(x)
            x = x.permute(1, 0, 2)
            input_length = torch.full(size=(x.size(1),), fill_value=x.size(0), dtype=torch.long)
            loss = self.ctc_loss(x, target, input_length, target_length)
            x = x.permute(1, 0, 2)
            return x, loss

        return self.softmax(x), None
