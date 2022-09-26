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

        self.gru1 = Bidirectional_GRU(input_size=544, hidden_size=128)
        self.gru2 = Bidirectional_GRU(input_size=256, hidden_size=128)
        self.fc1 = nn.LazyLinear(out_features=128)
        self.fc2 = nn.LazyLinear(out_features=num_classes)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        # print(x.size())

        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.gru2(self.gru1(x))
        x = self.fc2(self.fc1(x))

        return x


if __name__ == '__main__':
    model = DeepSpeech2(num_classes=54)
    x = torch.rand(1, 200, 128)
    model(x)
