import torch.nn as nn
import torch


class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                              padding=kernel // 2)
        self.cnn2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                              padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual

        return x


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, batch_first=True):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=batch_first,
                          bidirectional=True)
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.gelu(x)
        x, _ = self.gru(x)
        x = self.dropout(x)
        return x


class DeepSpeech2(nn.Module):
    def __init__(self, dropout, n_feats, rnn_dim, num_classes):
        super(DeepSpeech2, self).__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=3 // 2)
        self.res_cnn1 = ResidualCNN(in_channels=32, out_channels=32, kernel=3, stride=1, dropout=dropout,
                                    n_feats=n_feats)

        self.res_cnn2 = ResidualCNN(in_channels=32, out_channels=32, kernel=3, stride=1, dropout=dropout,
                                    n_feats=n_feats)
        self.gelu = nn.GELU()

        self.fc = nn.Linear(n_feats * 32, rnn_dim)
        self.bigru1 = BiGRU(input_size=rnn_dim, hidden_size=rnn_dim, dropout=dropout)
        self.bigru2 = BiGRU(input_size=rnn_dim * 2, hidden_size=rnn_dim, dropout=dropout)

        self.classifier = nn.Sequential(nn.Linear(rnn_dim * 2, rnn_dim), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(rnn_dim, num_classes))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.res_cnn1(x)
        x = self.res_cnn2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.gelu(self.fc(x))
        x = self.bigru1(x)
        x = self.bigru2(x)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    x = torch.rand(1, 1, 128, 141)
    model = DeepSpeech2(dropout=0.2,n_feats=128,rnn_dim=256,num_classes=54)
    print(model(x).size())
