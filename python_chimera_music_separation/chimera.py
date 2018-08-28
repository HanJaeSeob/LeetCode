import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


def l2_normalize(x, dim=0, eps=1e-12):
    assert (dim < x.dim())
    norm = torch.norm(x, 2, dim, keepdim=True)
    return x / (norm + eps)


class chimeraNet(nn.Module):
    def __init__(self, num_bins, rnn="lstm",embedding_dim=20,
                 num_layers=2, hidden_size=300, dropout=0.0,
                 non_linear="tanh", bidirectional=True):
        super(chimeraNet, self).__init__()
        self.rnn = nn.LSTM(num_bins, hidden_size, num_layers, batch_first=True,
                           dropout=dropout, bidirectional=bidirectional)
        # self.drops = nn.Dropout(p=dropout)
        self.embed = nn.Linear(hidden_size*2, num_bins*embedding_dim)

        self.IRM = nn.Linear(hidden_size*2, num_bins*2)

        self.embedding_dim = embedding_dim

    def forward(self, x):  # x (128,100,257)
        N = x.size(0)
        x, _ = self.rnn(x)
        # x = self.drops(x)
        y = self.embed(x)
        y = F.tanh(y)
        y = y.view(N, -1, self.embedding_dim)  # x (128,100*257,20)
        y = F.normalize(y, p=2, dim=-1, eps=1e-12)

        z = self.IRM(x)  # y (128,100,257*2)
        z = z.view(N, -1, 2)  # x (128,100*257,2)
        # x = l2_normalize(x, -1)
        z = F.softmax(z, dim=-1)
        return y, z


