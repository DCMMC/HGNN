from torch import nn
from models import HGNN_conv
import torch.nn.functional as F


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5, layers=2):
        super(HGNN, self).__init__()
        self.dropout = dropout
        hgc_list = []
        self.layers = layers
        for layer in range(layers):
            if layer == 0:
                hgc_list.append(HGNN_conv(in_ch, n_hid))
            elif layer == layers - 1:
                hgc_list.append(HGNN_conv(n_hid, n_class))
            else:
                hgc_list.append(HGNN_conv(n_hid, n_hid))
        self.hgc_list = nn.ModuleList(hgc_list)

    def forward(self, x, G):
        x = self.hgc_list[0](x, G)
        for layer in range(1, self.layers):
            x = F.relu(x)
            # (DCMMC) dropout 放在哪里需要进一步明确
            if layer in [1, self.layers - 1]:
                x = F.dropout(x, self.dropout)
            x = self.hgc_list[layer](x, G)
        return x
