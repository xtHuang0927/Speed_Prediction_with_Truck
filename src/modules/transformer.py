# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]



class Transformer(nn.Module):
    def __init__(self, feature_size=10, num_layers=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        # self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        # self.pos_encoder = torch.nn.Embedding(512,256)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_layers, dropout=dropout,
                                                        dim_feedforward=7*feature_size)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 5))  # Size target length
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)  # , self.src_mask) , self.src_mask
        # print(output.shape)x
        output = self.decoder(output)
        output = self.maxpool(output)
        return output

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

