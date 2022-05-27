import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

from dyrelu import DyReLUB


class RNN(nn.Module):
    def __init__(self, num_layers=1, mid_features=1024,
                 if_bidirectional=False, Center=True, Gender=True):
        super(RNN, self).__init__()

        '''
        Epoch = 30
        BatchSize = 32
        BatchSize = 16
        TimeStep = 28
        InputSize = 28
        LR = 0.01

        torch.Size([32, 60, 251])
        input_x = torch.Size([32, 251, 60])
        '''
        self.InputSize = 3*60
        self.Center = Center
        self.Gender = Gender

        self.mid_features = mid_features
        self.num_layers = num_layers
        self.if_bidirectional = if_bidirectional

        if self.if_bidirectional:
            self.in_features = self.mid_features * 2
        else:
            self.in_features = self.mid_features

        # self.norm = nn.LayerNorm(self.InputSize)

        self.lstm = nn.LSTM(
            input_size=self.InputSize, hidden_size=self.mid_features, num_layers=self.num_layers,
            batch_first=True, bidirectional=self.if_bidirectional
        )

        '''
        self.l1 = nn.Linear(in_features=self.in_features, out_features=self.in_features)#特征输入
        self.l3 = nn.BatchNorm1d(self.in_features)
        '''
        self.prelu = nn.PReLU()
        self.out = nn.Linear(in_features=self.in_features, out_features=4)

        self.integral_fc_emotion = nn.Linear(in_features=self.in_features, out_features=4)
        if self.Center:
            self.integral_fc_emotion_center = nn.Linear(in_features=self.in_features, out_features=4)
        if self.Gender:
            self.integral_fc_sex = nn.Linear(in_features=self.in_features, out_features=2)
        if self.Center and self.Gender:
            self.integral_fc_sex_center = nn.Linear(in_features=self.in_features, out_features=2)

    def forward(self, x):
        # x1 = x
        # x1 = self.norm(x1.reshape(-1, x1.shape[1]))
        # input_x = x1.view(x.shape[0], x.shape[1], x.shape[2]).permute(0, 2, 1)
        input_x = x.permute(0, 2, 1)
        out_x, (h_n, h_c) = self.lstm(input_x, None)  # (batch, time_step, input_size)

        out_x = self.prelu(out_x[:, -1, :])

        # out_x = self.tan(out_x[:,-1,:])
        #out_x = out_x[:,-1,:]
        # out_x = self.dropout(out_x)
        # print('out_x: ', out_x.shape)
        out_emotion = self.integral_fc_emotion(out_x)
        # print('out: ', out_emotion.shape)
        if self.Gender and self.Center:
            out_sex = self.integral_fc_sex(out_x)
            out_sex_center = self.integral_fc_sex_center(out_x)
            out_emotion_center = self.integral_fc_emotion_center(out_x)
            return out_x, out_emotion, out_sex, out_emotion_center, out_sex_center
        if self.Gender:
            out_sex = self.integral_fc_sex(out_x)
            return out_x, out_emotion, out_sex
        if self.Center:
            out_emotion_center = self.integral_fc_emotion_center(out_x)
            return out_x, out_emotion, out_emotion_center
        else:
            return out_x, out_emotion

    #