import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

from dyrelu import DyReLUB


class RC_mdp_AFF(nn.Module):
    def __init__(self, DyRELU=True, Center=True, Gender=True, if_bidirectional=False, inter_channels=32):
        super(RC_mdp_AFF, self).__init__()

        # ------------------CNN-DEFINE------------------#
        self.DyRELU = DyRELU
        self.Center = Center
        self.Gender = Gender

        self.in_features = 64+128
        self.conv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=8)
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=8, out_channels=32, padding=1)
        # self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=96, padding=1)
        self.conv6 = nn.Conv2d(kernel_size=(3, 3), in_channels=96, out_channels=128, padding=1)
        # self.conv7 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=512, padding=1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(96)
        self.bn6 = nn.BatchNorm2d(128)
        # self.bn7 = nn.BatchNorm2d(512)

        if self.DyRELU:
            self.relu1 = DyReLUB(8, conv_type='2d')
            self.relu2 = DyReLUB(32, conv_type='2d')
            # self.relu3 = DyReLUB(32, conv_type='2d')
            self.relu4 = DyReLUB(64, conv_type='2d')
            self.relu5 = DyReLUB(96, conv_type='2d')
            self.relu6 = DyReLUB(128, conv_type='2d')
            # self.relu7 = DyReLUB(512, conv_type='2d')
        else:
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            self.relu5 = nn.ReLU()
            self.relu6 = nn.ReLU()

        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ------------------LSTM-DEFINE------------------#
        self.mid_features = 1024
        self.num_layers = 1
        self.if_bidirectional = if_bidirectional
        if self.if_bidirectional:
            self.mid_features = int(self.mid_features / 2)

        # self.norm = nn.LayerNorm(3*60)

        # self.linear = nn.Linear(in_features=3*60, out_features=3*60)

        self.lstm = nn.LSTM(
            input_size=3*60, hidden_size=self.mid_features, num_layers=self.num_layers,
            batch_first=True, bidirectional=self.if_bidirectional
        )

        # self.prelu = nn.PReLU()

        # ------------------CONCAT-DEFINE------------------#


        # ------------------CLASSIFY-DEFINE------------------#
        self.integral_fc_emotion = nn.Linear(in_features=self.in_features, out_features=4)
        if self.Center:
            self.integral_fc_emotion_center = nn.Linear(in_features=self.in_features, out_features=4)
        if self.Gender:
            self.integral_fc_sex = nn.Linear(in_features=self.in_features, out_features=2)
        if self.Center and self.Gender:
            self.integral_fc_sex_center = nn.Linear(in_features=self.in_features, out_features=2)

    def forward(self, *input):
        # ------------------CNN-FORWARD------------------#
        input_cnn = input[0].unsqueeze(1)
        print('input_cnn: ', input_cnn.shape)
        x1_cnn = self.conv1(input_cnn)
        x1_cnn = self.bn1(x1_cnn)
        x1_cnn = self.relu1(x1_cnn)

        x2_cnn = self.conv2(x1_cnn)
        x2_cnn = self.bn2(x2_cnn)

        x2_cnn = self.relu2(x2_cnn)

        # x3_cnn = self.maxp(x2_cnn)
        # x3_cnn = self.conv3(x3_cnn)
        # x3_cnn = self.bn3(x3_cnn)
        #
        # x3_cnn = self.relu3(x3_cnn)

        x4_cnn = self.maxp(x2_cnn)
        x4_cnn = self.conv4(x4_cnn)
        x4_cnn = self.bn4(x4_cnn)

        x4_cnn = self.relu4(x4_cnn)

        x5_cnn = self.maxp(x4_cnn)
        x5_cnn = self.conv5(x5_cnn)
        x5_cnn = self.bn5(x5_cnn)

        x5_cnn = self.relu5(x5_cnn)

        #
        x6_cnn = self.maxp(x5_cnn)
        x6_cnn = self.conv6(x6_cnn)
        x6_cnn = self.bn6(x6_cnn)

        x6_cnn = self.relu6(x6_cnn)

        # x7_cnn = self.maxp(x6_cnn)
        # x7_cnn = self.conv7(x7_cnn)
        # x7_cnn = self.bn7(x7_cnn)
        #
        # x7_cnn = self.relu7(x7_cnn)

        x_cnn = self.dropout(x6_cnn)
        # x_cnn = self.dropout(x5_cnn)
        # print(x_cnn.shape)
        # print('attn: ', attn.shape)
        x_static = self.pool(x_cnn)
        # x_static = torch.reshape(x_static, (x_static.shape[0], x_static.shape[1]))

        # ------------------LSTM-FORWARD------------------#

        input_rnn = input[1].permute(0, 2, 1)
        # print(x.shape)
        # x_w = self.linear(torch.mean(x, dim=2).reshape(x.shape[0], x.shape[-1]))

        # input_weights = torch.softmax(torch.autograd.Variable(torch.ones(x.shape[0], 3), requires_grad = True).cuda(), dim=-1).unsqueeze(-1).unsqueeze(-1)
        # input_weights = input_weights.expand(x.shape[0], 3,  x.shape[2], x.shape[3])
        #
        # input_rnn = torch.mul(x, input_weights).reshape(x.shape[0], x.shape[2], x.shape[1]*x.shape[3])



        #
        # print(input_rnn.shape)

        # x1 = self.norm(x1.reshape(-1, x1.shape[1]))
        # input_rnn = x1.permute(0, 2, 1)
        # input_rnn = torch.cat((x1,x2, x3), dim=-1)
        x_lstm, (h_n, h_c) = self.lstm(input_rnn, None)  # (batch, time_step, input_size)
        # x_dynamic = self.prelu(x_dynamic[:, -1, :])
        # x_dynamic = self.prelu(x_dynamic)
        # print(x_dynamic.shape)
        # x_dynamic = x_dynamic[:, -1, :]
        x_lstm = self.dropout(x_lstm)
        print("x_lstm", x_lstm.shape)
        x_lstm = nn.AdaptiveAvgPool1d(64)(x_lstm)
        print("x_lstm", x_lstm.shape)
        # ------------------CONCAT-FORWARD------------------#


        print('x_static: ', x_static.shape)
        # self.pool(x_cnn).reshape(x_static.shape[0], x_static.shape[1]),
        #                            torch.mean(x_lstm, dim=-1).reshape(x_lstm.shape[0], x_lstm.shape[1]),
        '''
        x11 = self.pool(x_cnn).reshape(x_static.shape[0], x_static.shape[1])
        x12 = torch.mean(x_lstm, dim=-2)
        x22 = x12.reshape(x_lstm.shape[0], x_lstm.shape[2])

        print('x11: ', x11.shape)
        print('x12: ', x12.shape)
        print('x22: ', x22.shape)
        '''
        out_x = torch.cat((self.pool(x_cnn).reshape(x_static.shape[0], x_static.shape[1]),
                                 torch.mean(x_lstm, dim=-2).reshape(x_lstm.shape[0], x_lstm.shape[2]),), dim=-1)
        print('out_x: ', out_x.shape)

        # out_x = torch.cat((x_static, x_dynamic),0)
        # out_x = torch.cat((out_x, x_fusion),0)

        out_emotion = self.integral_fc_emotion(out_x)
        # print('out: ', out_emotion.shape)
        if self.Gender and self.Center:
            out_sex = self.integral_fc_sex(out_x)
            out_sex_center = self.integral_fc_sex_center(out_x)
            out_emotion_center = self.integral_fc_emotion_center(out_x)
            print('out_sex: ', out_sex.shape)
            print('out_sex_center: ', out_sex_center.shape)
            print('out_emotion_center: ', out_emotion_center.shape)
            return out_x, out_emotion, out_sex, out_emotion_center, out_sex_center
        if self.Gender:
            out_sex = self.integral_fc_sex(out_x)
            return out_x, out_emotion, out_sex
        if self.Center:
            out_emotion_center = self.integral_fc_emotion_center(out_x)
            return out_x, out_emotion, out_emotion_center
        else:
            return out_x, out_emotion
        # ------------------END-FORWARD------------------#


if __name__ == '__main__':
    x = torch.randn(32, 60, 251)
    x1 = torch.randn(32, 180, 251)
    model = RC_mdp_AFF()
    model(x, x1,x )
