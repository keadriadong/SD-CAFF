#! /usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from center_loss import CenterLoss
from tqdm import tqdm
import pickle
import logging
import time
import random
import warnings
warnings.filterwarnings('ignore')

from data_loader_mfcc import DataSet
from LSTM import RNN


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(111111)
# setup_seed(123456)
# setup_seed(0)
# setup_seed(999999)
def train(dataset, feature_type, model_name, data_num, root_dir):
    setup_seed(987654)

    # logger setting
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)  # Log等级总开关
    #
    # # 第二步，创建一个handler，用于写入日志文件
    # log_dir = root_dir + r'{}/{}_res/{}/{}/{}_loggings/'.format(dataset, dataset, model_name, feature_type, feature_type)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # logfile = log_dir + 'train_{}_{}.log'.format(dataset, data_num)
    # fh = logging.FileHandler(logfile, mode='w')
    # fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    #
    # # 第三步，定义handler的输出格式
    # formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    # fh.setFormatter(formatter)
    #
    # # 第四步，将logger添加到handler里面
    # logger.addHandler(fh)

    # parameters setting
    alpha = 0.7
    beta = 0.3
    center_rate = 0.15
    learning_rate = 0.0005
    lr_cent = 0.1
    Epoch = 30
    BatchSize = 32
    # MODEL_NAME='MULITMANET_with_gender'

    # MODEL_PATH = './model_result/IEMOCAP.pth'.format(str(case),element, file_num)

    # load features file
    print('load features data ...')
    # logging.info('load features data...')
    # file = r'/home/IEMOCAP_leave_{}_data.pkl'.format(data_num)
    if dataset == 'IEMOCAP':
        file = r'/home/data/IEMOCAP_leave_{}_data.pkl'.format(str(data_num + 1))
    elif dataset == 'MSP':
        file = root_dir + r'MSP/MSP_data/MSP_leave_{}_data_16000.pkl'.format(str(data_num + 1))
    elif dataset == 'MELD':
        file = root_dir + r'MELD/MELD_data/MELD_data_{}.pkl'.format(str(data_num))
    else:
        print('Wrong dataset name!')

    with open(file, 'rb') as f:
        features = pickle.load(f)

    if dataset == 'IEMOCAP':
        val_X = features['valid_x']
        val_delta_X = features['valid_d']  # delta
        val_delta2_X = features['valid_dd']  # delta

        val_y = features['valid_emo']
        val_sex = features['valid_sex']

        train_X = features['train_x']
        train_delta_X = features['train_d']
        train_delta2_X = features['train_dd']

        train_y = features['train_emo']
        train_sex = features['train_sex']

    elif dataset == 'MSP':
        val_X = features['valid_x']
        val_delta_X = np.array(features['valid_x_delta'])  # delta
        val_delta2_X = np.array(features['valid_x_delta2'])  # delta

        val_y = features['valid_emo']
        val_sex = features['valid_sex']

        train_X = features['train_x']
        train_delta_X = np.array(features['train_x_delta'])
        train_delta2_X = np.array(features['train_x_delta2'])

        train_y = features['train_emo']
        train_sex = features['train_sex']

    elif dataset == 'MELD':
        val_X = np.array(features['valid_x'])
        val_delta_X = np.array(features['valid_x_delta'])  # delta
        val_delta2_X = np.array(features['valid_x_delta2'])  # delta

        val_y = np.array(features['valid_emo'])
        val_sex = np.array(features['valid_sex'])

        train_X = np.array(features['train_x'])
        train_delta_X = np.array(features['train_x_delta'])
        train_delta2_X = np.array(features['train_x_delta2'])

        train_y = np.array(features['train_emo'])
        train_sex = np.array(features['train_sex'])

    else:
        print('unknown dataset')

    print(train_X.shape)
    '''training processing'''
    print('start training...')
    # logging.info('start training....')

    # load data
    #  train_trans, train_trans_len,
    train_data = DataSet(train_X, train_delta_X, train_delta2_X, train_y, train_sex)  # 加上 delta
    train_loader = DataLoader(train_data, batch_size=BatchSize, shuffle=True)

    # load model
    # ahead_text = 7, ahidden_text = 96

    rnn = RNN()

    if torch.cuda.is_available():
        rnn = rnn.cuda()

    # criterion
    criterion = nn.CrossEntropyLoss()
    center_loss = CenterLoss(num_classes=4, feat_dim=4, use_gpu=True)
    center_loss_sex = CenterLoss(num_classes=2, feat_dim=2, use_gpu=True)

    params = list(rnn.parameters()) + list(center_loss.parameters()) + list(center_loss_sex.parameters())
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=1e-6)

    # result saving
    maxWA = 0
    maxUA = 0
    totalrunningtime = 0

    for i in range(Epoch):
        start_time = time.time()
        # tq = tqdm(len(train_y))

        rnn.train()
        print_loss = 0

        for _, data in tqdm(enumerate(train_loader)):
            # x, delta_x, delta_delta_x, y, sex
            x1, x2, x3, y, sex = data
            if feature_type == "mfcc":
                x = x1
            elif feature_type == "delta":
                x = x2
            elif feature_type == "delta2":
                x = x3
            elif feature_type == "delta_delta2":
                x = x2 + x3
            elif feature_type == "mfcc_delta":
                x = x1 + x2
            elif feature_type == "mfcc_delta2":
                x = x1 + x3
            elif feature_type == "mfcc_delta_delta2":
                x = torch.cat((x1 ,x2, x3), dim=-2)
            else:
                print("feature type error!")

            if torch.cuda.is_available():
                x = x.cuda()

                y = y.cuda()
                sex = sex.cuda()

            attn, out_emotion, out_gender, out_emotion_center, out_gender_center = rnn(x)
            #

            loss_emotion = criterion(out_emotion.squeeze(1), y.squeeze(1))
            center_loss_emotion = center_loss(out_emotion_center, y.squeeze(1))

            loss_gender = criterion(out_gender, sex.squeeze(1))
            center_loss_gender = center_loss_sex(out_gender_center, sex.squeeze(1))

            loss = alpha * (loss_emotion + center_rate * center_loss_emotion) + beta * (
                    loss_gender + center_rate * center_loss_gender)

            print_loss += loss.data.item() * BatchSize
            optimizer.zero_grad()
            loss.backward()
            for param in center_loss.parameters():
                param.grad.data *= (lr_cent / (center_rate * learning_rate))
            optimizer.step()
            # tq.update(BatchSize)
        # tq.close()
        print('epoch: {}, loss: {:.4}'.format(i, print_loss / len(train_y)))
        # logging.info('epoch: {}, loss: {:.4}'.format(i, print_loss))
        if i > 0 and i % 10 == 0:
            learning_rate = learning_rate / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        '''validation process'''
        end_time = time.time()
        totalrunningtime += end_time - start_time
        print('total_running_time:', totalrunningtime)
        rnn.eval()
        UA = [0, 0, 0, 0]
        num_correct = 0
        class_total = [0, 0, 0, 0]
        matrix = np.mat(np.zeros((4, 4)), dtype=int)

        for i in range(len(val_y)):
            x1 = torch.from_numpy(val_X[i]).float()
            x2 = torch.from_numpy(val_delta_X[i]).float()
            x3 = torch.from_numpy(val_delta2_X[i]).float()

            if feature_type == "mfcc":
                x = x1
            elif feature_type == "delta":
                x = x2
            elif feature_type == "delta2":
                x = x3
            elif feature_type == "delta_delta2":
                x = x2 + x3
            elif feature_type == "mfcc_delta":
                x = x1 + x2
            elif feature_type == "mfcc_delta2":
                x = x1 + x3
            elif feature_type == "mfcc_delta_delta2":
                x = torch.cat((x1 ,x2, x3), dim=-2)
            else:
                print("feature type error!")

            y = torch.from_numpy(np.array(val_y[i])).long()
            sex = torch.from_numpy(np.array(val_sex[i])).long()

            if torch.cuda.is_available():
                x = x.cuda()

                y = y.cuda()
                sex = sex.cuda()

            out_a, out_emotion, out_gender, out_emotion_center, out_gender_center = rnn(x.unsqueeze(0))
            #
            pred_emotion = torch.max(out_emotion, 1)[1]
            pred_gender = torch.max(out_gender, 1)[1]
            # print("pred_emotion: ",pred_emotion)
            # print("y: ",y.item())

            if pred_emotion[0] == y.item():
                num_correct += 1
            matrix[int(y.item()), int(pred_emotion[0])] += 1

        for i in range(4):
            for j in range(4):
                class_total[i] += matrix[i, j]
            UA[i] = round(matrix[i, i] / class_total[i], 3)
        WA = num_correct / len(val_y)
        if (maxWA < WA):
            maxWA = WA
            best_matrix = matrix

        if (maxUA < sum(UA) / 4):
            maxUA = sum(UA) / 4

        print('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
        # logging.info('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
        print(matrix)

        logging.info(matrix)

    res_dir = root_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    result_file = os.path.join(res_dir, 'MSP_{}_results.pkl'.format( data_num))

    results = {
        'WA': maxWA,
        'UA': maxUA,
        'Matrix': best_matrix
    }

    print(result_file)
    with open(result_file, 'wb') as f:
        pickle.dump(results, f)

    return maxWA, maxUA, best_matrix

if __name__ == '__main__':
    model_name = 'LSTM'

    datasets = ['IEMOCAP']
    feature_types = ['mfcc_delta_delta2']
    num = 5

    # datasets = ['IEMOCAP', 'MSP', 'MELD']
    # feature_types = [ 'mfcc_delta', 'mfcc_delta2', 'mfcc_delta_delta2']
    # feature_types = ['mfcc', 'delta', 'delta2', 'delta_delta2', 'mfcc_delta', 'mfcc_delta2', 'mfcc_delta_delta2']
    # feature_types = ['mfcc', 'delta_delta2', 'mfcc_delta', 'mfcc_delta2', 'mfcc_delta_delta2']
    for dataset in datasets:
        for feature_type in feature_types:
            root_dir = "/home/results_LSTM"
            avg_WA, avg_UA = 0, 0
            for n in range(num):
                maxWA, maxUA, best_matrix = train(dataset, feature_type, model_name, n, root_dir)
                avg_WA += maxWA
                avg_UA += maxUA
            avg_WA /= 5
            avg_UA /= 5

            result_dir = root_dir
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            result_file = os.path.join(result_dir, 'avg_results.pkl')

            results = {
                'avg_WA': avg_WA,
                'avg_UA': avg_UA,
            }

            print(result_file)
            with open(result_file, 'wb') as f:
                pickle.dump(results, f)

    print("end")