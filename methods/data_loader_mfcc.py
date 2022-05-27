import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

dict = {
    0: torch.Tensor([0]),
    1: torch.Tensor([1]),
    2: torch.Tensor([2]),
    3: torch.Tensor([3]),
}


class DataSet(Dataset):
    def __init__(self, X, delta_X, delta_delta_X, Y, sex):
        self.X = X
        self.delta_X = delta_X
        self.delta_delta_X = delta_delta_X
        self.Y = Y
        self.sex = sex

    def __getitem__(self, index):
        x = self.X[index]

        x = torch.from_numpy(x.astype(np.float32))
        x = x.float()

        delta_x = self.delta_X[index]

        delta_x = torch.from_numpy(delta_x.astype(np.float32))
        delta_x = delta_x.float()

        delta_delta_x = self.delta_delta_X[index]

        delta_delta_x = torch.from_numpy(delta_delta_x.astype(np.float32))
        delta_delta_x = delta_delta_x.float()

        y = self.Y[index]
        y = dict[y]
        y = y.long()

        sex = self.sex[index]
        sex = dict[sex]
        sex = sex.long()

        return x, delta_x, delta_delta_x, y, sex

    def __len__(self):
        return len(self.X)

# if __name__=='__main__':
#     import pickle
#     file = r'../processing/features_mfcc_all.pkl'
#     with open(file, 'rb') as f:
#         features = pickle.load(f)
#
#     val_X_f = features['val_X_f']
#     val_X_t = features['val_X_t']
#     val_y = features['val_y']
#     val_sex = features['val_sex']
#
#     val_data = DataSet(val_X_f, val_X_t, val_y, val_sex)
#     val_loader = DataLoader(val_data, batch_size=10, shuffle=True)
#     for i ,data in enumerate(val_loader):
#         x_f, x_t, y, sex = data
#         print(y)
#         print(sex)
#         print(val_X_f.shape)
#         print(val_X_t.shape)
#         break
