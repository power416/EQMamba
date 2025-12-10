import os
import torch
import torch.utils.data as data
import numpy as np
import random
from options import *

random.seed(1143)
s_num = 20


class Dataset(data.Dataset):
    def __init__(self, path, file_name, norm_mode=opt.norm_mode, n_channels=3):
        super(Dataset, self).__init__()
        self.n_channels = n_channels
        self.norm_mode = norm_mode
        self.data_dir = os.path.join(path, file_name)
        self.data_list = np.load(self.data_dir, allow_pickle=True)

    def __getitem__(self, index):
        df = self.data_list[index]
        # p = int(df['p_travel_sample'])
        # df_data = df['data'][p-100:p+2900, :]
        df_data = df['data']

        source_longitude = df['source_longitude_deg']
        source_latitude = df['source_latitude_deg']
        receiver_longitude = df['station_longitude_deg']
        receiver_latitude = df['station_latitude_deg']

        longitude = source_longitude - receiver_longitude
        latitude = source_latitude - receiver_latitude

        depth = df['source_depth_km']

        label_o = np.zeros(1)
        label_a = np.zeros(1)
        label_d = np.zeros(1)

        label_o[:] = float(longitude)
        label_a[:] = float(latitude)
        label_d[:] = float(depth)

        if self.norm_mode:
            df_data = self._normalize(df_data, self.norm_mode)
 
        df_data, label_o, label_a, label_d = self.augData(df_data, label_o, label_a, label_d)

        return df_data, label_o, label_a, label_d

    def augData(self, x, x_o, x_a, x_d):

        x = torch.from_numpy(x)
        x = torch.permute(x, [1, 0])

        x_o = torch.from_numpy(x_o).float()
        x_a = torch.from_numpy(x_a).float()
        x_d = torch.from_numpy(x_d).float()

        return x, x_o, x_a, x_d

    def _normalize(self, data, mode='std'):
        # Normalize waveforms in each batch
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            max_data[max_data == 0] = 1
            data /= max_data

        elif mode == 'std':
            std_data = np.std(data, axis=0, keepdims=True)
            std_data[std_data == 0] = 1
            data /= std_data
        return data

    def __len__(self):

        return len(self.data_list)
