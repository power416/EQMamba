import os
import torch
import torch.utils.data as data
import numpy as np
import random
from options import *
from scipy import signal

random.seed(1143)
s_num = 20


class Dataset(data.Dataset):
    def __init__(self, path, file_name, filter_data=True, norm_mode=opt.norm_mode, n_channels=3):
        super(Dataset, self).__init__()
        self.n_channels = n_channels
        self.filter_data = filter_data
        self.norm_mode = norm_mode
        self.data_dir = os.path.join(path, file_name)
        self.data_list = np.load(self.data_dir, allow_pickle=True)
        
        self.m1, self.n1 = signal.butter(2, [1.0, 45.0], btype='bandpass', fs=100)

    def __getitem__(self, index):
        df = self.data_list[index]
        p = int(df['p_arrival_sample'])
        start = max(0, p-100)
        end = start + 3000
        df_data = df['data'][start:end, :].astype(np.float32)

        longitude = float(df['source_longitude_deg']-df['station_longitude_deg'])
        label_o = torch.tensor([longitude], dtype=torch.float32)

        latitude = float(df['source_latitude_deg']-df['station_latitude_deg'])
        label_a = torch.tensor([latitude], dtype=torch.float32)

        depth = float(df['source_depth_km'])
        label_d = torch.tensor([depth], dtype=torch.float32)

        if self.filter_data:
            df_data = signal.filtfilt(self.m1, self.n1, df_data, axis=0)

        if self.norm_mode:
            df_data = self._normalize(df_data, self.norm_mode)

        df_data = torch.from_numpy(df_data.copy()).float().permute(1, 0)

        return df_data, label_o, label_a, label_d

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
