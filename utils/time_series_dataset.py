import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, selected_sensors, window_size=30, max_time_fraction=0.2):
        self.data = data
        self.selected_sensors = selected_sensors
        self.window_size = window_size
        self.max_time_fraction = max_time_fraction
        self.windows = self._create_windows()

    def _create_windows(self):
        windows = []
        groups = self.data.groupby(['unit', 'dataset'])

        for (unit, dataset), group_data in groups:
            group_data = group_data.sort_values(by='time_cycles')
            max_time = int(len(group_data) * self.max_time_fraction)
            group_data = group_data.iloc[:max_time]

            for start in range(0, len(group_data) - self.window_size + 1):
                end = start + self.window_size
                window = group_data.iloc[start:end][self.selected_sensors].values
                windows.append(window)

        return np.array(windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)
