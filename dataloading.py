import math
import os
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
from constants import KEY_LM_HIDDEN_STATES, KEY_LM_INPUT_IDS, KEY_LM_LABELS
from utils import load_config


class HDF5Dataset(Dataset):
    def __init__(self, h5_files_dir, split):
        """
        Args:
            h5_files_dir (str): 
            split (str): train/validation/test
        """
        h5_file_path = os.path.join(h5_files_dir, split+'.h5')
        self.h5_file = h5py.File(h5_file_path, 'r')
        
        self.hidden_states = self.h5_file[KEY_LM_HIDDEN_STATES]
        self.input_ids = self.h5_file[KEY_LM_INPUT_IDS]
        self.labels = self.h5_file[KEY_LM_LABELS]
        self.total_samples = int(self.h5_file.attrs['total_samples'])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        hidden_states = torch.tensor(self.hidden_states[idx], dtype=torch.float)

        return {
            KEY_LM_INPUT_IDS: input_ids,
            KEY_LM_LABELS: labels,
            KEY_LM_HIDDEN_STATES: hidden_states
        }

    def close(self):
        self.h5_file.close()


class ChunkedHDF5Dataset(HDF5Dataset):
    def __init__(self, h5_files_dir, split, chunk_size:int):
        super().__init__(h5_files_dir, split)
        self.chunk_size = chunk_size
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        hidden_states = item['hidden_states']
        item['hidden_states'] = hidden_states[self.chunk_size-1::self.chunk_size, :]
        return item


def get_chunked_h5dataloader(config_path, split, shuffle=None):
    config = load_config(config_path=config_path)
    num_workers = 8  # Set num workers to 0 to enable debugging
    if shuffle is None:
        shuffle = split == 'train'
    dataset = ChunkedHDF5Dataset(config['h5_file_path'], split, chunk_size=config['chunk_size'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == '__main__':
    dataloader = get_chunked_h5dataloader('conf/data/example.yaml', 'test')

    for batch in dataloader:
        input_ids = batch[KEY_LM_INPUT_IDS]
        labels = batch[KEY_LM_LABELS]
        hidden_states = batch[KEY_LM_HIDDEN_STATES]
        
        print(f"Input IDs: {input_ids.shape}")
        print(f"Labels: {labels.shape}")
        print(f"Hidden States: {hidden_states.shape}")
        break  # 这里只打印一个批次的数据
