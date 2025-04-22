import math
import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from constants import KEY_LM_HIDDEN_STATES, KEY_LM_INPUT_IDS, KEY_LM_LABELS
from utils import load_config


class HDF5Dataset(Dataset):
    def __init__(self, h5_files_dir, split, batch_size):
        """
        Args:
            h5_files_dir (str): 
            split (str): train/validation/test
        """
        # import pdb; pdb.set_trace()
        h5_file_path = os.path.join(h5_files_dir, split+'.h5')
        self.h5_file = h5py.File(h5_file_path, 'r')
        
        self.hidden_states = self.h5_file[KEY_LM_HIDDEN_STATES]
        self.batch_size = batch_size
        self.total_samples = self.hidden_states.shape[0]
        self.len = math.ceil(self.total_samples / self.batch_size)

        self.idxs = torch.arange(0, self.total_samples, self.batch_size)
        self.idxs = torch.cat([self.idxs, torch.tensor([self.total_samples])])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            KEY_LM_HIDDEN_STATES: self.hidden_states[self.idxs[idx]:self.idxs[idx+1]],
        }

    def close(self):
        self.h5_file.close()


class ChunkedHDF5Dataset(HDF5Dataset):
    def __init__(self, h5_files_dir, split, chunk_size:int, batch_size:int):
        super().__init__(h5_files_dir, split, batch_size)
        self.chunk_size = chunk_size
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        hidden_states = item['hidden_states']
        # last
        # item['hidden_states'] = hidden_states[self.chunk_size-1::self.chunk_size, :]

        # mean pooling
        assert hidden_states.shape[1] % self.chunk_size == 0, "行数不能被 chunk_size 整除"

        # 重塑张量为 (bs, num_chunks, chunk_size, num_features)
        bs = hidden_states.shape[0]
        hidden_states_reshaped = hidden_states.reshape(bs, -1, self.chunk_size, hidden_states.shape[2])

        import pdb; pdb.set_trace()
        # 对每个 chunk 计算均值，axis=1 表示按第二个维度（即每个块的行）计算均值
        pooled_hidden_states = hidden_states_reshaped.mean(dim=2)

        # 结果的形状是 (256, 768)
        item['hidden_states'] = pooled_hidden_states
        
        return item


def get_chunked_h5dataloader(config_path, split, shuffle=None):
    config = load_config(config_path=config_path)
    num_workers = 0  # Set num workers to 0 to enable debugging
    if shuffle is None:
        shuffle = split == 'train'
    dataset = ChunkedHDF5Dataset(config['h5_file_path'], split, chunk_size=config['chunk_size'], batch_size=config['batch_size'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# dataloader = get_chunked_h5dataloader('conf/data/example.yaml', 'test')

if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    dataloader = get_chunked_h5dataloader('exp/0407_rsimvq_CN64_CS16_LR1E-3_EPOCH1_openwebtext/data_config.yaml', 'test')
    # dataloader = get_chunked_h5dataloader('conf/data/layer6_mlp.yaml', 'train')

    for batch in dataloader:
        input_ids = batch[KEY_LM_INPUT_IDS]
        # labels = batch[KEY_LM_LABELS]
        # hidden_states = batch[KEY_LM_HIDDEN_STATES]
        
        print(f"Input IDs: {input_ids.shape}")
        # print(f"Labels: {labels.shape}")
        # print(f"Hidden States: {hidden_states.shape}")
        break  # 这里只打印一个批次的数据
