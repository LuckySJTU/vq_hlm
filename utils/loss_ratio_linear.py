import sys
sys.path.append("..")

from dataloading_pooling import get_chunked_h5dataloader
import torch
from constants import KEY_LM_HIDDEN_STATES, KEY_LM_INPUT_IDS, KEY_LM_LABELS
import numpy as np
from tqdm import tqdm

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_dataset_stats(dataloader, ratio=0.7):
    total_loss = 0
    # 遍历整个数据集并收集所有的 hidden_states
    for i, batch in enumerate(dataloader):
        hidden_states = batch[KEY_LM_HIDDEN_STATES]
        out = ratio * hidden_states
        # 计算重构损失
        # rec_loss = (out - hidden_states).abs().mean() # l1
        rec_loss = ((out - hidden_states) ** 2).mean() # l2

        total_loss += rec_loss.item()

    total_loss /= len(dataloader)
    return total_loss

if __name__ == '__main__':
    seed_everything(10086)
    dataloader = get_chunked_h5dataloader('../conf/data/example.yaml', 'test')
    ratios = np.arange(0.6, 1.02, 0.02)  # 1.02 是为了确保包含 1.0

    for ratio in ratios:
        # 线性方法只需要计算一次即可
        loss = calculate_dataset_stats(dataloader, ratio)
        # 输出均值损失
        print(f'Average Loss for ratio {ratio:.2f}: {loss:.5f}')