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
        mean = hidden_states.mean()
        var = hidden_states.var()
        # 创建一个形状为 (256,) 的掩码，标记出比例为 (1-ratio) 的部分
        mask = torch.rand(hidden_states.size(1)) < ratio  # mask的大小是 (256,)
        
        # 扩展掩码到 (128, 256, 768) 形状
        mask = mask.unsqueeze(0).unsqueeze(2).expand(hidden_states.size(0), hidden_states.size(1), hidden_states.size(2))
        
        # 生成一个形状与 hidden_states 相同的高斯分布张量
        # gaussian_fill = mean + torch.randn_like(hidden_states)*var  # 生成均值为0，方差为1的高斯分布
        gaussian_fill = torch.normal(mean=hidden_states.mean(),std=hidden_states.std())

        
        # 将被掩盖为零的部分用高斯分布填充
        masked_hidden_states = hidden_states * mask.float() + gaussian_fill * (1 - mask.float())
        
        # 计算重构损失
        # rec_loss = (masked_hidden_states - hidden_states).abs().mean() # l1
        rec_loss = ((masked_hidden_states - hidden_states) ** 2).mean() # l2

        total_loss += rec_loss.item()

    total_loss /= len(dataloader)
    return total_loss

if __name__ == '__main__':
    seed_everything(10086)
    dataloader = get_chunked_h5dataloader('../conf/data/example.yaml', 'test')
    ratios = np.arange(0.6, 1.02, 0.02)  # 1.02 是为了确保包含 1.0

    for ratio in ratios:
        losses = []  # 用于存储每个 ratio 下的 100 次损失值
        
        # 运行 100 次计算损失
        for _ in range(100):
            loss = calculate_dataset_stats(dataloader, ratio)
            losses.append(loss)
        
        # 计算损失的均值
        avg_loss = np.mean(losses)
        
        # 输出均值损失
        print(f'Average Loss for ratio {ratio:.2f}: {avg_loss:.5f}')