import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import glob

class CSVDataset(Dataset):
    """
    自定义数据集类，用于从目录结构中加载CSV文件。
    目录结构要求：根目录下包含以类别标签命名的子目录（如0/、1/），
    每个子目录下包含train/和test/文件夹，存放对应类别的CSV样本文件。
    """
    def __init__(self, root_dir, train=True):
        """
        初始化数据集。
        
        参数:
        root_dir: str - 根目录路径，包含0/、1/等类别子目录
        train: bool - True加载训练集，False加载测试集
        """
        self.root_dir = root_dir
        self.train = train
        self.data_files = []    # 存储所有CSV文件路径
        self.labels = []        # 存储对应标签
        self.input_size = -1    # 特征维度数（行数）
        self.sequence_length = -1  # 时序长度（列数）

        # 遍历根目录下的所有类别子目录（0, 1）
        for label in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label)
            if os.path.isdir(class_dir):
                # 确定是加载train还是test数据
                mode = 'train' if self.train else 'test'
                data_dir = os.path.join(class_dir, mode)
                if os.path.isdir(data_dir):
                    # 获取该类别下所有CSV文件
                    files = glob.glob(os.path.join(data_dir, '*.csv'))
                    self.data_files.extend(files)
                    # 为每个文件添加对应的标签（转换为整数）
                    self.labels.extend([int(label)] * len(files))
        
        # 从第一个文件推断输入维度和序列长度
        if len(self.data_files) > 0:
            # 读取第一个CSV文件，无表头
            sample_data = pd.read_csv(self.data_files[0], header=None).values
            self.input_size = sample_data.shape[0]      # 行数 = 特征维度
            self.sequence_length = sample_data.shape[1]  # 列数 = 时序长度
            print(f"数据形状推断: input_size={self.input_size}, sequence_length={self.sequence_length}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        获取单个样本。
        
        参数:
        idx: int - 样本索引
        
        返回:
        data_tensor: torch.Tensor - 形状为(features, sequence_length)的张量
        label_tensor: torch.Tensor - 标签张量（0或1）
        """
        file_path = self.data_files[idx]
        label = self.labels[idx]
        
        # 读取CSV文件，无表头，转换为numpy数组
        data = pd.read_csv(file_path, header=None).values
        # 转换为PyTorch张量
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return data_tensor, label_tensor


# 测试代码：验证是否能正确读取我们生成的数据
if __name__ == "__main__":
    # 假设数据存放在'data'目录下
    print("测试训练集加载...")
    train_dataset = CSVDataset(root_dir="data", train=True)
    print(f"训练集样本数量: {len(train_dataset)}")
    
    print("\n测试测试集加载...")
    test_dataset = CSVDataset(root_dir="data", train=False)
    print(f"测试集样本数量: {len(test_dataset)}")
    
    # 验证单个样本
    if len(train_dataset) > 0:
        sample_data, sample_label = train_dataset[0]
        print(f"\n单个样本验证:")
        print(f"数据形状: {sample_data.shape} (应为[8, 8])")
        print(f"标签: {sample_label.item()}")
        print(f"数据范围: min={sample_data.min().item():.3f}, max={sample_data.max().item():.3f}")
    
    # 统计类别分布
    print("\n类别分布统计:")
    from collections import Counter
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    test_labels = [test_dataset[i][1].item() for i in range(len(test_dataset))]
    print(f"训练集: {Counter(train_labels)}")
    print(f"测试集: {Counter(test_labels)}")