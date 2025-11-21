import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    """
    基于LSTM的二分类器，专为时序学生行为数据设计。
    优化点：
    1. 改进权重初始化：使用Xavier初始化
    2. 增加序列长度自适应处理：支持变长序列
    3. 优化特征提取方式：使用最后隐藏状态+平均池化
    """
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        """
        初始化LSTM分类器。
        
        参数:
        input_size: int - 输入特征维度
        hidden_size: int - LSTM隐藏层大小
        num_layers: int - LSTM层数
        """
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 批量归一化
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)
        # Sigmoid激活
        self.sigmoid = nn.Sigmoid()
        
        # 改进权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用Xavier初始化改进权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # 初始化全连接层
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x, seq_lengths=None):
        """
        前向传播过程。
        
        参数:
        x: torch.Tensor - 输入张量，形状为(batch, features, seq_len)
        seq_lengths: torch.Tensor - 可选，每个样本的实际序列长度
        
        返回:
        torch.Tensor - 预测概率，形状为(batch, 1)
        """
        # 转换维度：(batch, features, seq_len) -> (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # 增加序列长度自适应处理
        if seq_lengths is not None:
            # 按序列长度降序排列
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            x = x[perm_idx]
            
            # 创建打包序列
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=True
            )
            
            # LSTM处理
            packed_output, (hidden, _) = self.lstm(packed_input)
            
            # 解包序列
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # 恢复原始顺序
            _, unperm_idx = perm_idx.sort(0)
            lstm_out = lstm_out[unperm_idx]
            hidden = hidden[:, unperm_idx]
        else:
            # 标准LSTM处理
            lstm_out, (hidden, _) = self.lstm(x)
        
        # 优化特征提取方式：使用最后隐藏状态
        last_output = hidden[-1]  # 取最后层的最后隐藏状态
        
        # 应用批量归一化
        last_output = self.batch_norm(last_output)
        
        # 全连接层 + Sigmoid
        out = self.fc(last_output)
        out = self.sigmoid(out)
        
        return out