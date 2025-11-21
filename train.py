import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataset import CSVDataset
from utils.model import LSTMClassifier
import torch.nn as nn
import numpy as np
import random
import os
from tqdm import tqdm

# 超参数设置 - 根据我们生成的数据特点优化
DATA_DIR = 'data'
HIDDEN_SIZE = 256          # 降低隐藏层大小，避免过拟合（数据较简单）
NUM_EPOCHS = 30           # 减少训练轮次，配合早停
BATCH_SIZE = 16           # 减小batch size，提高训练稳定性
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4       # 添加L2正则化
EARLY_STOPPING_PATIENCE = 5  # 早停耐心值
USE_BATCH_NORM = True
VALIDATION_SPLIT = 0.2    # 20%训练数据用于验证
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = 'ckpt/best_model.pth'

def set_seed(seed=42):
    """设置随机种子确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience):
    """
    训练循环，包含验证和早停机制。
    
    参数:
    model: LSTM分类模型
    train_loader: 训练数据加载器
    val_loader: 验证数据加载器  
    criterion: 损失函数
    optimizer: 优化器
    num_epochs: 最大训练轮次
    patience: 早停耐心值（验证损失连续多少轮不改善就停止）
    
    返回:
    最佳模型
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs.data > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels.unsqueeze(1)).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                
                val_loss += loss.item()
                predicted = (outputs.data > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels.unsqueeze(1)).sum().item()
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f'  🔥 验证损失改善！保存最佳模型')
        else:
            patience_counter += 1
            print(f'  ⏰ 验证损失未改善，耐心值: {patience_counter}/{patience}')
        
        if patience_counter >= patience:
            print(f'  🛑 早停触发！训练结束')
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, test_loader):
    """
    在测试集上评估模型性能。
    
    参数:
    model: 训练好的模型
    test_loader: 测试数据加载器
    
    返回:
    准确率、精确率、召回率、F1分数
    """
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            predicted = (outputs.data > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # 计算各项指标
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions).flatten()
    
    accuracy = 100 * np.mean(all_predictions == all_labels)
    
    # 精确率、召回率、F1分数
    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print('\n' + '='*50)
    print(f'测试集评估结果:')
    print(f'准确率: {accuracy:.2f}%')
    print(f'精确率: {precision:.4f}')
    print(f'召回率: {recall:.4f}')
    print(f'F1分数: {f1:.4f}')
    print('='*50)
    
    return accuracy, precision, recall, f1

def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f'模型已保存至: {path}')

if __name__ == '__main__':
    # 设置随机种子确保可重现性
    set_seed(42)
    
    print('🚀 开始训练学业风险预测模型')
    print(f'使用设备: {DEVICE}')
    
    # 创建数据集
    full_train_dataset = CSVDataset(root_dir=DATA_DIR, train=True)
    test_dataset = CSVDataset(root_dir=DATA_DIR, train=False)
    
    # 验证数据集分割
    val_size = int(VALIDATION_SPLIT * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    print(f'数据集划分:')
    print(f'  训练集: {train_size} 个样本')
    print(f'  验证集: {val_size} 个样本') 
    print(f'  测试集: {len(test_dataset)} 个样本')
    
    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 从数据集推断输入维度
    input_size = full_train_dataset.input_size  # 应为8（8个行为维度）
    sequence_length = full_train_dataset.sequence_length  # 应为8（8周时序）
    
    print(f'推断的输入维度: {input_size}')
    print(f'推断的序列长度: {sequence_length}')

    # 初始化模型
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=1
    ).to(DEVICE)
    
    print('模型结构:')
    print(model)
    print(f'总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY  # L2正则化
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )

    # 训练模型
    print('\n' + '='*50)
    print('开始训练...')
    print('='*50)
    
    best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE
    )
    
    # 在测试集上评估
    print('\n' + '='*50)
    print('在测试集上评估最佳模型...')
    print('='*50)
    
    accuracy, precision, recall, f1 = evaluate_model(best_model, test_loader)
    
    # 保存最佳模型
    save_model(best_model, PATH)
    
    print('🎉 训练完成！')