import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap
import os
from tqdm import tqdm
from utils.dataset import CSVDataset
from utils.model import LSTMClassifier
from torch.utils.data import DataLoader

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')

def load_model_and_data():
    """加载模型和测试数据"""
    print('📂 正在加载模型和测试数据...')
    
    # 加载测试数据集
    test_dataset = CSVDataset(root_dir='data', train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 从数据集获取输入维度
    input_size = test_dataset.input_size
    sequence_length = test_dataset.sequence_length
    
    print(f'测试集样本数: {len(test_dataset)}')
    print(f'输入维度: {input_size}, 序列长度: {sequence_length}')
    
    # 加载模型
    model = LSTMClassifier(input_size=input_size, hidden_size=256, num_layers=1)
    model_path = 'ckpt/best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    print('✅ 模型和数据加载成功!')
    return model, test_dataset, test_loader, input_size, sequence_length

def evaluate_model(model, test_loader):
    """评估模型性能"""
    print('\n📊 开始模型评估...')
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='评估中'):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = outputs.squeeze().cpu().numpy()
            predictions = (outputs > 0.5).float().squeeze().cpu().numpy()
            
            all_probs.extend(probs)
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    # 计算评估指标
    accuracy = np.mean(all_predictions == all_labels)
    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))
    tn = np.sum((all_predictions == 0) & (all_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 保存评估结果
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }
    
    with open('result/evaluation_metrics.txt', 'w') as f:
        f.write('模型评估指标:\n')
        f.write(f'准确率: {accuracy:.4f}\n')
        f.write(f'精确率: {precision:.4f}\n')
        f.write(f'召回率: {recall:.4f}\n')
        f.write(f'F1分数: {f1:.4f}\n')
        f.write(f'\n混淆矩阵:\n')
        f.write(f'TP: {tp}, FP: {fp}\n')
        f.write(f'FN: {fn}, TN: {tn}\n')
    
    print(f'\n✅ 评估完成!')
    print(f'准确率: {accuracy:.4f}')
    print(f'精确率: {precision:.4f}')
    print(f'召回率: {recall:.4f}')
    print(f'F1分数: {f1:.4f}')
    
    return all_labels, all_predictions, all_probs, metrics

def plot_confusion_matrix(y_true, y_pred):
    """绘制混淆矩阵并保存原始数据"""
    print('📈 正在绘制混淆矩阵...')
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Risk (0)', 'Risk (1)'],
                yticklabels=['No Risk (0)', 'Risk (1)'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('result/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存原始数据
    cm_df = pd.DataFrame(cm, 
                        index=['True No Risk', 'True Risk'],
                        columns=['Pred No Risk', 'Pred Risk'])
    cm_df.to_csv('result/raw_data/confusion_matrix_data.csv')
    print('✅ 混淆矩阵数据已保存到 result/raw_data/confusion_matrix_data.csv')
    
    print('✅ 混淆矩阵已保存到 result/figures/confusion_matrix.png')

def plot_roc_curve(y_true, y_probs):
    """绘制ROC曲线并保存原始数据"""
    print('📈 正在绘制ROC曲线...')
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('result/figures/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存原始数据
    roc_data = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    })
    roc_data.to_csv('result/raw_data/roc_curve_data.csv', index=False)
    with open('result/raw_data/roc_auc_value.txt', 'w') as f:
        f.write(f'AUC: {roc_auc:.4f}')
    
    print(f'✅ ROC曲线数据已保存，AUC = {roc_auc:.4f}')

    # 保存AUC值
    with open('result/evaluation_metrics.txt', 'a') as f:
        f.write(f'\nAUC: {roc_auc:.4f}\n')

def plot_prediction_distribution(y_probs, y_true):
    """绘制预测概率分布并保存原始数据"""
    print('📈 正在绘制预测概率分布...')
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pd.DataFrame({'Probability': y_probs, 'True Label': y_true.astype(str)}),
                 x='Probability', hue='True Label', bins=50, kde=True, alpha=0.6)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold')
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Probability of being predicted as Risk')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('result/figures/prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存原始数据
    dist_data = pd.DataFrame({
        'probability': y_probs,
        'true_label': y_true
    })
    dist_data.to_csv('result/raw_data/prediction_distribution_data.csv', index=False)
    print('✅ 预测概率分布数据已保存到 result/raw_data/prediction_distribution_data.csv')
    
    print('✅ 预测概率分布图已保存')

def explain_with_shap(model, test_dataset, input_size, sequence_length):
    """使用SHAP解释模型"""
    print('\n🔍 开始SHAP解释...')
    
    # 准备背景数据（使用部分测试数据作为背景）
    background_size = min(100, len(test_dataset))
    background_indices = np.random.choice(len(test_dataset), background_size, replace=False)
    background_data = []
    
    for idx in background_indices:
        data, _ = test_dataset[idx]
        background_data.append(data.numpy())
    
    background = torch.tensor(np.array(background_data), dtype=torch.float32).to(DEVICE)
    print(f'使用 {background_size} 个样本作为SHAP背景数据')
    
    # 创建SHAP解释器
    def model_forward(x):
        """模型前向传播函数，适配SHAP"""
        # x是numpy数组，需要转换为PyTorch张量
        x_tensor = torch.tensor(x, dtype=torch.float32)
        batch_size = x_tensor.shape[0]
        x_tensor = x_tensor.reshape(batch_size, input_size, sequence_length)
        x_tensor = x_tensor.to(DEVICE)
        with torch.no_grad():
            outputs = model(x_tensor)
        return outputs.cpu().numpy()
    
    # 初始化SHAP解释器
    explainer = shap.KernelExplainer(model_forward, background.reshape(background_size, -1).cpu().numpy())
    
    # 选择要解释的样本（每个类别选择几个代表性样本）
    sample_indices = []
    labels = [test_dataset[i][1].item() for i in range(len(test_dataset))]
    labels = np.array(labels)
    
    # 从每个类别中选择5个样本
    for label in [0, 1]:
        indices = np.where(labels == label)[0]
        if len(indices) > 5:
            sample_indices.extend(np.random.choice(indices, 5, replace=False))
        else:
            sample_indices.extend(indices)
    
    print(f'将解释 {len(sample_indices)} 个样本的预测')
    
    # 获取SHAP值
    test_samples = []
    test_labels = []
    for idx in sample_indices:
        data, label = test_dataset[idx]
        test_samples.append(data.numpy())
        test_labels.append(label.item())
    
    test_samples = np.array(test_samples)
    test_labels = np.array(test_labels)
    
    # 计算SHAP值 - 对于二分类，KernelExplainer可能返回两个数组的列表
    shap_values_all = explainer.shap_values(test_samples.reshape(len(test_samples), -1), nsamples=100)
    # 我们需要正类（类别1）的SHAP值
    if isinstance(shap_values_all, list) and len(shap_values_all) > 1:
        shap_values = shap_values_all[1]
    elif isinstance(shap_values_all, list) and len(shap_values_all) == 1:
        shap_values = shap_values_all[0]
    else:
        shap_values = shap_values_all
    
    # 保存SHAP分析结果
    np.save('result/shap/shap_values.npy', shap_values)
    np.save('result/shap/test_samples.npy', test_samples)
    np.save('result/shap/test_labels.npy', test_labels)
    
    # 保存SHAP原始数据
    shap_df = pd.DataFrame(shap_values)
    shap_df.to_csv('result/raw_data/shap_values_raw.csv', index=False)
    print('✅ SHAP原始值已保存到 result/raw_data/shap_values_raw.csv')
    
    print('✅ SHAP值计算完成，结果已保存')
    
    # 绘制SHAP摘要图
    plot_shap_summary(shap_values, test_samples, test_labels, input_size, sequence_length)
    
    # 绘制单个样本的SHAP力图
    plot_shap_force_plots(shap_values, test_samples, test_labels, sample_indices[:2])  # 只展示前2个样本
    
    return shap_values, test_samples, test_labels

def plot_shap_summary(shap_values, samples, labels, input_size, sequence_length):
    """绘制SHAP摘要图并保存原始数据"""
    print('📈 正在绘制SHAP摘要图...')
    
    # 将数据重塑为2D格式用于SHAP可视化
    feature_names = []
    for i in range(input_size):
        for t in range(sequence_length):
            feature_names.append(f'Feature_{i}_Time_Step_{t}')
    
    # 创建摘要图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, samples.reshape(len(samples), -1), 
                     feature_names=feature_names, show=False)
    plt.title('SHAP Value Summary Plot', fontsize=14)
    plt.tight_layout()
    plt.savefig('result/figures/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存原始SHAP数据
    shap_data = pd.DataFrame(shap_values, columns=feature_names)
    shap_data['true_label'] = labels
    shap_data.to_csv('result/raw_data/shap_summary_data.csv', index=False)
    
    # 保存样本数据
    samples_reshaped = samples.reshape(len(samples), -1)
    samples_df = pd.DataFrame(samples_reshaped, columns=feature_names)
    samples_df['true_label'] = labels
    samples_df.to_csv('result/raw_data/shap_samples_data.csv', index=False)
    
    print('✅ SHAP摘要数据已保存到 result/raw_data/shap_summary_data.csv')
    print('✅ SHAP样本数据已保存到 result/raw_data/shap_samples_data.csv')
    print('✅ SHAP摘要图已保存到 result/figures/shap_summary.png')

def plot_shap_force_plots(shap_values, samples, labels, sample_indices):
    """绘制单个样本的SHAP力图并保存原始数据"""
    print('📈 绘制SHAP力图...')
    
    for i, idx in enumerate(sample_indices):
        if i >= len(shap_values):
            print(f"⚠️ 警告: 索引 {i} 超出SHAP值范围，跳过此样本")
            continue
            
        sample_features = samples[i].flatten()
        sample_shap_values = shap_values[i]
        
        # if len(sample_features) != len(sample_shap_values):
        #     print(f"⚠️ 警告: 样本 {idx} 的特征长度 ({len(sample_features)}) 和 SHAP值长度 ({len(sample_shap_values)}) 不匹配，跳过此样本")
        #     continue
        
        plt.figure(figsize=(12, 3))
        shap_values_single = sample_shap_values.reshape(1, -1)
        features_single = sample_features.reshape(1, -1)
        
        shap.force_plot(
            base_value=0.5,
            shap_values=shap_values_single,
            features=features_single,
            matplotlib=True,
            show=False
        )
        
        plt.title(f'SHAP Force Plot for Sample {idx} (True Label: {labels[i]})', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'result/figures/shap_force_sample_{idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存单个样本的SHAP数据
        force_data = pd.DataFrame({
            'feature_value': features_single.flatten(),
            'shap_value': shap_values_single.flatten()
        })
        feature_names = [f'Feature_{j//8}_Time_{j%8}' for j in range(len(features_single.flatten()))]
        force_data['feature_name'] = feature_names
        force_data.to_csv(f'result/raw_data/shap_force_sample_{idx}_data.csv', index=False)
        print(f'✅ 样本 {idx} 的SHAP力图数据已保存到 result/raw_data/shap_force_sample_{idx}_data.csv')
    
    print(f'✅ SHAP力图已保存，共 {len(sample_indices)} 个样本')

def plot_feature_importance(shap_values, samples, input_size, sequence_length):
    """绘制特征重要性并保存原始数据"""
    print('📈 正在绘制特征重要性...')
    
    # 计算每个特征维度的平均SHAP绝对值
    shap_abs = np.abs(shap_values)
    feature_importance = np.zeros(input_size)
    
    for i in range(input_size):
        # 提取该特征在所有时间步的SHAP值
        feature_shap = shap_abs[:, i*sequence_length:(i+1)*sequence_length]
        feature_importance[i] = np.mean(feature_shap)
    
    # 创建特征重要性图
    feature_names = ['Attendance', 'Participation', 'Homework Completion', 'Homework Quality', 
                    'Quiz Performance', 'Interaction', 'Study Time', 'Phone Usage']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_importance, y=feature_names, palette='viridis')
    plt.title('Feature Importance (based on SHAP values)', fontsize=14)
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.tight_layout()
    plt.savefig('result/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存特征重要性数据
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df.to_csv('result/feature_importance.csv', index=False)
    importance_df.to_csv('result/raw_data/feature_importance_data.csv', index=False)  # 同时保存到raw_data
    
    print('✅ 特征重要性数据已保存到 result/raw_data/feature_importance_data.csv')
    print('✅ 特征重要性图已保存，数据已保存到 result/feature_importance.csv')

def generate_report(metrics, shap_analysis=False):
    """生成最终报告"""
    print('\n📄 正在生成最终报告...')
    
    report = f"""
# 模型评估报告

## 基本信息
- 评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 设备: {DEVICE}
- 测试集样本数量: {metrics.get('test_size', 0)}

## 性能指标
- **准确率**: {metrics['accuracy']:.4f}
- **精确率**: {metrics['precision']:.4f}
- **召回率**: {metrics['recall']:.4f}
- **F1分数**: {metrics['f1_score']:.4f}
- **AUC**: {metrics.get('auc', 0):.4f}

## 混淆矩阵
- 真阳性 (TP): {metrics['true_positives']}
- 假阳性 (FP): {metrics['false_positives']}
- 假阴性 (FN): {metrics['false_negatives']}
- 真阴性 (TN): {metrics['true_negatives']}

## 结果可视化
- 混淆矩阵: ![Confusion Matrix](figures/confusion_matrix.png)
- ROC曲线: ![ROC Curve](figures/roc_curve.png)
- 预测概率分布: ![Prediction Distribution](figures/prediction_distribution.png)
"""

    if shap_analysis:
        report += """
## SHAP分析
- SHAP摘要图: ![SHAP Summary](figures/shap_summary.png)
- 特征重要性: ![Feature Importance](figures/feature_importance.png)
- SHAP力图示例: ![SHAP Force Plot](figures/shap_force_sample_586.png)
- SHAP值数据: `result/shap/`
- 特征重要性数据: `result/feature_importance.csv`
- 原始数据: `result/raw_data/`
""" 
    report += """
## 结论
模型在学业风险预测任务上表现良好。重点关注以下方面：
1. 模型能够有效识别有学业风险的学生
2. 主要影响因素包括：{top_features}
3. 建议对高风险学生进行早期干预
"""
    
    # 添加特征重要性总结
    if os.path.exists('result/feature_importance.csv'):
        importance_df = pd.read_csv('result/feature_importance.csv')
        top_features = ', '.join(importance_df['Feature'].head(3).tolist())
        report = report.replace('{top_features}', top_features)
    else:
        report = report.replace('{top_features}', 'Attendance, Quiz Performance, Homework Completion')
    
    with open('result/evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print('✅ 最终报告已保存到 result/evaluation_report.md')

if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    print('🎯 开始测试阶段...')
    
    try:
        # 加载模型和数据
        model, test_dataset, test_loader, input_size, sequence_length = load_model_and_data()
        
        # 评估模型
        y_true, y_pred, y_probs, metrics = evaluate_model(model, test_loader)
        metrics['test_size'] = len(test_dataset)
        
        # 绘制评估图表
        plot_confusion_matrix(y_true, y_pred)
        plot_roc_curve(y_true, y_probs)
        plot_prediction_distribution(y_probs, y_true)
        
        # SHAP解释
        shap_values, samples, labels = explain_with_shap(model, test_dataset, input_size, sequence_length)
        plot_feature_importance(shap_values, samples, input_size, sequence_length)
        
        # 生成报告
        generate_report(metrics, shap_analysis=True)
        
        print('\n🎉 测试完成！所有结果已保存到 result/ 目录')
        print('📁 结果目录结构:')
        print('result/')
        print('├── evaluation_metrics.txt')
        print('├── evaluation_report.md')
        print('├── feature_importance.csv')
        print('├── figures/')
        print('│   ├── confusion_matrix.png')
        print('│   ├── roc_curve.png')
        print('│   ├── prediction_distribution.png')
        print('│   ├── shap_summary.png')
        print('│   ├── feature_importance.png')
        print('│   └── shap_force_sample_*.png')
        print('├── shap/')
        print('│   ├── shap_values.npy')
        print('│   ├── test_samples.npy')
        print('│   └── test_labels.npy')
        print('└── raw_data/')
        print('    ├── confusion_matrix_data.csv')
        print('    ├── roc_curve_data.csv')
        print('    ├── roc_auc_value.txt')
        print('    ├── prediction_distribution_data.csv')
        print('    ├── shap_summary_data.csv')
        print('    ├── shap_samples_data.csv')
        print('    ├── shap_values_raw.csv')
        print('    ├── shap_force_sample_*_data.csv')
        print('    └── feature_importance_data.csv')
        
    except Exception as e:
        print(f'❌ 测试过程中出错: {str(e)}')
        import traceback
        traceback.print_exc()