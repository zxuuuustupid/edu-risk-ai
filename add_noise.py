import numpy as np
import pandas as pd
import os
import random
import glob
from tqdm import tqdm

def apply_simple_jitter(data, intensity=0.2):
    """简单的高强度抖动"""
    noise = np.random.normal(0, intensity, data.shape)
    return np.clip(data + noise, 0, 1)

def apply_simple_scaling(data, scale_range=(0.6, 1.4)):
    """简单的随机缩放"""
    n_dims = data.shape[0]
    scale_factors = np.random.uniform(scale_range[0], scale_range[1], n_dims)
    scaled_data = data.copy()
    for i in range(n_dims):
        scaled_data[i] = np.clip(scaled_data[i] * scale_factors[i], 0, 1)
    return scaled_data

def inject_simple_outliers(data, outlier_prob=0.25, outlier_intensity=0.5):
    """简单的异常值注入"""
    outlier_data = data.copy()
    n_dims, n_steps = data.shape
    
    for dim in range(n_dims):
        for step in range(n_steps):
            if random.random() < outlier_prob:
                if random.random() > 0.5:
                    outlier_data[dim, step] = min(1.0, outlier_data[dim, step] + random.uniform(0.2, outlier_intensity))
                else:
                    outlier_data[dim, step] = max(0.0, outlier_data[dim, step] - random.uniform(0.2, outlier_intensity))
    
    return outlier_data

def apply_simple_missing(data, missing_prob=0.15):
    """简单的缺失值处理"""
    missing_data = data.copy()
    n_dims, n_steps = data.shape
    
    for dim in range(n_dims):
        for step in range(n_steps):
            if random.random() < missing_prob:
                # 用随机值替换，避免维度问题
                missing_data[dim, step] = random.uniform(0.1, 0.9)
    
    return missing_data

def apply_simple_magnitude_warp(data, warp_prob=0.7):
    """简单的幅度扭曲"""
    warped_data = data.copy()
    n_dims, n_steps = data.shape
    
    for dim in range(n_dims):
        if random.random() < warp_prob:
            mean_val = np.mean(data[dim])
            warp_factor = random.uniform(0.5, 2.0)
            warped_data[dim] = mean_val + (data[dim] - mean_val) * warp_factor
    
    return np.clip(warped_data, 0, 1)

def add_simple_bias(data, risk_level):
    """简单的系统性偏差"""
    biased_data = data.copy()
    n_dims, n_steps = data.shape
    
    if risk_level == 0:  # 无风险学生
        # 随机选择2-3个维度添加负面偏差
        bad_dims = random.sample(range(n_dims), random.randint(2, 3))
        for dim in bad_dims:
            bias = random.uniform(-0.3, -0.15)
            biased_data[dim] = np.clip(biased_data[dim] + bias, 0, 1)
    
    else:  # 有风险学生
        # 随机选择2-3个维度添加正面偏差
        good_dims = random.sample(range(n_dims), random.randint(2, 3))
        for dim in good_dims:
            bias = random.uniform(0.2, 0.4)
            biased_data[dim] = np.clip(biased_data[dim] + bias, 0, 1)
    
    return biased_data

def apply_safe_perturbation(data, risk_level):
    """安全的扰动组合，避免维度问题"""
    try:
        # 应用各种扰动，但避免复杂的时间扭曲
        data = apply_simple_jitter(data, intensity=0.25)
        data = apply_simple_scaling(data, scale_range=(0.5, 1.6))
        data = inject_simple_outliers(data, outlier_prob=0.3, outlier_intensity=0.6)
        data = apply_simple_missing(data, missing_prob=0.2)
        data = apply_simple_magnitude_warp(data, warp_prob=0.8)
        data = add_simple_bias(data, risk_level)
        
        return data
    except Exception as e:
        print(f"扰动过程中出错: {e}")
        return data  # 返回原始数据作为回退

def validate_and_fix_data(data):
    """验证数据维度并修复常见问题"""
    expected_dims = 8  # 8个行为维度
    expected_steps = 8  # 8个时间步
    
    # 检查维度
    if data.shape[0] != expected_dims or data.shape[1] != expected_steps:
        print(f"发现维度问题: {data.shape}, 修复中...")
        
        # 修复行数
        if data.shape[0] < expected_dims:
            # 添加缺失的行
            missing_rows = expected_dims - data.shape[0]
            new_rows = np.random.uniform(0.2, 0.8, (missing_rows, expected_steps))
            data = np.vstack([data, new_rows])
        elif data.shape[0] > expected_dims:
            # 截断多余的行
            data = data[:expected_dims, :]
        
        # 修复列数
        if data.shape[1] < expected_steps:
            # 添加缺失的列
            missing_cols = expected_steps - data.shape[1]
            new_cols = np.random.uniform(0.2, 0.8, (expected_dims, missing_cols))
            data = np.hstack([data, new_cols])
        elif data.shape[1] > expected_steps:
            # 截断多余的列
            data = data[:, :expected_steps]
    
    return data

def perturb_and_overwrite_safe(data_dir='data'):
    """
    安全的扰动程序，直接覆盖原有数据
    """
    print("警告：此操作将直接覆盖原始数据文件！")
    print("按 Enter 继续，或 Ctrl+C 取消...")
    input()
    
    # 设置随机种子
    np.random.seed(44)  # 使用新的种子
    random.seed(44)
    
    total_files = 0
    successful_files = 0
    error_files = 0
    
    # 遍历所有风险等级和数据集划分
    for risk_level in [0, 1]:
        for split in ['train', 'test']:
            input_path = os.path.join(data_dir, str(risk_level), split)
            
            if not os.path.exists(input_path):
                print(f"警告：路径不存在 {input_path}")
                continue
                
            # 获取所有CSV文件
            csv_files = glob.glob(os.path.join(input_path, '*.csv'))
            total_files += len(csv_files)
            
            print(f"\n处理 {risk_level}/{split}: {len(csv_files)} 个文件")
            
            for file_path in tqdm(csv_files, desc=f"{risk_level}/{split}"):
                try:
                    # 读取数据，使用更安全的方式
                    df = pd.read_csv(file_path, header=None)
                    
                    # 转换为numpy数组
                    data = df.values.astype(float)
                    
                    # 验证并修复数据维度
                    data = validate_and_fix_data(data)
                    
                    # 应用扰动
                    perturbed_data = apply_safe_perturbation(data, risk_level)
                    
                    # 确保维度正确
                    if perturbed_data.shape != (8, 8):
                        print(f"警告：文件 {file_path} 扰动后维度不正确 {perturbed_data.shape}，重新修复")
                        perturbed_data = validate_and_fix_data(perturbed_data)
                    
                    # 覆盖原文件
                    perturbed_df = pd.DataFrame(perturbed_data)
                    perturbed_df.to_csv(file_path, header=False, index=False)
                    
                    successful_files += 1
                    
                except Exception as e:
                    error_files += 1
                    print(f"处理文件 {file_path} 时出错: {e}")
                    # 尝试简单的修复
                    try:
                        # 生成一个合理的随机数据作为替代
                        fallback_data = np.random.uniform(0.2, 0.8, (8, 8))
                        fallback_df = pd.DataFrame(fallback_data)
                        fallback_df.to_csv(file_path, header=False, index=False)
                        print(f"已为 {file_path} 生成替代数据")
                    except:
                        print(f"无法修复文件 {file_path}")
    
    print(f"\n扰动完成！")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {successful_files}")
    print(f"错误文件: {error_files}")
    print(f"原始数据已被扰动覆盖（错误文件已用随机数据替代）")
    
    # 验证结果
    print("\n验证扰动结果:")
    for risk_level in [0, 1]:
        for split in ['train', 'test']:
            check_path = os.path.join(data_dir, str(risk_level), split)
            if os.path.exists(check_path):
                file_count = len(glob.glob(os.path.join(check_path, '*.csv')))
                print(f"{check_path}: {file_count} 个文件")
                
                # 检查几个样本文件
                sample_files = glob.glob(os.path.join(check_path, '*.csv'))[:3]
                for sample_file in sample_files:
                    try:
                        sample_df = pd.read_csv(sample_file, header=None)
                        print(f"  样本 {os.path.basename(sample_file)}: 形状 {sample_df.shape}")
                    except Exception as e:
                        print(f"  无法读取样本 {os.path.basename(sample_file)}: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("学生行为数据扰动程序（安全版本）")
    print("此程序将直接覆盖 data/ 目录下的原始数据文件")
    print("特点：避免维度问题，自动修复损坏文件")
    print("=" * 60)
    
    # 执行扰动
    perturb_and_overwrite_safe('data')
    
    print("\n" + "=" * 60)
    print("扰动完成！")
    print("程序已自动处理维度不匹配问题，并为错误文件生成了替代数据。")
    print("现在数据包含更多真实世界的噪声和变化。")
    print("=" * 60)

if __name__ == "__main__":
    main()