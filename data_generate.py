import numpy as np
import pandas as pd
import os
import random
import shutil

def generate_student_data(num_samples, risk_level, split='train'):
    """
    为指定风险等级和数据集划分生成学生行为数据
    risk_level: 0 (无风险) 或 1 (有风险)
    """
    data_dir = f'data/{risk_level}/{split}/'
    os.makedirs(data_dir, exist_ok=True)
    
    # 定义8个行为维度（行），这些维度反映学生学业表现的关键指标
    dimensions = [
        'attendance_rate',      # 出勤率
        'class_participation',  # 课堂参与度  
        'homework_completion',  # 作业完成率
        'homework_quality',     # 作业质量
        'quiz_performance',     # 小测验表现
        'teacher_interaction', # 与教师互动
        'study_hours',          # 学习时间
        'phone_usage'           # 手机使用频率
    ]
    
    # 定义8个时间步（列）- 代表一个学期的8周
    time_steps = 8
    
    for i in range(num_samples):
        # 根据风险等级设置不同的基础模式
        if risk_level == 0:  # 无风险 - 稳定的良好表现
            base_values = {
                'attendance_rate': np.random.uniform(0.85, 0.98),
                'class_participation': np.random.uniform(0.75, 0.95),
                'homework_completion': np.random.uniform(0.88, 1.0),
                'homework_quality': np.random.uniform(0.80, 0.95),
                'quiz_performance': np.random.uniform(0.75, 0.92),
                'teacher_interaction': np.random.uniform(0.40, 0.75),
                'study_hours': np.random.uniform(0.65, 0.90),
                'phone_usage': np.random.uniform(0.10, 0.30)
            }
            
            # 添加小幅正向趋势和随机噪声
            data = []
            for dim in dimensions:
                base = base_values[dim]
                # 70%的概率有小幅提升趋势
                trend = np.random.uniform(0.01, 0.03) if random.random() > 0.3 else 0
                noise = np.random.normal(0, 0.05, time_steps)
                sequence = np.clip(base + np.linspace(0, trend, time_steps) + noise, 0, 1)
                data.append(sequence)
                
        else:  # 有风险 - 表现持续下降
            base_values = {
                'attendance_rate': np.random.uniform(0.65, 0.80),
                'class_participation': np.random.uniform(0.30, 0.55),
                'homework_completion': np.random.uniform(0.50, 0.75),
                'homework_quality': np.random.uniform(0.35, 0.60),
                'quiz_performance': np.random.uniform(0.30, 0.55),
                'teacher_interaction': np.random.uniform(0.10, 0.25),
                'study_hours': np.random.uniform(0.20, 0.45),
                'phone_usage': np.random.uniform(0.60, 0.85)
            }
            
            # 强烈的负向趋势，带有现实的波动
            data = []
            for dim in dimensions:
                base = base_values[dim]
                # 为不同维度设置不同的下降速率
                decline_rates = {
                    'attendance_rate': 0.08,      # 出勤率下降最快
                    'class_participation': 0.06,  # 课堂参与度下降
                    'homework_completion': 0.07,  # 作业完成率下降
                    'homework_quality': 0.05,     # 作业质量下降
                    'quiz_performance': 0.09,     # 测验成绩下降最快
                    'teacher_interaction': 0.04,  # 与教师互动减少
                    'study_hours': 0.06,          # 学习时间减少
                    'phone_usage': 0.07           # 手机使用增加
                }
                decline = decline_rates[dim]
                
                # 生成负向趋势
                trend = -np.linspace(0, decline, time_steps)
                # 40%的概率在学期中期有改善尝试（现实中的挣扎）
                if random.random() > 0.4:
                    mid_point = time_steps // 2
                    # 从中期开始，下降速度减缓，模拟学生尝试努力
                    trend[mid_point:] = trend[mid_point] - np.linspace(0, decline * 0.6, time_steps - mid_point)
                
                noise = np.random.normal(0, 0.08, time_steps)
                sequence = np.clip(base + trend + noise, 0, 1)
                data.append(sequence)
        
        # 创建DataFrame并保存为CSV，不包含行索引（只有纯数字）
        df = pd.DataFrame(np.array(data))
        df.to_csv(f'{data_dir}student_{i:04d}.csv', header=False, index=False)
        
        if (i + 1) % 100 == 0:
            print(f'已生成 {i + 1}/{num_samples} 个样本，路径: {risk_level}/{split}')

def clean_previous_data():
    """清理之前生成的数据，确保重新生成时覆盖旧数据"""
    if os.path.exists('data'):
        print("清理之前的生成数据...")
        shutil.rmtree('data')
        print("旧数据已清理完成")

def main():
    """主函数：生成完整的学生行为数据集"""
    # 设置随机种子确保结果可重现
    np.random.seed(42)
    random.seed(42)
    
    # 清理之前的数据，确保覆盖
    clean_previous_data()
    
    # 为每个类别生成1000个样本
    print("开始生成无风险(0)训练数据...")
    generate_student_data(1000, risk_level=0, split='train')
    
    print("\n开始生成无风险(0)测试数据...")
    generate_student_data(1000, risk_level=0, split='test')
    
    print("\n开始生成有风险(1)训练数据...")
    generate_student_data(1000, risk_level=1, split='train')
    
    print("\n开始生成有风险(1)测试数据...")
    generate_student_data(1000, risk_level=1, split='test')
    
    print("\n数据生成完成！")
    print("总样本数: 4000")
    print("目录结构:")
    print("data/")
    print("├── 0/")
    print("│   ├── train/ (1000个文件)")
    print("│   └── test/ (1000个文件)")
    print("└── 1/")
    print("    ├── train/ (1000个文件)")
    print("    └── test/ (1000个文件)")
    
    # 验证生成结果
    print("\n验证生成结果:")
    for risk in [0, 1]:
        for split in ['train', 'test']:
            dir_path = f'data/{risk}/{split}/'
            file_count = len([f for f in os.listdir(dir_path) if f.endswith('.csv')])
            print(f"{dir_path}: {file_count} 个文件")

if __name__ == "__main__":
    main()