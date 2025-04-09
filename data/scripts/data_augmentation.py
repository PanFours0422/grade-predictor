import pandas as pd
import numpy as np
import random
import os
from sklearn.utils import resample

def generate_synthetic_data(original_df, n_synthetic_samples=500):
    """
    通过对原始数据进行抽样和添加噪声生成合成数据
    
    参数:
    - original_df: 原始数据DataFrame
    - n_synthetic_samples: 要生成的合成样本数量
    
    返回:
    - 合成数据DataFrame
    """
    # 特征列
    feature_cols = ['课程音视频(100%)', '章节学习次数(100%)', '作业(100%)', '签到(100%)', '课程互动(100%)']
    
    # 确保原始数据中有这些列
    missing_cols = [col for col in feature_cols if col not in original_df.columns]
    if missing_cols:
        raise ValueError(f"原始数据中缺少以下列: {missing_cols}")
    
    # 创建合成数据样本
    synthetic_rows = []
    
    # 对原始数据进行有放回抽样
    for _ in range(n_synthetic_samples):
        # 随机选择一个原始样本
        random_idx = random.randint(0, len(original_df) - 1)
        original_row = original_df.iloc[random_idx]
        
        # 创建新合成样本
        synthetic_sample = {}
        
        # 如果有学生姓名列，则创建一个新名字
        if '学生姓名' in original_df.columns:
            first_names = ['张', '王', '李', '赵', '刘', '陈', '杨', '黄', '周', '吴',
                           '郑', '孙', '马', '朱', '胡', '林', '郭', '何', '高', '罗']
            name_chars = ['小', '大', '明', '华', '强', '伟', '文', '勇', '静', '秀',
                          '志', '国', '建', '军', '平', '杰', '超', '海', '龙', '飞']
            
            first_name = random.choice(first_names)
            name_length = random.randint(1, 2)
            full_name = first_name + ''.join(random.choices(name_chars, k=name_length))
            synthetic_sample['学生姓名'] = full_name
        
        # 为每个特征添加随机变化
        for col in feature_cols:
            original_value = original_row[col]
            
            # 根据原始值的大小添加不同程度的噪声
            if original_value > 0:
                noise_factor = 0.2  # 20%的噪声
                noise = original_value * noise_factor * (random.random() * 2 - 1)  # -noise_factor到+noise_factor之间
                new_value = max(0, original_value + noise)  # 确保值不小于0
                synthetic_sample[col] = round(new_value, 2)
            else:
                # 对于原值为0的情况
                if random.random() < 0.2:  # 20%的概率变为非0
                    synthetic_sample[col] = round(random.random() * 10, 2)
                else:
                    synthetic_sample[col] = 0
        
        # 如果有综合成绩列，需要根据特征值计算一个合理的成绩
        if '综合成绩' in original_df.columns:
            # 这里我们可以简单地使用线性组合来模拟成绩
            audio_weight = 0.2
            learning_weight = 0.2
            assignment_weight = 0.3
            sign_weight = 0.2
            interact_weight = 0.1
            
            synthetic_score = (
                audio_weight * synthetic_sample['课程音视频(100%)'] +
                learning_weight * synthetic_sample['章节学习次数(100%)'] +
                assignment_weight * synthetic_sample['作业(100%)'] +
                sign_weight * synthetic_sample['签到(100%)'] +
                interact_weight * synthetic_sample['课程互动(100%)']
            )
            
            # 添加一些随机性
            noise = synthetic_score * 0.1 * (random.random() * 2 - 1)
            synthetic_score += noise
            
            # 确保成绩在合理范围内
            synthetic_score = max(0, min(100, synthetic_score))
            synthetic_sample['综合成绩'] = round(synthetic_score, 2)
        
        synthetic_rows.append(synthetic_sample)
    
    # 创建合成数据DataFrame
    synthetic_df = pd.DataFrame(synthetic_rows)
    
    return synthetic_df

def balance_data_by_grades(df, target_col='综合成绩', n_samples_per_grade=None):
    """
    平衡不同成绩等级的数据
    
    参数:
    - df: 原始数据DataFrame
    - target_col: 目标列（成绩列）
    - n_samples_per_grade: 每个等级的样本数量，如果为None则使用数量最多的等级的样本数
    
    返回:
    - 平衡后的DataFrame
    """
    if target_col not in df.columns:
        raise ValueError(f"数据集中没有'{target_col}'列")
    
    # 根据分位数定义成绩等级
    quantiles = [0, 0.1, 0.5, 0.9, 1]
    boundaries = df[target_col].quantile(quantiles).values
    
    # 定义等级标签
    df['grade_label'] = pd.cut(
        df[target_col], 
        bins=boundaries, 
        labels=['不及格', '及格', '良', '优'],
        include_lowest=True
    )
    
    # 获取每个等级的数据
    grade_dfs = {grade: df[df['grade_label'] == grade] for grade in ['不及格', '及格', '良', '优']}
    
    # 确定每个等级要采样的数量
    if n_samples_per_grade is None:
        n_samples_per_grade = max(len(grade_df) for grade_df in grade_dfs.values())
    
    # 对每个等级进行上采样或下采样
    balanced_dfs = []
    for grade, grade_df in grade_dfs.items():
        if len(grade_df) == 0:
            continue
            
        if len(grade_df) < n_samples_per_grade:
            # 上采样
            resampled_df = resample(
                grade_df,
                replace=True,
                n_samples=n_samples_per_grade,
                random_state=42
            )
        elif len(grade_df) > n_samples_per_grade:
            # 下采样
            resampled_df = resample(
                grade_df,
                replace=False,
                n_samples=n_samples_per_grade,
                random_state=42
            )
        else:
            resampled_df = grade_df
            
        balanced_dfs.append(resampled_df)
    
    # 合并所有等级的数据
    balanced_df = pd.concat(balanced_dfs)
    
    # 移除临时等级标签列
    balanced_df = balanced_df.drop(columns=['grade_label'])
    
    return balanced_df

def main():
    # 确保数据目录存在
    if not os.path.exists('../processed'):
        os.makedirs('../processed')
    
    # 加载原始数据
    try:
        original_data_path = '../raw/chaoxing-data.xlsx'
        print(f"加载原始数据: {original_data_path}")
        original_df = pd.read_excel(original_data_path)
        print(f"原始数据形状: {original_df.shape}")
    except Exception as e:
        print(f"加载原始数据出错: {e}")
        return
    
    # 生成合成数据
    try:
        print("\n生成合成数据...")
        synthetic_df = generate_synthetic_data(original_df, n_synthetic_samples=500)
        print(f"合成数据形状: {synthetic_df.shape}")
        
        # 合并原始数据和合成数据
        combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
        print(f"合并后数据形状: {combined_df.shape}")
    except Exception as e:
        print(f"生成合成数据出错: {e}")
        return
    
    # 平衡不同成绩等级的数据
    try:
        print("\n平衡不同成绩等级的数据...")
        balanced_df = balance_data_by_grades(combined_df)
        print(f"平衡后数据形状: {balanced_df.shape}")
        
        # 检查平衡后各等级的分布
        if '综合成绩' in balanced_df.columns:
            quantiles = [0, 0.1, 0.5, 0.9, 1]
            boundaries = original_df['综合成绩'].quantile(quantiles).values
            balanced_df['temp_grade'] = pd.cut(
                balanced_df['综合成绩'], 
                bins=boundaries, 
                labels=['不及格', '及格', '良', '优'],
                include_lowest=True
            )
            
            grade_counts = balanced_df['temp_grade'].value_counts()
            print("\n各等级样本数量:")
            print(grade_counts)
            
            balanced_df = balanced_df.drop(columns=['temp_grade'])
    except Exception as e:
        print(f"平衡数据出错: {e}")
        balanced_df = combined_df  # 如果平衡失败，使用未平衡的数据
    
    # 保存增强后的数据
    try:
        augmented_data_path = '../processed/augmented_data.xlsx'
        balanced_df.to_excel(augmented_data_path, index=False)
        print(f"\n增强后的数据已保存到: {augmented_data_path}")
    except Exception as e:
        print(f"保存数据出错: {e}")
    
if __name__ == "__main__":
    main() 