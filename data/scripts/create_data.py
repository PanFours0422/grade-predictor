import pandas as pd
import numpy as np
import random
import os

# 生成200条测试数据
n_samples = 200

# 生成随机姓名（至少30个常用姓氏）
first_names = ['张', '王', '李', '赵', '刘', '陈', '杨', '黄', '周', '吴',
               '郑', '孙', '马', '朱', '胡', '林', '郭', '何', '高', '罗',
               '郝', '邓', '肖', '秦', '唐', '许', '韩', '冯', '邹', '魏']

# 生成更多的名字组合（至少30个常用字）
name_chars = ['小', '大', '明', '华', '强', '伟', '文', '勇', '静', '秀',
              '志', '国', '建', '军', '平', '杰', '超', '海', '龙', '飞',
              '智', '博', '宇', '浩', '宏', '鑫', '磊', '雷', '雨', '阳']

# 生成较长的名字
def generate_long_name():
    name = random.choice(first_names)
    # 添加2-3个字的名字
    name_length = random.randint(2, 3)
    for _ in range(name_length):
        name += random.choice(name_chars)
    return name

# 生成名字列表
names = [generate_long_name() for _ in range(n_samples)]

# 生成数据
data = {
    '学生姓名': names,
    '课程音视频(100%)': np.random.randint(0, 20, n_samples),
    '章节学习次数(100%)': np.random.randint(0, 60, n_samples),
    '作业(100%)': np.random.randint(0, 60, n_samples),
    '签到(100%)': np.random.randint(10, 50, n_samples),
    '课程互动(100%)': np.random.randint(0, 10, n_samples)
}

# 创建DataFrame
df = pd.DataFrame(data)

# 确保raw目录存在
os.makedirs('../processed', exist_ok=True)

# 保存到data文件夹
df.to_excel('../processed/test_data.xlsx', index=False)

print("测试数据已生成到 ../processed/test_data.xlsx")

# 打印前几条数据以供检查
print("\n生成的前20条数据：")
print(df.head(20))