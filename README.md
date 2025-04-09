# 成绩预测系统

## 项目介绍

本系统是一个基于机器学习的学生成绩预测平台，通过分析学生的学习行为数据（如课程音视频观看、章节学习次数、作业完成情况、签到记录和课程互动等指标），预测学生的最终综合成绩和学习投入程度。系统旨在帮助教师提前发现学习困难的学生，及时进行干预和指导。

## 核心功能

1. **单个学生成绩预测**：输入单个学生的学习行为数据，预测其最终成绩和学习投入程度
2. **批量成绩预测**：通过上传Excel文件进行批量学生成绩预测
3. **学习投入度分析**：将学生分为低投入性、中低投入性、中投入性、中高投入性和高投入性五个等级
4. **成绩等级评定**：根据预测分数将学生成绩划分为不及格、及格、良、优四个等级
5. **数据可视化**：直观展示学生成绩和投入度分布情况

## 技术架构

### 前端
- HTML/CSS/JavaScript
- Bootstrap框架
- Echarts图表库

### 后端
- Flask Web框架
- 数据处理：Pandas, NumPy
- 机器学习算法：
  - XGBoost回归（成绩预测）
  - K-means聚类（学习投入度分析）
- 数据可视化：Matplotlib, Seaborn
- 模型存储：Joblib

## 系统要求

- Python 3.6+
- 安装requirements.txt中的依赖包

## 安装步骤

1. 克隆代码仓库
```bash
git clone <仓库地址>
cd grade-predictor
```

2. 创建并激活虚拟环境（可选）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 运行应用
```bash
python app.py
```

5. 在浏览器中访问 http://127.0.0.1:5000/ 使用系统

## 目录结构

```
grade-predictor/
├── app.py                 # 主应用程序文件
├── requirements.txt       # 项目依赖
├── data/                  # 数据目录
│   └── raw/               # 原始数据
├── model/                 # 模型目录
│   ├── train/             # 模型训练脚本
│   ├── kmeans_model_tuned.pkl      # K-means聚类模型
│   ├── scaler_tuned.pkl            # 标准化处理器
│   └── xgboost_regression_model_tuned.model  # XGBoost回归模型
├── static/                # 静态资源
│   ├── css/               # 样式表
│   ├── js/                # JavaScript文件
│   └── images/            # 图片资源
├── templates/             # HTML模板
│   ├── index.html         # 首页
│   ├── single.html        # 单个学生预测页面
│   └── batch.html         # 批量预测页面
└── front/                 # 前端资源
    └── simhei.ttf         # 中文字体文件
```

## 模型说明

### XGBoost回归模型
用于预测学生的综合成绩，基于学习行为数据进行训练，经过网格搜索和交叉验证优化参数。

### K-means聚类模型
用于分析学生的学习投入度，将学生分为5个投入度等级，并基于聚类中心与学生特征的差异生成个性化解释。

## 使用说明

### 单个学生预测
1. 打开单个学生预测页面
2. 输入学生姓名和各项学习行为数据
3. 点击"预测"按钮获取成绩预测结果和投入度分析

### 批量学生预测
1. 打开批量预测页面
2. 下载模板Excel文件
3. 按模板格式填写多个学生的数据
4. 上传Excel文件进行批量预测
5. 查看预测结果并可下载结果报告

## 贡献与开发
欢迎贡献代码或提出改进建议！如需参与开发，请按以下步骤操作：

1. Fork本仓库
2. 创建你的功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的更改 (`git commit -m '添加某某功能'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建一个Pull Request

## 许可证
[请根据项目实际情况添加许可证] 
