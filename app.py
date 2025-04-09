from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
import json
import numpy as np
import re

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

warnings.filterwarnings("default", category=FutureWarning)


# 数据预处理
def preprocess_data(data):
    # 使用众数填充缺失值
    data_filled = data.fillna(data.mode().iloc[0])
    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_filled)
    return data_scaled


@app.route('/')
def index():
    return render_template('index.html')


# 成绩等级划分
def get_boundary_scores():
    # 从训练数据中获取成绩分界点
    file_path = "./data/raw/chaoxing-data.xlsx"
    train_df = pd.read_excel(file_path)
    percentages = [0, 0.1, 0.5, 0.9, 1]
    _grades = ['不及格', '及格', '良', '优']
    boundary_scores = train_df['综合成绩'].quantile(percentages)
    scores_list = boundary_scores.values.tolist()
    scores_list = [round(i, 2) for i in scores_list]
    return scores_list[1], scores_list[2], scores_list[3]  # 返回及格、良、优的分界点


def get_statistics(result_list):
    # 统计投入性分布
    investment_stats = {
        '低投入性': 0,
        '中低投入性': 0,
        '中投入性': 0,
        '中高投入性': 0,
        '高投入性': 0
    }

    # 统计成绩等级分布
    grade_stats = {
        '不及格': 0,
        '及格': 0,
        '良': 0,
        '优': 0
    }

    for result in result_list:
        investment_stats[result[6]] += 1
        grade_stats[result[9]] += 1

    return {
        'investment': investment_stats,
        'grade': grade_stats
    }


def generate_explanation(student_data, cluster_id, kmeans_model, feature_names):
    """
    生成投入性分类的解释，基于学生数据与聚类中心的比较

    Args:
        student_data: 学生的特征数据（已标准化）
        cluster_id: 学生所属的聚类ID
        kmeans_model: 训练好的K-means模型
        feature_names: 特征名称列表

    Returns:
        解释文本
    """
    # 获取聚类中心
    cluster_centers = kmeans_model.cluster_centers_
    student_cluster_center = cluster_centers[cluster_id]

    # 计算学生数据与聚类中心的差异
    differences = student_data[0] - student_cluster_center

    # 创建特征名称与差异的字典，并去除括号及括号内的内容
    feature_diff = {re.sub(r"\(.*?\)", "", feature): diff for feature, diff in zip(feature_names, differences)}

    # 按差异绝对值排序（找出差异最大的特征）
    sorted_features = sorted(feature_diff.items(), key=lambda x: abs(x[1]), reverse=True)

    # 确定主要影响因素（取前3个差异最大的特征）
    main_factors = sorted_features[:3]

    # 根据投入性级别生成不同的解释模板
    explanation = ""
    negative_features = []
    positive_features = []

    # 区分正向和负向特征
    for feature, diff in main_factors:
        if diff < -0.5:  # 显著低于聚类中心
            negative_features.append(feature)
        elif diff > 0.5:  # 显著高于聚类中心
            positive_features.append(feature)

    # 生成解释文本
    if negative_features:
        explanation += f"在 {', '.join(negative_features)} 方面低于该类别平均水平。"

    if positive_features:
        if explanation:
            explanation += f" 但在 {', '.join(positive_features)} 方面表现较好。"
        else:
            explanation += f"在 {', '.join(positive_features)} 方面表现较好。"

    if not explanation:
        explanation = "各指标与该类别平均水平接近。"

    return explanation


@app.route('/single', methods=['GET', 'POST'])
def single():
    if request.method == 'GET':
        return render_template('single.html')
    else:
        name = request.form['name']
        audio = float(request.form['audio'])
        learning_time = float(request.form['learning_time'])
        assignment = float(request.form['assignment'])
        sign_time = float(request.form['sign_time'])
        course_interact = float(request.form['course_interact'])

        data = {
            '学生姓名': [name],
            '课程音视频(100%)': [audio],
            '章节学习次数(100%)': [learning_time],
            '作业(100%)': [assignment],
            '签到(100%)': [sign_time],
            '课程互动(100%)': [course_interact]
        }
        df = pd.DataFrame(data)

        loaded_model = xgb.Booster()
        loaded_model.load_model('./model/xgboost_regression_model_tuned.model')
        dtrain = xgb.DMatrix(
            df[['课程音视频(100%)', '章节学习次数(100%)', '作业(100%)', '签到(100%)', '课程互动(100%)']])

        predictions = loaded_model.predict(dtrain)
        predict_list = predictions.tolist()
        predict_list = [round(i, 2) for i in predict_list]

        warnings.filterwarnings("ignore")

        kmeans = joblib.load('./model/kmeans_model_tuned.pkl')
        scaler = joblib.load('./model/scaler_tuned.pkl')

        feature_names = ['课程音视频(100%)', '章节学习次数(100%)', '作业(100%)', '签到(100%)', '课程互动(100%)']
        data_part = df[feature_names]
        data_filled = data_part.fillna(data_part.mode().iloc[0])
        new_data = scaler.transform(data_filled)
        predicted_cluster = kmeans.predict(new_data)

        # 生成投入性解释
        explanation = generate_explanation(
            new_data,
            predicted_cluster[0],
            kmeans,
            feature_names
        )

        cluster_label_mapping = {
            4: "高投入性",
            3: "中高投入性",
            2: "中投入性",
            1: "中低投入性",
            0: "低投入性"
        }

        invest_levels = [cluster_label_mapping[label] for label in predicted_cluster]
        df['投入性'] = invest_levels
        # 保留投入性解释数据，后续通过 tooltip 显示（前端将不再单独显示该列）
        df['投入性解释'] = [explanation]

        if '综合成绩' in df.columns:
            df.drop(columns=['综合成绩'], inplace=True)

        df['相对预测分数'] = predict_list
        score_1, score_2, score_3 = get_boundary_scores()
        df['等级'] = df['相对预测分数'].apply(
            lambda x: '不及格' if x < score_1 else '及格' if x < score_2 else '良' if x < score_3 else '优')

        head_list = df.columns.values.tolist()
        result_list = df.values.tolist()

        # 获取统计数据
        stats = get_statistics(result_list)

        return render_template('single.html',
                               head_list=head_list,
                               result_list=result_list,
                               stats=json.dumps(stats))


@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'GET':
        return render_template('batch.html')
    else:
        excle_file = request.files['file']
        df = pd.read_excel(excle_file)

        warnings.filterwarnings("ignore", category=FutureWarning)

        loaded_model = xgb.Booster()
        loaded_model.load_model('./model/xgboost_regression_model_tuned.model')

        indexs = ['课程音视频(100%)', '章节学习次数(100%)', '作业(100%)', '签到(100%)', '课程互动(100%)']
        for index in indexs:
            if index not in df.columns:
                df[index] = 0

        dtrain = xgb.DMatrix(
            df[['课程音视频(100%)', '章节学习次数(100%)', '作业(100%)', '签到(100%)', '课程互动(100%)']])
        predictions = loaded_model.predict(dtrain)
        predict_list = predictions.tolist()
        predict_list = [round(float(i), 2) for i in predict_list]

        warnings.filterwarnings("ignore")
        kmeans = joblib.load('./model/kmeans_model_tuned.pkl')
        scaler = joblib.load('./model/scaler_tuned.pkl')

        feature_names = ['课程音视频(100%)', '章节学习次数(100%)', '作业(100%)', '签到(100%)', '课程互动(100%)']
        data_part = df[feature_names]
        data_filled = data_part.fillna(data_part.mode().iloc[0])
        new_data = scaler.transform(data_filled)
        predicted_cluster = kmeans.predict(new_data)

        # 为每个学生生成投入性解释
        explanations = []
        for i, cluster_id in enumerate(predicted_cluster):
            explanation = generate_explanation(
                new_data[i:i + 1],
                cluster_id,
                kmeans,
                feature_names
            )
            explanations.append(explanation)

        cluster_label_mapping = {
            0: "高投入性",
            1: "中高投入性",
            2: "中投入性",
            3: "中低投入性",
            4: "低投入性"
        }

        invest_levels = [cluster_label_mapping[label] for label in predicted_cluster]
        df['投入性'] = invest_levels
        # 保留投入性解释数据，前端通过 tooltip 显示
        df['投入性解释'] = explanations

        df['相对预测分数'] = predict_list
        score_1, score_2, score_3 = get_boundary_scores()
        df['等级'] = df['相对预测分数'].apply(
            lambda x: '不及格' if x < score_1 else '及格' if x < score_2 else '良' if x < score_3 else '优')

        if '综合成绩' in df.columns:
            df.drop(columns=['综合成绩'], inplace=True)
        head_list = df.columns.values.tolist()
        result_list = df.values.tolist()

        # 获取统计数据
        stats = get_statistics(result_list)

        return render_template('batch.html',
                               head_list=head_list,
                               result_list=result_list,
                               stats=json.dumps(stats))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8189)
