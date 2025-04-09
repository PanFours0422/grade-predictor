import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.stats import randint, uniform

# 忽略警告
warnings.filterwarnings('ignore')

# 创建输出目录
os.makedirs('../../model/tuning_results', exist_ok=True)

def load_and_prepare_data(file_path='../../data/raw/chaoxing-data.xlsx', test_size=0.2, random_state=42):
    """
    Load data and prepare training and testing sets
    """
    print(f"Loading data: {file_path}")
    df = pd.read_excel(file_path)

    # 提取特征和目标变量
    features = ['课程音视频(100%)', '章节学习次数(100%)', '作业(100%)', '签到(100%)', '课程互动(100%)']
    X = df[features]
    y = df['综合成绩'] if '综合成绩' in df.columns else None

    # 使用众数填充缺失值
    X_filled = X.fillna(X.mode().iloc[0])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_filled, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, X_filled

def tune_xgboost_grid(X_train, X_test, y_train, y_test):
    """
    Use grid search to tune XGBoost hyperparameters
    """
    print("\n=== XGBoost Grid Search Parameter Tuning ===")

    # 定义参数网格
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }

    # 简化网格以加快测试速度
    simple_param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # 初始化XGBoost回归器
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    print("Running grid search...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=simple_param_grid,  # 使用简化参数网格
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1  # 使用所有CPU核心
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = np.sqrt(-grid_search.best_score_)  # 转换为RMSE

    print(f"\nBest parameters: {best_params}")
    print(f"Best cross-validation RMSE: {best_score:.4f}")

    # 使用最佳参数训练模型
    best_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **best_params
    )

    best_model.fit(X_train, y_train)

    # 在测试集上评估
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Test set RMSE: {rmse:.4f}")

    # 保存结果
    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv('../../model/tuning_results/xgboost_grid_search_results.csv', index=False)

    # 可视化参数重要性
    param_importance = {}
    for param in simple_param_grid.keys():
        param_values = []
        param_scores = []

        for idx, row in results.iterrows():
            param_values.append(row[f'param_{param}'])
            param_scores.append(-row['mean_test_score'])  # 转换为正MSE

        param_importance[param] = np.std(param_scores)

    # 归一化参数重要性
    total = sum(param_importance.values())
    param_importance = {k: v/total for k, v in param_importance.items()}

    plt.figure(figsize=(10, 6))
    plt.bar(param_importance.keys(), param_importance.values())
    plt.title('Parameter Importance')
    plt.ylabel('Importance')
    plt.savefig('../../model/tuning_results/xgboost_param_importance.png')

    return best_model, best_params

def tune_xgboost_random(X_train, X_test, y_train, y_test, n_iter=100):
    """
    Use random search to tune XGBoost hyperparameters
    """
    print("\n=== XGBoost Random Search Parameter Tuning ===")

    # 定义参数空间
    param_dist = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(50, 500),
        'min_child_weight': randint(1, 6),
        'gamma': uniform(0, 0.5),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }

    # 初始化XGBoost回归器
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    print(f"Running random search ({n_iter} iterations)...")
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_score = np.sqrt(-random_search.best_score_)  # 转换为RMSE

    print(f"\nBest parameters: {best_params}")
    print(f"Best cross-validation RMSE: {best_score:.4f}")

    # 使用最佳参数训练模型
    best_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **best_params
    )

    best_model.fit(X_train, y_train)

    # 在测试集上评估
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Test set RMSE: {rmse:.4f}")

    # 保存结果
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv('../../model/tuning_results/xgboost_random_search_results.csv', index=False)

    return best_model, best_params

def tune_kmeans_comprehensive(X):
    """
    Comprehensive optimization for K-means clustering
    """
    print("\n=== K-means Comprehensive Clustering Optimization ===")

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 尝试不同的聚类数
    k_range = range(2, 15)

    # 不同评估指标的结果
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    inertia_values = []

    print("Evaluating different numbers of clusters...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)

        labels = kmeans.labels_
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        inertia = kmeans.inertia_

        silhouette_scores.append(silhouette)
        calinski_scores.append(calinski)
        davies_bouldin_scores.append(davies_bouldin)
        inertia_values.append(inertia)

        print(f"Clusters {k}: Silhouette={silhouette:.4f}, CH={calinski:.1f}, DB={davies_bouldin:.4f}, Inertia={inertia:.1f}")

    # 创建评估指标可视化
    plt.figure(figsize=(16, 12))

    # 轮廓系数 (越高越好)
    plt.subplot(2, 2, 1)
    plt.plot(k_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score (Higher is Better)')
    plt.grid(True)

    # Calinski-Harabasz指数 (越高越好)
    plt.subplot(2, 2, 2)
    plt.plot(k_range, calinski_scores, 'go-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Score (Higher is Better)')
    plt.grid(True)

    # Davies-Bouldin指数 (越低越好)
    plt.subplot(2, 2, 3)
    plt.plot(k_range, davies_bouldin_scores, 'ro-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Score (Lower is Better)')
    plt.grid(True)

    # 惯性 (找拐点)
    plt.subplot(2, 2, 4)
    plt.plot(k_range, inertia_values, 'mo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (SSE)')
    plt.title('Elbow Method')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('../../model/tuning_results/kmeans_evaluation_metrics.png')

    # 寻找最佳聚类数
    best_k_silhouette = k_range[np.argmax(silhouette_scores)]
    best_k_calinski = k_range[np.argmax(calinski_scores)]
    best_k_davies = k_range[np.argmin(davies_bouldin_scores)]

    print("\nBest Number of Clusters:")
    print(f"Based on Silhouette Score: {best_k_silhouette}")
    print(f"Based on Calinski-Harabasz Score: {best_k_calinski}")
    print(f"Based on Davies-Bouldin Score: {best_k_davies}")
    print("Based on Elbow Method: Please check the graph for the elbow point")

    # 可视化聚类结果 (使用PCA降维到2D)
    try:
        from sklearn.decomposition import PCA

        # 对现有业务使用5个聚类
        k_final = 5
        kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
        labels = kmeans_final.fit_predict(X_scaled)

        # 使用PCA降维到2D进行可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')

        # 绘制聚类中心
        centers = kmeans_final.cluster_centers_
        centers_pca = pca.transform(centers)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

        plt.title('K-means Clustering Visualization (PCA Dimension Reduction)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.savefig('../../model/tuning_results/kmeans_cluster_visualization.png')

        # 分析各聚类的特征分布
        X_df = pd.DataFrame(X)
        X_df['cluster'] = labels

        cluster_stats = X_df.groupby('cluster').agg(['mean', 'std'])
        cluster_stats.to_csv('../../model/tuning_results/kmeans_cluster_statistics.csv')

        # 可视化各聚类的特征分布
        plt.figure(figsize=(14, 10))
        for i, feature in enumerate(X.columns):
            plt.subplot(3, 2, i+1)
            for cluster in range(k_final):
                sns.kdeplot(X_df[X_df['cluster'] == cluster][feature], label=f'Cluster {cluster}')
            plt.title(f'Feature Distribution: {feature}')
            plt.legend()

        plt.tight_layout()
        plt.savefig('../../model/tuning_results/kmeans_feature_distributions.png')

    except Exception as e:
        print(f"可视化过程出错: {e}")

    # 返回5个聚类的模型 (与业务逻辑保持一致)
    final_model = KMeans(n_clusters=5, random_state=42, n_init=10)
    final_model.fit(X_scaled)

    return final_model, scaler

def main():
    """
    Main function
    """
    print("\n=== Grade Prediction System Hyperparameter Tuning ===")

    # 加载数据
    try:
        X_train, X_test, y_train, y_test, X_full = load_and_prepare_data()
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # XGBoost调优
    try:
        print("\nStarting XGBoost tuning...")

        # 网格搜索调优
        xgb_model_grid, best_params_grid = tune_xgboost_grid(X_train, X_test, y_train, y_test)

        # 随机搜索调优 (可选，计算资源充足时使用)
        # xgb_model_random, best_params_random = tune_xgboost_random(X_train, X_test, y_train, y_test, n_iter=50)

        # 保存最终模型
        xgb_model_grid.save_model('../../model/xgboost_regression_model_tuned.model')
        print("XGBoost model saved to: ../../model/xgboost_regression_model_tuned.model")

        # 将最佳参数保存到文件
        with open('../../model/tuning_results/best_xgboost_params.txt', 'w') as f:
            f.write("Best parameters from grid search:\n")
            for param, value in best_params_grid.items():
                f.write(f"{param}: {value}\n")

    except Exception as e:
        print(f"XGBoost tuning failed: {e}")

    # K-means调优
    try:
        print("\nStarting K-means tuning...")
        kmeans_model, scaler = tune_kmeans_comprehensive(X_full)

        # 保存最终模型和缩放器
        joblib.dump(kmeans_model, '../../model/kmeans_model_tuned.pkl')
        joblib.dump(scaler, '../../model/scaler_tuned.pkl')
        print("K-means model saved to: ../../model/kmeans_model_tuned.pkl")
        print("Scaler saved to: ../../model/scaler_tuned.pkl")

    except Exception as e:
        print(f"K-means tuning failed: {e}")

    print("\nHyperparameter tuning completed! Please check the result files in '../../model/tuning_results/' directory.")

if __name__ == "__main__":
    main()