import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def collect_data(root_folders):
    all_data = []
    for root_folder in root_folders:
    # 遍历所有以RO开头的文件夹
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for dirname in dirnames:
                if dirname.startswith('WO10'):
                    data_file = os.path.join(dirpath, dirname, 'data_hint2_.xlsx')
                    if os.path.exists(data_file):
                        df = pd.read_excel(data_file)
                        all_data.append(df)

    # 合并所有数据
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


def regression_train(df):
    # 提取参数和目标变量
    X = df.drop(columns=['latency'])  # 去掉'latency'列，剩下的是参数列
    y = df['latency']  # 提取'latency'列作为目标变量

    # 数据预处理，比如处理缺失值和数据转换
    # 这里可以添加你的数据清洗和准备步骤

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化并训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 评估模型性能
    mse = mean_squared_error(y_test, y_pred)
    print(f"均方误差 (MSE): {mse}")

    r2_score = model.score(X_test, y_test)
    print(f"R²分数: {r2_score}")

def RandomForestRegresso_train(df):
    # 提取参数和目标变量
    X = df.drop(columns=['latency'])  # 去掉'latency'列，剩下的是参数列
    y = df['latency']  # 提取'latency'列作为目标变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_rf_pred = rf_model.predict(X_test)

    # 评估模型性能
    rf_mse = mean_squared_error(y_test, y_rf_pred)
    rf_r2_score = rf_model.score(X_test, y_test)

    print(f"随机森林均方误差 (MSE): {rf_mse}")
    print(f"随机森林R²分数: {rf_r2_score}")

    return rf_model

if __name__ == '__main__':
    #root_folder1 = Path('../../results/mysqlResult/heavy') # 请替换为您的实际路径
    root_folder2 = Path('../../results/pgResult')
    #df = collect_data([root_folder1, root_folder2])
    df = collect_data([root_folder2])

    if df.empty:
        print("没有找到任何数据文件")
    else:
        print(f"收集到的数据样本数: {len(df)}")

    if not df.empty:
        model = RandomForestRegresso_train(df)
        joblib_file = "../../results/estimate_models/random_forest_model_pg.pkl"
        joblib.dump(model, joblib_file)

    joblib_file = "../../results/estimate_models/random_forest_model_pg.pkl"
    loaded_rf_model = joblib.load(joblib_file)
    #
    # # 使用模型进行预测
    confs = pd.DataFrame({
        'work_mem': [4096],
        'wal_buffers': [512],
        'temp_buffers': [1024],
        'shared_buffers': [16384],
        'effective_cache_size': [524288],
        'maintenance_work_mem': [65536],
        'max_connections': [100],
        'bgwriter_lru_multiplier': [2],
        'backend_flush_after': [0],
        'bgwriter_delay': [200],
        'max_parallel_workers': [8],
        'hash_mem_multiplier': [2],
        'checkpoint_flush_after': [32],
        'max_wal_size': [1024],
        'join_collapse_limit': [8],
        'vacuum_cost_page_dirty': [20],
        'min_parallel_table_scan_size': [1024],
        'min_parallel_index_scan_size': [64],
        'max_parallel_workers_per_gather': [2]
    })
    #
    predicted_latency = loaded_rf_model.predict(confs)
    print(f"Predicted Latency: {int(predicted_latency[0])}")


