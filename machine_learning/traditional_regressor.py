import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse


def regression_algorithm(model, name, x_train, y_train, x_test, y_test, mse_dict, score_dict):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse_dict[name] = mse(y_pred, y_test)
    score_dict[name] = model.score(x_test, y_test) * 100
    print(f"Mean squared error({name}): {mse_dict[name]:.2f}")
    print(f"Score({name}): {score_dict[name]:.2f}%\n")


# 使用波士顿房价预测数据集测试传统机器学习模型对于回归问题的效果
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
sc, models, mse_dict, score_dict = StandardScaler(), {}, {}, {}

# 线性回归算法
models["LinearRegression"] = LinearRegression()
regression_algorithm(models["LinearRegression"], "LinearRegression", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# 岭回归算法
models["Ridge"] = Ridge()
regression_algorithm(models["Ridge"], "Ridge", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# Lasso回归算法
models["Lasso"] = Lasso()
regression_algorithm(models["Lasso"], "Lasso", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# SGD回归算法
models["SGDRegressor"] = SGDRegressor()
regression_algorithm(models["SGDRegressor"], "SGDRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# KNN回归算法
models["KNN"] = KNeighborsRegressor()
regression_algorithm(models["KNN"], "KNN", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# 线性SVM回归算法
models["LinearSVR"] = LinearSVR(dual='auto')
regression_algorithm(models["LinearSVR"], "LinearSVR", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# 核方法SVM回归算法
models["SVR"] = SVR()
regression_algorithm(models["SVR"], "SVR", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# 朴素贝叶斯岭回归算法
models["BayesianRidge"] = BayesianRidge()
regression_algorithm(models["BayesianRidge"], "BayesianRidge", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# 决策树回归算法
models["DecisionTreeRegressor"] = DecisionTreeRegressor()
regression_algorithm(models["DecisionTreeRegressor"], "DecisionTreeRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# 随机森林回归算法
models["RandomForestRegressor"] = RandomForestRegressor()
regression_algorithm(models["RandomForestRegressor"], "RandomForestRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# 极端随机森林回归算法
models["ExtraTreesRegressor"] = ExtraTreesRegressor()
regression_algorithm(models["ExtraTreesRegressor"], "ExtraTreesRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# Bootstrap Aggregating(套袋法)回归算法
models["BaggingRegressor"] = BaggingRegressor()
regression_algorithm(models["BaggingRegressor"], "BaggingRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# AdaBoost(自适应权重分配)回归算法
models["AdaBoostRegressor"] = AdaBoostRegressor()
regression_algorithm(models["AdaBoostRegressor"], "AdaBoostRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# 梯度提升回归算法
models["GradientBoosting"] = GradientBoostingRegressor()
regression_algorithm(models["GradientBoosting"], "GradientBoosting", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# XGBoost回归算法
models["XGBRegressor"] = XGBRegressor()
regression_algorithm(models["XGBRegressor"], "XGBRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# CatBoost回归算法
models["CatBoostRegressor"] = CatBoostRegressor(iterations=10)
regression_algorithm(models["CatBoostRegressor"], "CatBoostRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# LightGBM回归算法
models["LGBMRegressor"] = LGBMRegressor(verbose=-1)
regression_algorithm(models["LGBMRegressor"], "LGBMRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# stacking堆叠回归算法
models["StackingRegressor"] = StackingRegressor(estimators=[(
    "XGBRegressor", models["XGBRegressor"]), ("RandomForestRegressor", models["RandomForestRegressor"])])
regression_algorithm(models["StackingRegressor"], "StackingRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# Voting投票回归算法
models["VotingRegressor"] = VotingRegressor(estimators=[(
    "GradientBoosting", models["GradientBoosting"]), ("RandomForestRegressor", models["RandomForestRegressor"])])
regression_algorithm(models["VotingRegressor"], "VotingRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, mse_dict, score_dict)

# 绘制模型性能图表
df_model = pd.DataFrame(index=models.keys(), columns=[
                        'Mean squared error', 'Score'])
df_model['Mean squared error'] = mse_dict.values()
df_model['Score'] = score_dict.values()
ax = df_model.plot.barh()
plt.show()
