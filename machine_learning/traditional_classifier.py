import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score


def classifier_algorithm(model, name, x_train, y_train, x_test, y_test, as_dict, ps_dict, rs_dict):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    as_dict[name] = accuracy_score(y_pred, y_test) * 100
    ps_dict[name] = precision_score(y_pred, y_test) * 100
    rs_dict[name] = recall_score(y_pred, y_test) * 100
    print(f"Accuracy score({name}): {as_dict[name]:.2f}")
    print(f"Precision score({name}): {ps_dict[name]:.2f}")
    print(f"Recall score({name}): {rs_dict[name]:.2f}%\n")


# 使用sklearn中的类别数据集测试传统机器学习模型对于分类问题的效果
breast_cancer_datas = load_breast_cancer()
x_datas, y_datas = breast_cancer_datas.data, breast_cancer_datas.target
x_train, x_test, y_train, y_test = train_test_split(
    x_datas, y_datas, test_size=0.3, random_state=0)
sc, models, as_dict, ps_dict, rs_dict = StandardScaler(), {}, {}, {}, {}

# 逻辑回归分类算法
models["LogisticRegression"] = LogisticRegression()
classifier_algorithm(models["LogisticRegression"], "LogisticRegression", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# SGD分类算法
models["SGDClassifier"] = SGDClassifier()
classifier_algorithm(models["SGDClassifier"], "SGDClassifier", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# KNN分类算法
models["KNN"] = KNeighborsClassifier(n_neighbors=3)
classifier_algorithm(models["KNN"], "KNN", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# 线性SVM分类算法
models["LinearSVC"] = LinearSVC(dual='auto')
classifier_algorithm(models["LinearSVC"], "LinearSVC", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# 核方法SVM分类算法
models["SVC"] = SVC()
classifier_algorithm(models["SVC"], "SVC", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# 朴素贝叶斯分类算法
models["GaussianNB"] = GaussianNB()
classifier_algorithm(models["GaussianNB"], "GaussianNB", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# 决策树分类算法
models["DecisionTreeClassifier"] = DecisionTreeClassifier()
classifier_algorithm(models["DecisionTreeClassifier"], "DecisionTreeClassifier", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# 随机森林分类算法
models["RandomForestClassifier"] = RandomForestClassifier()
classifier_algorithm(models["RandomForestClassifier"], "RandomForestClassifier", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# 极端随机森林分类算法
models["ExtraTreesClassifier"] = ExtraTreesClassifier()
classifier_algorithm(models["ExtraTreesClassifier"], "ExtraTreesRegressor", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# Bootstrap Aggregating(套袋法)分类算法
models["BaggingClassifier"] = BaggingClassifier()
classifier_algorithm(models["BaggingClassifier"], "BaggingClassifier", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# AdaBoost(自适应权重分配)分类算法
models["AdaBoostClassifier"] = AdaBoostClassifier(algorithm='SAMME')
classifier_algorithm(models["AdaBoostClassifier"], "AdaBoostClassifier", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# 梯度提升分类算法
models["GradientBoosting"] = GradientBoostingClassifier()
classifier_algorithm(models["GradientBoosting"], "GradientBoosting", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# XGBoost分类算法
models["XGBClassifier"] = XGBClassifier()
classifier_algorithm(models["XGBClassifier"], "XGBClassifier", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# CatBoost分类算法
models["CatBoostClassifier"] = CatBoostClassifier(iterations=10)
classifier_algorithm(models["CatBoostClassifier"], "CatBoostClassifier", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# LightGBM分类算法
models["LGBMClassifier"] = LGBMClassifier(verbose=-1)
classifier_algorithm(models["LGBMClassifier"], "LGBMClassifier", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# stacking堆叠分类算法
models["StackingClassifier"] = StackingClassifier(estimators=[(
    "XGBClassifier", models["XGBClassifier"]), ("RandomForestClassifier", models["RandomForestClassifier"])])
classifier_algorithm(models["StackingClassifier"], "StackingClassifier", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# Voting投票分类算法
models["VotingClassifier"] = VotingClassifier(estimators=[(
    "GradientBoosting", models["GradientBoosting"]), ("RandomForestClassifier", models["RandomForestClassifier"])])
classifier_algorithm(models["VotingClassifier"], "VotingClassifier", sc.fit_transform(
    x_train), y_train, sc.transform(x_test), y_test, as_dict, ps_dict, rs_dict)

# 绘制模型性能图表
df_model = pd.DataFrame(index=models.keys(), columns=[
                        'Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = as_dict.values()
df_model['Precision'] = ps_dict.values()
df_model['Recall'] = rs_dict.values()

ax = df_model.plot.barh()
ax.legend(
    ncol=len(models.keys()),
    bbox_to_anchor=(0, 1),
    loc='lower left',
    prop={'size': 15}
)
plt.show()
