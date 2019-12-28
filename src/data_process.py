# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


def data_process():
    # read data
    cor = pd.read_csv("../data/cor.csv", encoding="unicode_escape")
    fea = cor[cor['sig'] == '***']['Unnamed: 0'].tolist()
    test = pd.read_csv("../data/test.csv", usecols=fea)
    test['target'] = 100
    test['flag'] = False
    fea.append("target")
    train = pd.read_csv("../data/train.csv", usecols=fea)
    train['flag'] = True
    sub = pd.read_csv('../data/sample_submission.csv')
    # data combine
    data = train.append(test)
    # anomaly detection
    data = remove_outlier(data, fea)
    train = data[data['flag'] == True]
    target = train.target
    train.drop(['flag', 'target'], axis=1, inplace=True)
    test.drop(['flag', 'target'], axis=1, inplace=True)
    data = train.append(test)
    scale = MinMaxScaler(feature_range=(-1, 1))
    data = pd.DataFrame(scale.fit_transform(data))
    test = data.iloc[-200000:, :]
    print("model")
    model = RandomForestRegressor(n_estimators=100, max_depth=100, n_jobs=-1)
    model.fit(train, target)
    result = model.predict(test)
    sub['target'] = result
    sub.to_csv('result.csv', index=False)
    print("over")


def remove_outlier(data: pd.DataFrame, usecols: list):
    # 三分位点的值+一三分位点的差值*1.5 (四分位)
    quartile_value = data.quantile(0.75) + (data.quantile(0.75) - data.quantile(0.25)) * 1.5
    # Z分数法
    z_max = data.mean() + 3 * data.std()
    z_min = data.mean() - 3 * data.std()
    for _ in usecols:
        data = data[(data[_] < quartile_value[_]) & (z_min[_] < data[_]) & (data[_] < z_max[_])]
    return data


data_process()
