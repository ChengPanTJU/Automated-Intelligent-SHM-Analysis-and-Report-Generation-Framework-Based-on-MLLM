"""
数据读取函数，并进行初步处理

Created on 2024

@author: Gong Fengzong
"""
import os
import warnings

import pandas as pd
import numpy as np


def Read_csv1(file_path):
    # 判断文件是否存在
    if not os.path.exists(file_path):
        return [], False
    # 读取文件内容，若出现乱码，则输出空文件
    try:
        # 监测 DtypeWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", pd.errors.DtypeWarning)
            # 读取 CSV 文件
            data = pd.read_csv(file_path, header=None, on_bad_lines="skip")

            # 如果捕获到 DtypeWarning，直接返回 NaN
            if any(issubclass(warning.category, pd.errors.DtypeWarning) for warning in w):
                return [], False
    except Exception as e:
        return [], False
    # 去除9999, 用nan填补缺失值
    data.replace(9999, np.nan, inplace=True)
    data.replace(0, np.nan, inplace=True)
    data.replace(999, np.nan, inplace=True)
    missing_ratio = data.isna().mean()
    cols_to_replace = missing_ratio[missing_ratio > 0.7].index
    cols_to_fill = missing_ratio[missing_ratio <= 0.7].index
    # 如果某一列缺失值大于一半，则将该列全部设置为nan
    data[cols_to_replace] = np.nan
    # 其余缺失用0填充
    for col in cols_to_fill:
        if pd.api.types.is_numeric_dtype(data[col]):
            # 去除 NaN 的有效值
            valid_values = data[col].dropna()

            # 计算有效值的均值
            mean_value = valid_values.mean()

            # 用均值填充 NaN 值，并重新赋值回列
            data[col] = data[col].fillna(mean_value)
    # data[cols_to_zero] = data[cols_to_zero].replace(np.nan, 0.0)

    return data, True



if __name__ == '__main__':
    datapath = 'G:/博士资料/监测数据自动化分析/data/2019-04-27 03-VIB.csv'
    data, file_exists = Read_csv1(datapath)
