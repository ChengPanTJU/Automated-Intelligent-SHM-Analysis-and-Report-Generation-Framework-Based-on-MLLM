"""
应变数据分析函数
要求读取数据文件得到的数据为pandas格式
函数输入为dataframe格式数据

Created on 2024

@author: Gong Fengzong
"""
import numpy as np
import pandas as pd

from DataRead import Read_file
from config import WdrConfig, OutputConfig
OutConfig = OutputConfig()
outputconfig = OutConfig.tasks

def Wdr_analyze(df_wdr, wdr_conf):
    """根据指定的任务执行相应的分析"""
    # 读取文件内容
    file_path = df_wdr['file_path']
    data, file_exists = Read_file.Read_csv1(file_path)

    # 分析各通道数据异常和正常情况，异常通道数据设置为NAN
    # data = data_preprocessing(data)
    # 任务与分析函数的映射
    analysis_functions = {
        'preprocess': Prep_WDR,  # 数据预处理函数
        'rms': rms,          # rms函数
        'mean': average,     # mean函数
    }
    results = {}
    # 如果文件不存在，则给各任务赋值为nan
    if not file_exists:
        for task, channels in wdr_conf.tasks_channels.items():
            results[task] = np.nan
        return pd.Series(results)

    for task, channels in wdr_conf.tasks_channels.items():

        if task in analysis_functions.keys():
            # 获取任务对应的分析函数
            analysis_func = analysis_functions[task]

            # 执行任务分析
            # 首先判断通道数是否正确
            if any(channel >= data.shape[1] or channel < 0 for channel in channels):
                results[task] = np.nan
            else:

                results[task] = analysis_func(data.iloc[:, channels], wdr_conf)

        else:
            print(f"任务 '{task}' 没有对应的分析函数！")

    return pd.Series(results)

""" =========== 数据预处理函数 ================="""
# 定义位移预处理方法
def Prep_WDR(data, wdr_conf):
    ############判断缺失（nan判断）#############
    df = data.iloc[:, :]
    num_columns = data.shape[1]
    result = []
    for i in range(num_columns):
        new_data = data.iloc[:, i]
        missing_index = new_data.isna().mean()
        result.append(missing_index)
    np_array = np.array(result)
    result = np_array.tolist()
    return result

def average(data, para_conf=0):
    """计算数据的平均值, 由于在Acc_Analyze中直接调用了，因此为了统一格式，保留参数输入接口"""
    """计算数据的 RMS，尝试转换字符串为数值"""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 Pandas DataFrame")

    # 尝试将所有列转换为数值，无法转换的变 NaN
    numeric_data = data.apply(pd.to_numeric, errors='coerce')
    numeric_data = zero_del(numeric_data)
    return numeric_data.mean().tolist()

def rms(data, para_conf=0):
    """计算数据的均方根（RMS）"""
    """计算数据的 RMS，尝试转换字符串为数值"""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 Pandas DataFrame")

    # 尝试将所有列转换为数值，无法转换的变 NaN
    numeric_data = data.apply(pd.to_numeric, errors='coerce')
    numeric_data = zero_del(numeric_data)

    # 计算标准差，NaN 不影响计算
    return numeric_data.std(ddof=0).tolist()

def zero_del(data):
    data = data.map(lambda x: np.nan if -5 <= x <= 5 else x)
    return data

if __name__ == '__main__':
    tmp_conf = WdrConfig()

    datapath=r'G:\宁波数据\湾头大桥\data\2020-11-01\2020-11-01 14-UAN.csv'

    rlt = Wdr_analyze(datapath, tmp_conf)
