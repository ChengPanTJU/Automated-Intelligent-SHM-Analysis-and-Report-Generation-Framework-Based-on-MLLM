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
from config import CbfConfig, OutputConfig
OutConfig = OutputConfig()
outputconfig = OutConfig.tasks

def Cbf_analyze(df_cbf, cbf_conf):
    """根据指定的任务执行相应的分析"""
    file_path = df_cbf['file_path']
    # 读取文件内容
    data, file_exists = Read_file.Read_csv1(file_path)

    # 分析各通道数据异常和正常情况，异常通道数据设置为NAN
    # data = data_preprocessing(data)
    # 任务与分析函数的映射
    analysis_functions = {
        'preprocess': Prep_CBF,  # 数据预处理函数
        'rms': rms,          # rms函数
        'mean': average,     # mean函数
        'max_min': max_min
    }
    results = {}
    # 如果文件不存在，则给各任务赋值为nan
    if not file_exists:
        for task, channels in cbf_conf.tasks_channels.items():
            results[task] = np.nan
        return pd.Series(results)

    for task, channels in cbf_conf.tasks_channels.items():
        if task in analysis_functions.keys():
            # 获取任务对应的分析函数
            analysis_func = analysis_functions[task]

            # 执行任务分析
            # 首先判断通道数是否正确
            if any(channel >= data.shape[1] or channel < 0 for channel in channels):
                results[task] = np.nan
            else:

                results[task] = analysis_func(data.iloc[:, channels], cbf_conf)
        else:
            print(f"任务 '{task}' 没有对应的分析函数！")

    return pd.Series(results)

""" =========== 数据预处理函数 ================="""
# 定义位移预处理方法
def Prep_CBF(data, cbf_conf):
    ############判断缺失（nan判断）#############
    df = data.iloc[:, :]
    num_columns = data.shape[1]
    result = []
    for i in range(num_columns):
        new_data = data.iloc[:, i]
        # print(new_data)
        missing_index = new_data.isna().mean()
        result.append(missing_index)
    np_array = np.array(result)
    result = np_array.tolist()
    return result

def average(data, para_conf=0):
    """计算数据的平均值, 由于在Acc_Analyze中直接调用了，因此为了统一格式，保留参数输入接口"""
    # data = str_preprocessing(data)
    cbfconf = CbfConfig()
    if cbfconf.fs <= 1:
        for i in range(data.shape[1]):
            import AnalysisFunction.analysis_methods as AM
            original_data = data.iloc[:, i].copy()
            data.iloc[:, i], _ = AM.rmoutliers_gesd(original_data)
    return data.mean().tolist()

def rms(data, para_conf=0):
    """计算数据的均方根（RMS）"""
    cbfconf = CbfConfig()
    if cbfconf.fs <= 1:
        for i in range(data.shape[1]):
            import AnalysisFunction.analysis_methods as AM
            original_data = data.iloc[:, i].copy()
            data.iloc[:, i], _ = AM.rmoutliers_gesd(original_data)
    return  data.std().tolist()

def max_min(data, para_conf=0):
    """计算数据的均方根（RMS）"""
    cbfconf = CbfConfig()
    if cbfconf.fs <= 1:
        for i in range(data.shape[1]):
            import AnalysisFunction.analysis_methods as AM
            original_data = data.iloc[:, i].copy()
            data.iloc[:, i], _ = AM.rmoutliers_gesd(original_data)

    return data.max().tolist()+data.min().tolist()

