"""
加速度数据分析函数，提供模态分析，统计分析等分析方法
要求读取数据文件得到的数据为pandas格式
函数输入为dataframe格式数据

Created on 2025

@author: Pan Cheng & Gong Fengzong
Email：2310450@tongji.edu.cn
"""
import numpy as np
import pandas as pd
import AnalysisFunction.analysis_methods as AM
from config import FileConfig, AccConfig
from generateFileList import generate_filenames
from DataRead import Read_file
import matplotlib.pyplot as plt

def Acc_analyze(file_path, acc_conf):
    """根据指定的任务执行相应的分析"""
    # 读取文件内容
    data, file_exists = Read_file.Read_csv1(file_path)
    if file_exists:
        for i in range(1,data.shape[1]):
            data.iloc[:,i]=data.iloc[:,i]-np.mean(data.iloc[:,i])
    # 任务与分析函数的映射
    analysis_functions = {
        'preprocess': Prep_ACC,  # 数据预处理函数
        'rms': rms,  # rms函数
        'mean': average,  # mean函数
        'OMA1': ACC_SSI,  # 支持三个类别通道单独计算
        'OMA2': ACC_SSI,
        'OMA3': ACC_SSI,
    }
    results = {}
    # 如果文件不存在，则给各任务赋值为nan
    if not file_exists:
        for task, channels in acc_conf.tasks_channels.items():
            results[task] = np.nan
        return results

    for task, channels in acc_conf.tasks_channels.items():

        if task in analysis_functions.keys():
            # 获取任务对应的分析函数
            analysis_func = analysis_functions[task]

            # 执行任务分析
            # 首先判断通道数是否正确
            if any(channel >= data.shape[1] or channel < 0 for channel in channels):
                results[task] = np.nan
            else:
                results[task] = analysis_func(data.iloc[:, channels], acc_conf)
        else:
            print(f"任务 '{task}' 没有对应的分析函数！")

    return results

# 定义加速度预处理方法
def Prep_ACC(data, acc_conf):
    ############ 判断缺失（nan判断） #############
    num_columns = data.shape[1]
    result = []
    for i in range(num_columns):
        new_data = data.iloc[:, i]
        new_data = new_data.apply(pd.to_numeric, errors='coerce')
        missing_index = new_data.isna().mean()
        if missing_index!=1:
    ############ 判断离群点（阈值判断）#############
            if max(new_data) > float(acc_conf.Pre_conf["up_lim"]) or min(new_data) < float(acc_conf.Pre_conf["low_lim"]):
                outlier_index = 1
            else:
                outlier_index = 0
        else:
            outlier_index=0
        result.append(missing_index)
        result.append(outlier_index)
    np_array = np.array(result)
    result = np_array.tolist()
    return result

# 定义SSI自动识别模态方法
def ACC_SSI(data, acc_conf):
    # 如果存在nan，则删除该通道的列，仅分析其余列

    data = data.dropna(axis=1, how='all')
    # 数据预处理
    data = data_preprocessing(data)

    # 转为np数组
    Y = data.to_numpy()
    Y=Y[:int(len(Y)*0.1)]

    # 先滤波
    low = acc_conf.SSI_conf['filter_band'][0]
    high = acc_conf.SSI_conf['filter_band'][1]
    Y = AM.butter_bandpass_filter(Y.T, low, high, acc_conf.fs)
    mask = np.isnan(Y).any(axis=1) | np.isinf(Y).any(axis=1)
    # 2. 删除包含 NaN 或 Inf 的行
    Y = Y[~mask]
    # 首先判断是否有nan
    if Y.size == 0 or np.isnan(Y).all():
        return np.nan
    else:
        if np.isnan(Y).any() or np.isinf(Y).any() or np.max(Y) > 1e3:
            return np.nan

    # 自动识别方法
    dt = acc_conf.dt
    order = acc_conf.SSI_conf['order']
    err = acc_conf.SSI_conf['err']
    order_num = acc_conf.SSI_conf['order_num']

    freq_orders, dp_orders = AM.AutoSSI(Y, dt, order, err, order_num)

    # 结果转换为列表，前一半为频率，后一半为阻尼比
    result = freq_orders.tolist() + dp_orders.tolist()

    return result

# 定义SSI自动识别模态方法
def ACC_SSI_withMS(data, acc_conf):
    # 如果存在nan，则删除该通道的列，仅分析其余列
    data = data.dropna(axis=1, how='all')
    # 数据预处理
    data = data_preprocessing(data)

    # 转为np数组
    Y = data.to_numpy()

    # 先滤波
    low = acc_conf.SSI_conf['filter_band'][0]
    high = acc_conf.SSI_conf['filter_band'][1]
    Y = AM.butter_bandpass_filter(Y.T, low, high, acc_conf.fs)
    mask = np.isnan(Y).any(axis=1) | np.isinf(Y).any(axis=1)
    # 2. 删除包含 NaN 或 Inf 的行
    Y = Y[~mask]
    # 首先判断是否有nan
    if Y.size == 0 or np.isnan(Y).all():
        return np.nan
    else:
        if np.isnan(Y).any() or np.isinf(Y).any() or np.max(Y) > 1e3:
            return np.nan

    # 自动识别方法
    dt = acc_conf.dt
    order = acc_conf.SSI_conf['order']
    err = acc_conf.SSI_conf['err']
    order_num = acc_conf.SSI_conf['order_num']

    freq_orders, dp_orders, Phi_orders = AM.AutoSSIwithMS(Y, dt, order, err, order_num)

    return freq_orders,Phi_orders

def average(data, para_conf=0):
    """计算数据的平均值, 由于在Acc_Analyze中直接调用了，因此为了统一格式，保留参数输入接口"""

    return data.mean().tolist()

def rms(data, para_conf=0):
    """计算数据的均方根（RMS）"""

    #return  np.sqrt((data**2).mean()).tolist()
    return data.std().tolist()

def data_preprocessing(data, rms_threshold=0.0001):
    # 计算均值和方差
    df_mean = average(data)
    df_rms = rms(data)
    max_rms =np.nanmax(df_rms) if len(df_rms) > 0 else 1e-6
    # 遍历每一列，检查是否符合条件
    for idx in range(len(df_mean)):
        # 判断 RMS 是否小于阈值的条件
        if df_rms[idx] < rms_threshold * max_rms:
            # print(f"Channel {i} has abnormal small RMS. Setting to NaN.")
            data.iloc[:, idx] = np.nan  # 将这一列赋值为 NaN

    return data

if __name__ == '__main__':
    acc_conf = AccConfig()
    file_conf = FileConfig()
    # datapath = 'F:/实验数据/明州大桥-18-20/明州大桥历史数据/data2018-9-/data/2019-04-27/2019-04-27 04-VIB.csv'
    datapath = r"G:\宁波数据\明洲大桥\data\2015-09-06\2015-09-06 00-VIB.csv"
    # 自动识别方法
    data, _ = Read_file.Read_csv1(datapath)
    # 自动化SSI识别模态和阻尼比(输入dataframe）
    Y = data.iloc[:, [10]]

    # 转为np数组
    Y = Y.to_numpy()
    # 先滤波
    low = acc_conf.SSI_conf['filter_band'][0]
    high = acc_conf.SSI_conf['filter_band'][1]
    Y = AM.butter_bandpass_filter(Y.T, low, high, acc_conf.fs)
    # 自动识别方法
    dt = acc_conf.dt
    order = acc_conf.SSI_conf['order']
    err = acc_conf.SSI_conf['err']
    order_num = acc_conf.SSI_conf['order_num']
    A, C, _, R0 = AM.ssidata(Y, order, 2 * order)
    f, psi, Phi = AM.modalparams(A, C, dt)

    # 计算稳定点
    IDs, stable_list, unstable_list = AM.out_stab(A, C, dt, err)
    AM.plot_stab(Y, dt, stable_list, unstable_list)

    # 自动识别方法
    rlt = Acc_analyze(datapath, acc_conf)
    print(rlt['OMA'])

    # freq_orders, dp_orders = AM.AutoSSI(Y, dt, order, err, order_num)
    # print(freq_orders)
    # print(dp_orders)
