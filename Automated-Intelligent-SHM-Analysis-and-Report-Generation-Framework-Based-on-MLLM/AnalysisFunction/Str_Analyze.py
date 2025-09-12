"""
应变数据分析函数
要求读取数据文件得到的数据为pandas格式
函数输入为dataframe格式数据

Created on 2024

@author: Gong Fengzong
"""
import numpy as np
import pandas as pd

from config import FileConfig, StrConfig
from DataRead import Read_file


def Str_analyze(file_path, str_conf):
    """根据指定的任务执行相应的分析"""
    # 读取文件内容
    data, file_exists = Read_file.Read_csv1(file_path)

    # 分析各通道数据异常和正常情况，异常通道数据设置为NAN
    # data = data_preprocessing(data)
    # 任务与分析函数的映射
    analysis_functions = {
        'preprocess': Prep_STR,  # 数据预处理函数
        'rms': rms,  # rms函数
        'mean': average,  # mean函数
        'max_min':max_min
    }
    results = {}
    # 如果文件不存在，则给各任务赋值为nan
    if not file_exists:
        for task, channels in str_conf.tasks_channels.items():
            results[task] = np.nan
        return results

    for task, channels in str_conf.tasks_channels.items():
        if task in analysis_functions.keys():
            # 获取任务对应的分析函数
            analysis_func = analysis_functions[task]

            # 执行任务分析
            # 首先判断通道数是否正确
            if any(channel >= data.shape[1] or channel < 0 for channel in channels):
                results[task] = np.nan
            else:

                results[task] = analysis_func(data.iloc[:, channels], str_conf)
        else:
            print(f"任务 '{task}' 没有对应的分析函数！")

    return results


""" =========== 数据预处理函数 ================="""


# 定义应变预处理方法
def Prep_STR(data, str_conf):
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
    # data = str_preprocessing(data)
    data=data.fillna(0)
    strconf = StrConfig()
    if strconf.fs <= 1:
        for i in range(data.shape[1]):
            import AnalysisFunction.analysis_methods as AM
            original_data = data.iloc[:, i].copy()
            data.iloc[:, i], _ = AM.rmoutliers_gesd(original_data)
    data=data.values

    # 参数设置
    fs = strconf.fs  # 采样频率（每分钟采样10次）
    cutoff_temp = 0.01  # 温度应变的截止频率（较低）
    import AnalysisFunction.analysis_methods as AA

    # 分离温度应变和交通荷载应变
    temp_strain = AA.low_pass_filter(data, cutoff_temp, fs)
    temp_strain=pd.DataFrame(temp_strain)
    result=temp_strain.mean().tolist()
    arr=np.array(result, dtype=float)  # 转为float以支持nan
    arr[arr == 0] = np.nan
    return arr.tolist()

def rms(data, para_conf=0):
    """计算数据的均方根（RMS）"""
    data=data.fillna(0)
    strconf = StrConfig()
    if strconf.fs <= 1:
        for i in range(data.shape[1]):
            import AnalysisFunction.analysis_methods as AM
            original_data = data.iloc[:, i].copy()
            data.iloc[:, i], _ = AM.rmoutliers_gesd(original_data)
    data=data.values

    # 参数设置
    fs = strconf.fs  # 采样频率（每分钟采样10次）
    cutoff_temp = 0.01  # 温度应变的截止频率（较低）
    import AnalysisFunction.analysis_methods as AA

    # 分离温度应变和交通荷载应变
    temp_strain = AA.high_pass_filter(data, cutoff_temp, fs)
    temp_strain=pd.DataFrame(temp_strain)
    result=temp_strain.std().tolist()
    arr=np.array(result, dtype=float)  # 转为float以支持nan
    arr[arr == 0] = np.nan
    return arr.tolist()

def max_min(data, para_conf=0):
    """计算数据的均方根（RMS）"""
    cbfconf = StrConfig()
    if cbfconf.fs <= 1:
        for i in range(data.shape[1]):
            import AnalysisFunction.analysis_methods as AM
            original_data = data.iloc[:, i].copy()
            data.iloc[:, i], _ = AM.rmoutliers_gesd(original_data)

    return data.max().tolist()+data.min().tolist()

if __name__ == '__main__':
    tmp_conf = StrConfig()
    file_conf = FileConfig()

    datapath = r'G:\宁波数据\湾头大桥\data\2020-11-02\2020-11-02 01-RSG.csv'

    rlt = Str_analyze(datapath, tmp_conf)
    1
