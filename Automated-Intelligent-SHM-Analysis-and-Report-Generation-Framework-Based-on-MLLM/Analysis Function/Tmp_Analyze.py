"""
温度数据分析函数，提供模态分析，统计分析等分析方法
要求读取数据文件得到的数据为pandas格式
函数输入为dataframe格式数据

Created on 2024

@author: Gong Fengzong
"""
import numpy as np
from config import FileConfig, TmpConfig
from DataRead import Read_file


def Tmp_analyze(file_path, tmp_conf):
    """根据指定的任务执行相应的分析"""
    # 读取文件内容
    data, file_exists = Read_file.Read_csv1(file_path)

    # 分析各通道数据异常和正常情况，异常通道数据设置为NAN
    # data = data_preprocessing(data)
    # 任务与分析函数的映射
    analysis_functions = {
        'preprocess': Prep_Tmp,  # 数据预处理函数
        'rms': rms,  # rms函数
        'mean': average,  # mean函数
    }
    results = {}
    # 如果文件不存在，则给各任务赋值为nan
    if not file_exists:
        for task, channels in tmp_conf.tasks_channels.items():
            results[task] = np.nan
        return results

    for task, channels in tmp_conf.tasks_channels.items():
        if task in analysis_functions.keys():
            # 获取任务对应的分析函数
            analysis_func = analysis_functions[task]

            # 执行任务分析
            # 首先判断通道数是否正确
            if any(channel >= data.shape[1] or channel < 0 for channel in channels):
                results[task] = np.nan
            else:
                results[task] = analysis_func(data.iloc[:, channels], tmp_conf)
        else:
            print(f"任务 '{task}' 没有对应的分析函数！")

    return results


""" =========== 数据预处理函数 ================="""


def Prep_Tmp(data, tmp_conf):
    ############判断缺失（nan判断）#############
    df = data.iloc[:, :]
    num_columns = data.shape[1]
    result = []
    for i in range(num_columns):
        new_data = data.iloc[:, i]
        # print(new_data)
        missing_index = new_data.isna().mean()
        ############判断离群点（阈值判断）#############
        if max(new_data) > tmp_conf.Pre_conf["up_lim"] or min(new_data) < tmp_conf.Pre_conf["low_lim"]:
            outlier_index = 1
        else:
            outlier_index = 0
        result.append(missing_index)
        result.append(outlier_index)
    np_array = np.array(result)
    result = np_array.tolist()
    return result


def average(data, para_conf=0):
    """计算数据的平均值, 由于在Acc_Analyze中直接调用了，因此为了统一格式，保留参数输入接口"""
    data = tmp_preprocessing(data)
    cbfconf = TmpConfig()
    if cbfconf.fs <= 1:
        for i in range(data.shape[1]):
            import AnalysisFunction.analysis_methods as AM
            original_data = data.iloc[:, i].copy()
            data.iloc[:, i], _ = AM.rmoutliers_gesd(original_data)
    return data.mean().tolist()


def rms(data, para_conf=0):
    """计算数据的均方根（RMS）"""
    data = tmp_preprocessing(data)
    cbfconf = TmpConfig()
    if cbfconf.fs <= 1:
        for i in range(data.shape[1]):
            import AnalysisFunction.analysis_methods as AM
            original_data = data.iloc[:, i].copy()
            data.iloc[:, i], _ = AM.rmoutliers_gesd(original_data)
    return  data.std().tolist()

def tmp_preprocessing(data):
    # 计算均值和最大值最小值，以此判断温度数据是否存在异常
    #
    df_mean = np.mean(data, axis=0).tolist()
    df_max = np.max(data, axis=0).tolist()
    df_min = np.min(data, axis=0).tolist()
    df_std = np.std(data, axis=0).tolist()
    # 遍历每一列，检查是否符合条件
    for idx in range(len(df_mean)):
        # 判断 均值 是否小于阈值的条件
        if df_mean[idx] > 60:
            # print(f"Channel {i} has abnormal small RMS. Setting to NaN.")
            data.iloc[:, idx] = np.nan  # 将这一列赋值为 NaN
        if df_std[idx] > 5:
            # print(f"Channel {i} has abnormal small RMS. Setting to NaN.")
            data.iloc[:, idx] = np.nan  # 将这一列赋值为 NaN


    return data


if __name__ == '__main__':
    tmp_conf = TmpConfig()
    file_conf = FileConfig()

    datapath =r'G:\宁波数据\湾头大桥\data\2020-11-02\2020-11-02 02-TMP.csv'
    rlt = Tmp_analyze(datapath, tmp_conf)
    1
