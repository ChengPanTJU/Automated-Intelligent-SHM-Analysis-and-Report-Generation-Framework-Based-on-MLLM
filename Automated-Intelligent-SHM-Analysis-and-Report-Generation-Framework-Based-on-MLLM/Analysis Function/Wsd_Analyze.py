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
from config import WsdConfig, OutputConfig
OutConfig = OutputConfig()
outputconfig = OutConfig.tasks

def Wsd_analyze(file_path, wsd_conf):
    """根据指定的任务执行相应的分析"""
    # 读取文件内容
    data, file_exists = Read_file.Read_csv1(file_path)

    # 分析各通道数据异常和正常情况，异常通道数据设置为NAN
    # data = data_preprocessing(data)
    # 任务与分析函数的映射
    analysis_functions = {
        'preprocess': Prep_WSD,  # 数据预处理函数
        'rms': rms,          # rms函数
        'mean': average,     # mean函数
    }
    results = {}
    # 如果文件不存在，则给各任务赋值为nan
    if not file_exists:
        for task, channels in wsd_conf.tasks_channels.items():
            results[task] = np.nan
        return results

    for task, channels in wsd_conf.tasks_channels.items():
        if task in analysis_functions.keys():
            # 获取任务对应的分析函数
            analysis_func = analysis_functions[task]

            # 执行任务分析
            # 首先判断通道数是否正确
            if any(channel >= data.shape[1] or channel < 0 for channel in channels):
                results[task] = np.nan
            else:
                results[task] = analysis_func(data.iloc[:, channels], wsd_conf)
        else:
            print(f"任务 '{task}' 没有对应的分析函数！")

    return results

""" =========== 数据预处理函数 ================="""
# 定义位移预处理方法
def Prep_WSD(data, wsd_conf):
    ############判断缺失（nan判断）#############
    df = data.iloc[:, :]
    num_columns = data.shape[1]
    result = []
    for i in range(num_columns):
        new_data = data.iloc[:, i]
        missing_index = new_data.isna().mean()
        ############判断离群点（阈值判断）#############
        if max(new_data) > wsd_conf.Pre_conf["up_lim"] or min(new_data) < wsd_conf.Pre_conf["low_lim"]:
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
    # 将数据按行分成6份
    split_data = np.array_split(data, 6)

    # 计算每一份的均值
    avg_values = [part.mean().tolist() for part in split_data]

    # 返回6份中的最大均值
    return max(avg_values)

def rms(data, para_conf=0):
    """计算数据的均方根（RMS）"""
    return  data.std().tolist()

if __name__ == '__main__':
    # 创建一个示例 DataFrame
    data = pd.DataFrame({
        'A': np.arange(1, 13),  # 从1到12的数字
        'B': np.arange(13, 25),  # 从13到24的数字
        'C': np.arange(25, 37)  # 从25到36的数字
    })

    # 打印原始数据
    print("Original Data:")
    print(data)

    # 调用 average 函数
    result = average(data)

    # 打印结果
    print("\nMaximum column mean from split data:", result)

