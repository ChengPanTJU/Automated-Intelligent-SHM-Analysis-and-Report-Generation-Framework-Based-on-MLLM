"""
加速度数据分析函数，提供模态分析，统计分析等分析方法
要求读取数据文件得到的数据为pandas格式
函数输入为dataframe格式数据
Created on 2024
@author: Gong Fengzong
"""
import numpy as np
import pandas as pd
from numpy.f2py.cfuncs import typedefs
from openpyxl.styles.builtins import title
import AnalysisFunction.analysis_methods as AM
from DataRead import Read_file
from config import FileConfig, VicConfig

def Vic_analyze(file_path, vic_conf):
    """根据指定的任务执行相应的分析"""
    # 读取文件内容
    # 读取文件内容
    data, file_exists = Read_file.Read_csv1(file_path)
    if file_exists:
        for i in range(1,data.shape[1]):
            data.iloc[:,i]=data.iloc[:,i]-np.mean(data.iloc[:,i])
    # 任务与分析函数的映射
    analysis_functions = {
        'preprocess': Prep_VIC,  # 数据预处理函数
        'rms': rms,  # rms函数
        'mean': average,  # mean函数
        'OMA': VIC_SSI4single_channel,  # 支持三个类别通道单独计算
    }
    results = {}
    # 如果文件不存在，则给各任务赋值为nan
    if not file_exists:
        for task, channels in vic_conf.tasks_channels.items():
            results[task] = np.nan
        return results

    for task, channels in vic_conf.tasks_channels.items():
        if task in analysis_functions.keys():
            # 获取任务对应的分析函数
            analysis_func = analysis_functions[task]
            # 执行任务分析
            # 首先判断通道数是否正确
            if any(channel >= data.shape[1] or channel < 0 for channel in channels):
                results[task] = np.nan
            else:
                results[task] = analysis_func(data.iloc[:, channels], vic_conf)
        else:
            print(f"任务 '{task}' 没有对应的分析函数！")
    return results

# 定义加速度预处理方法
def Prep_VIC(data, vic_conf):
    ############ 判断缺失（nan判断） #############
    num_columns = data.shape[1]
    result = []
    for i in range(num_columns):
        new_data = data.iloc[:, i]
        new_data = new_data.apply(pd.to_numeric, errors='coerce')
        missing_index = new_data.isna().mean()
        if missing_index!=1:
    ############ 判断离群点（阈值判断）#############
            if max(new_data) > float(vic_conf.Pre_conf["up_lim"]) or min(new_data) < float(vic_conf.Pre_conf["low_lim"]):
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
def VIC_SSI4single_channel1(data, vic_conf):
    data = data.dropna(axis=1, how='all')
    total_result=[]
    for i in vic_conf.tasks_channels['OMA']:
        if i in data.columns:
            single_channel_data = pd.DataFrame(data[i])
            result=VIC_SSI_single(single_channel_data,vic_conf)
            if type(result)==float:
                result=[-1,-1]
        else:
            result = [-1, -1]
        result = padding(result, vic_conf.SSI_conf['order_num'])
        total_result=total_result+result
    return total_result

def VIC_SSI4single_channel(data, vic_conf):
    data = data.fillna(0)
    total_result=[]
    for i in vic_conf.tasks_channels['OMA']:
        single_channel_data = pd.DataFrame(data[i])
        result=VIC_SSI_single(single_channel_data,vic_conf)
        if type(result)==float:
            result=[0,0]
        result = padding(result, vic_conf.SSI_conf['order_num'])
        total_result=total_result+result
    return total_result

def VIC_SSI_single(data, vic_conf):
    # 如果存在nan，则删除该通道的列，仅分析其余列
    # 转为np数组
    Y = data.to_numpy()
    Y=Y[:int(len(Y)*0.1)]
    # 先滤波
    low = vic_conf.SSI_conf['filter_band'][0]
    high = vic_conf.SSI_conf['filter_band'][1]
    Y = AM.butter_bandpass_filter(Y.T, low, high, vic_conf.fs)
    # 首先判断是否有nan
    if np.isnan(Y).any() or np.isinf(Y).any() or np.max(Y) > 1e3:
        return np.nan
    # 自动识别方法
    dt = vic_conf.dt
    order = vic_conf.SSI_conf['order']
    err = vic_conf.SSI_conf['err']
    order_num = vic_conf.SSI_conf['order_num']
    freq_orders, dp_orders = AM.AutoSSI(Y, dt, order, err, order_num)
    # 结果转换为列表，前一半为频率，后一半为阻尼比
    result = freq_orders.tolist() + dp_orders.tolist()
    return result

def average(data, para_conf=0):
    """计算数据的平均值, 由于在Acc_Analyze中直接调用了，因此为了统一格式，保留参数输入接口"""
    return data.mean().tolist()

def rms(data, para_conf=0):
    """计算数据的均方根（RMS）"""
    return  data.std().tolist()

def padding(data,order):
    data_length=len(data)
    if data_length<order*2:
        pad_len=int((order*2-data_length)/2)
        result=data[:int(data_length/2)]+[float(0)]*pad_len+data[int(data_length/2):]+[float(0)]*pad_len
        return result
    else:
        return data


if __name__ == '__main__':
    vic_conf = VicConfig()
    datapath = r'G:\宁波数据\外滩大桥\数据\2019-01-01\2019-01-01 00-VIB.csv'
    rlt = Vic_analyze(datapath, vic_conf)
    rlt=pd.DataFrame(rlt)

    print(len(rlt['OMA']))
    # freq_orders, dp_orders = AM.AutoSSI(Y, dt, order, err, order_num)
    # print(freq_orders)
    # print(dp_orders)
