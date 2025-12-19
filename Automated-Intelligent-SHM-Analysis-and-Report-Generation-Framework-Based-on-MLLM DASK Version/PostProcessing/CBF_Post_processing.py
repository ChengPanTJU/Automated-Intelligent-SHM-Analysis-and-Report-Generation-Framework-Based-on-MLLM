"""
Created on 2024

@author: Gong Fengzong
"""
import os
import copy
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
import matplotlib.dates as mdates
from config import FileConfig, OutputConfig, CbfConfig
import AnalysisFunction.analysis_methods as AM
OutConfig = OutputConfig()
outputconfig = OutConfig.tasks
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir,'cable_force','rlt_table')
# 确保目标文件夹存在
os.makedirs(results_dir, exist_ok=True)
results_dir = os.path.join(current_dir,'cable_force','rlt_figure')
os.makedirs(results_dir, exist_ok=True)
def CBF_rlt_read(file_path,cbf_conf):
    # 任务与分析函数的映射
    rlt_functions = {
        'preprocess': Prep_rlt,
        'rms': rms_rlt,  # rms函数
        'mean': average_rlt,  # mean函数
        'max_min':max_min_rlt
    }
    # 输入文件路径，加速度配置文件
    rlt_csv = pd.read_csv(file_path)

    rlt_csv['timestamp'] = pd.to_datetime(rlt_csv['Time'])
    rlt = rlt_csv.sort_values(by='timestamp')
    time_path = AM.save_array_to_results_folder('rlt_time.csv')
    rlt['timestamp'].to_csv(time_path, index=False, header=False)
    # 分析对应任务的结果
    for task, channels in cbf_conf.tasks_channels.items():
        if task in rlt_functions.keys():
            # 获取任务对应的分析函数
            rlt_func = rlt_functions[task]

            # 执行任务分析
            rlt_func(rlt)
        else:
            print(f"任务 '{task}' 没有对应的后处理函数！")

"""================= 各分析方法结果对应的提取函数 ====================="""
def Prep_rlt(rlt):
    rlt['preprocess'] = rlt['preprocess'].apply(convert_to_list)
    abnormal_list = rlt['preprocess'].tolist()
    result_list = []
    cbf_conf = CbfConfig()
    for task, channels in cbf_conf.tasks_channels.items():
        if task=='preprocess':
            channel_num=len(channels)
    for i in range(len(abnormal_list)):
        if len(abnormal_list[i])==1:
            for j in range(channel_num):
                abnormal_list[i].append(1)
    for i in range(len(abnormal_list)):
        an_hour_result = []
        for j in range(channel_num):
            if abnormal_list[i][j] == 0 :
                single_channel_result = 0
            else:
                single_channel_result = 1
            an_hour_result.append(single_channel_result)
        result_list.append(an_hour_result)
    drawing_list = copy.deepcopy(result_list)
    evalue_list = []
    for j in range(channel_num):
        evalue_list.append(
            1 - sum([0 if result_list[i][j] == 0 else 1 for i in range(len(result_list))]) / len(result_list))
    result_list.insert(0, evalue_list)
    result_list = pd.DataFrame(result_list, index=None)
    preprocess_path = outputconfig['cable_force']['preprocess']['save_path']
    np.savetxt(preprocess_path, result_list, delimiter=',')
    ################### 绘图 ###########################################
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 12  # 设置字体大小为12
    if channel_num>20:
        fig_height=min(8*channel_num/15,24.5)
    else:
        fig_height=8
    fig_width = 14 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = fig_height * 0.393701  # cm 转换为英寸
    color_mapping = {
        0: '#8DD2C5',
        1: '#F47F72',
    }
    legend_labels = {
        0: '正常',
        1: '异常',
    }
    drawing_list = pd.DataFrame(drawing_list, index=None)
    matrix = drawing_list.T
    # 获取矩阵中的所有唯一值
    unique_values = np.unique(matrix)
    # 根据唯一值创建颜色列表，确保颜色顺序与unique_values对应
    colors = [color_mapping.get(value, 'white') for value in unique_values]  # 未定义的数字默认为白色
    # 创建一个有序的颜色映射表
    cmap = mcolors.ListedColormap(colors)
    # 定义边界以确保每个数字都有独立的颜色
    bounds = unique_values - 0.5
    bounds = np.append(bounds, unique_values[-1] + 0.5)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    # 创建图形和轴，设置图形大小
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # 使用 imshow 绘制矩阵
    cax = ax.imshow(matrix, aspect='auto', cmap=cmap)
    x_t = rlt['timestamp'].dt.strftime("%m.%d")
    xtick_numbers = np.arange(0, len(x_t), int(len(x_t) / 15))
    if xtick_numbers[-1] < len(x_t) - 1:
        xtick_numbers = np.append(xtick_numbers, len(x_t) - 1)
    # plt.xlim(0,len(x_t))
    custom_xtick_labels = [x_t[i] for i in xtick_numbers]  # 替换为您需要的标签
    # 设置 X 轴的标签位置
    xtick_positions = xtick_numbers
    plt.xlim(0, len(x_t) - 1)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(custom_xtick_labels, rotation=30, fontfamily='Times New Roman', fontsize=10)
    # 绘制自定义位置的水平网格线
    ytick_positions = np.arange(0, matrix.shape[0], 1)
    ax.set_yticks(ytick_positions)
    # 生成列表
    channels_list = outputconfig['cable_force']['preprocess']['channels']
    ytick_labels = [f"CBF{str(i).zfill(2)}" for i in channels_list]
    ax.set_yticklabels(ytick_labels, fontfamily='Times New Roman', fontsize=10)
    grid_lines = np.arange(-0.5, matrix.shape[0] + 0.5, 1)  # 网格线的位置，这里偏移了 -0.5 和 +0.5
    for line in grid_lines:
        ax.axhline(y=line, color='black', linestyle='-', linewidth=1)
    ax.grid(which='both', color='black', linestyle='-', linewidth=1, axis='x')
    # 设置 x 轴和 y 轴的标签
    plt.xlabel('时间 (月.日)')
    plt.ylabel('传感器通道')
    # 添加图例
    fig_height_in_inches = fig.get_size_inches()[1]  # 获取 figure 高度（单位：英寸）
    cm_to_inch = 0.8 / 2.54  # 2cm 转换为英寸
    bottom_position = cm_to_inch / fig_height_in_inches  # 计算 2cm 对应的归一化坐标
    # 预留底部空间，确保图例不会重叠
    fig.subplots_adjust(bottom=bottom_position + 1.1 / 2.54 / fig_height_in_inches)  # 额外增加一些空间
    # 创建图例
    handles = [patches.Patch(facecolor=color_mapping[key], edgecolor='black', label=legend_labels.get(key, 'Unknown'))
               for key in color_mapping]
    legend = ax.legend(handles=handles, loc='upper center',
                       bbox_to_anchor=(0.5, bottom_position),
                       bbox_transform=fig.transFigure,  # 使位置相对于整个 figure
                       ncol=3, fontsize=10)
    title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
    plt.title(title_name[0] + "至" + title_name[len(x_t) - 1] + "索力传感器情况")
    # 调整图形布局
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    figure_path = outputconfig['cable_force']['preprocess']['figure_path']
    plt.savefig(figure_path, format='png', dpi=1000)  # 保存为 PNG 文件，300 dpi 清晰度
    plt.close(fig)

def rms_rlt(rlt):
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 8  # 设置字体大小为12
    fig_width = 7 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = 4 * 0.393701  # cm 转换为英寸
    rlt['rms'] = rlt['rms'].apply(convert_to_list)
    rms_path = outputconfig['cable_force']['rms']['save_path']
    rms = pad_lists(rlt['rms'].tolist(), len(outputconfig['cable_force']['rms']['channels']))
    save_data = pd.DataFrame(rms, index=None)
    for i in range(save_data.shape[1]):
        original_data = save_data.iloc[:,i].copy()
        save_data.iloc[:,i],_=AM.rmoutliers_gesd(original_data)
    np.savetxt(rms_path, save_data, delimiter=',')
#################################保存汇总表#################################
    time_data = rlt['timestamp'].dt.strftime("%m.%d-%H")
    filled_data = save_data.fillna(0)  # 用0填充避免idxmin失败
    result = pd.DataFrame({
        '通道号': outputconfig['cable_force']['rms']['channels'],  # 设置第一列为通道号
        '索力均方根最大值(kN)': save_data.max().values,  # 最大值
        '最大值时刻(月.日-时)': filled_data.idxmax().values,  # 最大值所在的行号
        '索力均方根最小值(kN)': save_data.min().values,  # 最小值
        '最小值时刻(月.日-时)': filled_data.idxmin().values  # 最小值所在的行号
    })
    # 用 time_data 的索引值替换 result 中的索引
    result['最大值时刻(月.日-时)'] = result['最大值时刻(月.日-时)'].apply(
        lambda x: time_data.loc[x] if pd.notna(x) and x in time_data.index else "通道数据缺失")
    result['最小值时刻(月.日-时)'] = result['最小值时刻(月.日-时)'].apply(
        lambda x: time_data.loc[x] if pd.notna(x) and x in time_data.index else "通道数据缺失")
    # 填充 NaN
    result = result.fillna('通道数据缺失')
    sum_data_path=outputconfig['cable_force']['rms']['sum_table_path']
    result.to_csv(sum_data_path, index=False)

    pic_num = len(outputconfig['cable_force']['rms']['figure_path'])
    x_t = rlt['timestamp'].dt.tz_localize(None)  # 时间轴
    rms = save_data.values.tolist()
    rms = pad_lists(rms, len(outputconfig['cable_force']['rms']['channels']))
    for pic in range(pic_num):
        single_rms = [rms[i][pic] for i in range(len(rms))]
        figure_path = outputconfig['cable_force']['rms']['figure_path'][pic]
        channel_num = outputconfig['cable_force']['rms']['channels'][pic]
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.plot(x_t, single_rms, label='rms',linewidth=0.2)
        # 设置 x 轴标签格式为 "日-月 时"
        title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
        plt.title(f'{title_name[0]}至{title_name[len(x_t) - 1]}日{channel_num}号索力计均方根')
        plt.xlabel('时间 (月-日）')
        plt.ylabel('索力均方根/(kN)')
        plt.tick_params(axis='y')
        plt.tick_params(axis='x')
        plt.xticks(fontsize=6)  # x 轴标签旋转 30 度
        plt.yticks(fontsize=6)
        plt.tick_params(axis='x')  # 调整字体大小
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置时间格式
        # 在右下角添加年份
        year = x_t.iloc[-1].year  # 获取时间序列的最后一年
        plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction',
                     xytext=(0, -20), textcoords='offset points', ha='right', va='top')
        plt.tight_layout()  # 自动调整布局以防止标签重叠
        plt.subplots_adjust(left=0.18, right=0.87, top=0.85, bottom=0.23)
        plt.savefig(figure_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
        plt.close(fig)  # 关闭当前图表以避免影响后面的图表

def average_rlt(rlt):
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 8  # 设置字体大小为12
    fig_width = 7 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = 4 * 0.393701  # cm 转换为英寸
    rlt['mean'] = rlt['mean'].apply(convert_to_list)
    mean_path = outputconfig['cable_force']['mean']['save_path']
    rms = pad_lists(rlt['mean'].tolist(), len(outputconfig['cable_force']['mean']['channels']))
    save_data = pd.DataFrame(rms, index=None)
    for i in range(save_data.shape[1]):
        original_data = save_data.iloc[:,i].copy()
        save_data.iloc[:,i],_=AM.rmoutliers_gesd(original_data)
    np.savetxt(mean_path, save_data, delimiter=',')
    #################################保存汇总表#################################
    time_data = rlt['timestamp'].dt.strftime("%m.%d-%H")
    filled_data = save_data.fillna(0)  # 用0填充避免idxmin失败
    result = pd.DataFrame({
        '通道号': outputconfig['cable_force']['mean']['channels'],  # 设置第一列为通道号
        '索力均值最大值(kN)': save_data.max().values,  # 最大值
        '最大值时刻(月.日-时)': filled_data.idxmax().values,  # 最大值所在的行号
        '索力均值最小值(kN)': save_data.min().values,  # 最小值
        '最小值时刻(月.日-时)': filled_data.idxmin().values  # 最小值所在的行号
    })
    # 用 time_data 的索引值替换 result 中的索引
    result['最大值时刻(月.日-时)'] = result['最大值时刻(月.日-时)'].apply(
        lambda x: time_data.loc[x] if pd.notna(x) and x in time_data.index else "通道数据缺失")
    result['最小值时刻(月.日-时)'] = result['最小值时刻(月.日-时)'].apply(
        lambda x: time_data.loc[x] if pd.notna(x) and x in time_data.index else "通道数据缺失")
    # 填充 NaN
    result = result.fillna('通道数据缺失')
    sum_data_path = outputconfig['cable_force']['mean']['sum_table_path']
    result.to_csv(sum_data_path, index=False)
    pic_num = len(outputconfig['cable_force']['mean']['figure_path'])
    x_t = rlt['timestamp'].dt.tz_localize(None)  # 时间轴
    rms = save_data.values.tolist()
    rms = pad_lists(rms, len(outputconfig['cable_force']['mean']['channels']))
    for pic in range(pic_num):
        single_rms = [rms[i][pic] for i in range(len(rms))]
        figure_path = outputconfig['cable_force']['mean']['figure_path'][pic]
        channel_num = outputconfig['cable_force']['mean']['channels'][pic]
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.plot(x_t, single_rms, label='mean',linewidth=0.2)
        # 设置 x 轴标签格式为 "日-月 时"
        title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
        plt.title(f'{title_name[0]}至{title_name[len(x_t) - 1]}日{channel_num}号索力计均值')
        plt.xlabel('时间 (月-日）')
        plt.ylabel('索力均值/(kN)')
        plt.tick_params(axis='y')
        plt.tick_params(axis='x')
        plt.xticks(fontsize=6)  # x 轴标签旋转 30 度
        plt.yticks(fontsize=6)
        plt.tick_params(axis='x')  # 调整字体大小
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置时间格式
        # 在右下角添加年份
        year = x_t.iloc[-1].year  # 获取时间序列的最后一年
        plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction',
                     xytext=(0, -20), textcoords='offset points', ha='right', va='top')
        plt.tight_layout()  # 自动调整布局以防止标签重叠
        plt.subplots_adjust(left=0.18, right=0.87, top=0.85, bottom=0.23)
        plt.savefig(figure_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
        plt.close(fig)  # 关闭当前图表以避免影响后面的图表

def max_min_rlt(rlt):
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 8  # 设置字体大小为12
    fig_width = 7 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = 4 * 0.393701  # cm 转换为英寸
    rlt['max_min'] = rlt['max_min'].apply(convert_to_list)
    mean_path = outputconfig['cable_force']['max_min']['save_path']
    rms = pad_lists(rlt['max_min'].tolist(), len(outputconfig['cable_force']['max_min']['channels'])*2)
    save_data = pd.DataFrame(rms, index=None)
    for i in range(save_data.shape[1]):
        original_data = save_data.iloc[:,i].copy()
        save_data.iloc[:,i],_=AM.rmoutliers_gesd(original_data)
    np.savetxt(mean_path, save_data, delimiter=',')
    #################################保存汇总表#################################
    time_data = rlt['timestamp'].dt.strftime("%m.%d-%H")
    max_data=save_data.iloc[:,:int(save_data.shape[1]/2)]
    filled_maxdata = max_data.fillna(0)  # 用0填充避免idxmin失败
    min_data = save_data.iloc[:, int(save_data.shape[1] / 2):]
    filled_mindata = min_data.fillna(0)  # 用0填充避免idxmin失败

    result = pd.DataFrame({
        '通道号': outputconfig['cable_force']['max_min']['channels'],  # 设置第一列为通道号
        '索力最大值(kN)': max_data.max().values,  # 最大值
        '最大值时刻(月.日-时)': filled_maxdata.idxmax().values,  # 最大值所在的行号
        '索力最小值(kN)': min_data.min().values,  # 最小值
        '最小值时刻(月.日-时)': filled_mindata.idxmin().values  # 最小值所在的行号
    })
    # 用 time_data 的索引值替换 result 中的索引
    result['最大值时刻(月.日-时)'] = result['最大值时刻(月.日-时)'].apply(
        lambda x: time_data.loc[x] if pd.notna(x) and x in time_data.index else "通道数据缺失")
    result['最小值时刻(月.日-时)'] = result['最小值时刻(月.日-时)'].apply(
        lambda x: time_data.loc[x] if pd.notna(x) and x in time_data.index else "通道数据缺失")
    # 填充 NaN
    result = result.fillna('通道数据缺失')
    sum_data_path = outputconfig['cable_force']['max_min']['sum_table_path']
    result.to_csv(sum_data_path, index=False)
    ##############################绘图#################################
    pic_num = len(outputconfig['cable_force']['max_min']['figure_path'])
    x_t = rlt['timestamp'].dt.tz_localize(None)  # 时间轴
    max_data = max_data.values.tolist()
    min_data = min_data.values.tolist()
    for pic in range(pic_num):
        single_max = [max_data[i][pic] for i in range(len(max_data))]
        single_min = [min_data[i][pic] for i in range(len(min_data))]
        figure_path = outputconfig['cable_force']['max_min']['figure_path'][pic]
        channel_num = outputconfig['cable_force']['max_min']['channels'][pic]
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.plot(x_t, single_max, label='max',linewidth=0.2)
        plt.plot(x_t, single_min, label='min',linewidth=0.2)
        # 设置 x 轴标签格式为 "日-月 时"
        title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
        plt.title(f'{title_name[0]}至{title_name[len(x_t) - 1]}日{channel_num}号索力传感器极值')
        plt.xlabel('时间 (月-日）')
        plt.ylabel('索力极值(kN)')
        plt.tick_params(axis='y')
        plt.tick_params(axis='x')
        plt.xticks(fontsize=6)  # x 轴标签旋转 30 度
        plt.yticks(fontsize=6)
        plt.tick_params(axis='x')  # 调整字体大小
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置时间格式
        # 在右下角添加年份
        year = x_t.iloc[-1].year  # 获取时间序列的最后一年
        plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction',
                     xytext=(0, -20), textcoords='offset points', ha='right', va='top')
        plt.tight_layout()  # 自动调整布局以防止标签重叠
        plt.subplots_adjust(left=0.18, right=0.87, top=0.85, bottom=0.23)
        plt.savefig(figure_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
        plt.close(fig)  # 关闭当前图表以避免影响后面的图表

def convert_to_list(s):
    # 先用正则替换 'nan' 为 'math.nan'
    s = str(s).strip()
    s = re.sub(r'\bnan\b', 'math.nan', s.strip())

    try:
        # 如果是列表字符串，直接eval
        val = eval(s, {"math": math})
    except:
        # 如果是单个值（如 "1.0" 或 "nan"）
        val = eval(re.sub(r'\bnan\b', 'math.nan', s), {"math": math})

    # 如果不是列表，包装成列表
    if not isinstance(val, list):
        val = [val]

    return val



def pad_lists(lst, target_length):
    result = []
    for sublist in lst:
        if not isinstance(sublist, list):  # 确保 sublist 是列表
            sublist = [sublist]  # 将单个值变成列表
        padded = sublist[:target_length] + [np.nan] * (target_length - len(sublist))
        result.append(padded)
    return result

if __name__ == '__main__':
    folder_path = r'D:\自动化程序\监测数据自动化分析\result_folder_cbf'
    acc_conf = CbfConfig()
    CBF_rlt_read(folder_path, acc_conf)