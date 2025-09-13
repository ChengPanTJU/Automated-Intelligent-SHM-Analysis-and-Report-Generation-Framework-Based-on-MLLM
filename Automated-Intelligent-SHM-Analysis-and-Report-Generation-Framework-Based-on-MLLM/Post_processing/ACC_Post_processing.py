"""
读取结果，汇总数据

Created on 2024

@author: Gong Fengzong
"""
import numpy as np
import pandas as pd
import os
import copy
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
import matplotlib.dates as mdates
from config import FileConfig, OutputConfig,AccConfig
OutConfig = OutputConfig()
outputconfig = OutConfig.tasks
import AnalysisFunction.analysis_methods as AM
def ACC_rlt_read(folder_path, acc_conf):
    # 任务与分析函数的映射
    rlt_functions = {
        'preprocess': Prep_rlt,
        'rms': rms_rlt,  # rms函数
        'mean': average_rlt,  # mean函数
        'OMA1': oma_rlt,  # 支持三个类别通道单独计算
        'OMA2': oma_rlt2,
        'OMA3': oma_rlt3,
    }
    # 输入文件路径，加速度配置文件
    rlt_csv = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # 使用 pandas 读取 CSV 文件，跳过表头
            df = pd.read_csv(file_path)
            rlt_csv.append(df)

    rlt_combined = pd.concat(rlt_csv, axis=0, ignore_index=True)
    rlt = pd.DataFrame({'timestamp': rlt_combined['timestamp']})
    rlt['timestamp'] = pd.to_datetime(rlt_combined['timestamp'])
    rlt = rlt.sort_values(by='timestamp')

    file_conf=FileConfig()
    start_time=file_conf.start_time
    end_time=file_conf.end_time
    # 截取指定时间段的数据
    rlt_combined = rlt_combined.sort_values(by='timestamp')
    rlt_combined = rlt_combined[(rlt['timestamp'] >= start_time) & (rlt['timestamp'] <= end_time)]
    rlt_combined = rlt_combined.reset_index(drop=True)
    rlt = rlt[(rlt['timestamp'] >= start_time) & (rlt['timestamp'] <= end_time)]
    rlt = rlt.reset_index(drop=True)
    fre_path = save_array_to_results_folder('rlt_time.csv')
    rlt['timestamp'].to_csv(fre_path, index=False, header=False)
    # 分析对应任务的结果
    for task, channels in acc_conf.tasks_channels.items():
        if task in rlt_functions.keys():
            # 获取任务对应的分析函数
            rlt_func = rlt_functions[task]

            # 执行任务分析
            rlt_func(rlt, rlt_combined)
        else:
            print(f"任务 '{task}' 没有对应的分析函数！")


"""============== 各分析方法结果对应的提取函数 =============="""

def Prep_rlt(rlt, rlt_combined):
    rlt['preprocess'] = rlt_combined['concat_ws(,, analysis_result[preprocess] AS preprocess)'].apply(convert_to_list)
    abnormal_list = rlt['preprocess'].tolist()
    result_list = []
    acc_conf = AccConfig()
    for task, channels in acc_conf.tasks_channels.items():
        if task=='preprocess':
            channel_num=len(channels)
    for i in range(len(abnormal_list)):
        if len(abnormal_list[i]) == 1:
            for j in range(channel_num):
                abnormal_list[i]=[1,0]*channel_num
    for i in range(len(abnormal_list)):
        an_hour_result = []
        for j in range(channel_num):
            if abnormal_list[i][2 * j] == 0 and abnormal_list[i][2 * j + 1] == 0:
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
    preprocess_path = outputconfig['vibration']['preprocess']['save_path']
    np.savetxt(preprocess_path, result_list, delimiter=',')
    ################### 绘图 ############################################
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
    channels_list = outputconfig['vibration']['preprocess']['channels']
    ytick_labels = [f"ACC{str(i).zfill(2)}" for i in channels_list]
    ax.set_yticklabels(ytick_labels, fontfamily='Times New Roman', fontsize=10)
    grid_lines = np.arange(-0.5, matrix.shape[0] + 0.5, 1)  # 网格线的位置，这里偏移了 -0.5 和 +0.5
    for line in grid_lines:
        ax.axhline(y=line, color='black', linestyle='-', linewidth=1)
    ax.grid(which='both', color='black', linestyle='-', linewidth=1, axis='x')
    # 设置 x 轴和 y 轴的标签
    plt.xlabel('时间 (月-日)')
    plt.ylabel('传感器通道')
    # 添加图例
    fig_height_in_inches = fig.get_size_inches()[1]  # 获取 figure 高度（单位：英寸）
    cm_to_inch = 0.8 / 2.54  # 2cm 转换为英寸
    bottom_position = cm_to_inch / fig_height_in_inches  # 计算 2cm 对应的归一化坐标
    # 预留底部空间，确保图例不会重叠
    fig.subplots_adjust(bottom=bottom_position + 1.1/2.54/fig_height_in_inches)  # 额外增加一些空间
    # 创建图例
    handles = [patches.Patch(facecolor=color_mapping[key], edgecolor='black', label=legend_labels.get(key, 'Unknown'))
               for key in color_mapping]
    legend = ax.legend(handles=handles, loc='upper center',
                       bbox_to_anchor=(0.5, bottom_position),
                       bbox_transform=fig.transFigure,  # 使位置相对于整个 figure
                       ncol=3, fontsize=10)
    title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
    plt.title(title_name[0] + "至" + title_name[len(x_t) - 1] + "主梁加速度传感器情况")
    # 调整图形布局
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    figure_path = outputconfig['vibration']['preprocess']['figure_path']
    plt.savefig(figure_path, format='png', dpi=1000)  # 保存为 PNG 文件，300 dpi 清晰度
    plt.close()

def oma_rlt(rlt, rlt_combined):
    rlt['OMA1'] = rlt_combined['concat_ws(,, analysis_result[OMA1] AS OMA1)'].apply(convert_to_list)
    fre_dp_SSI_list = rlt['OMA1'].tolist()
    max_len = max(len(row) for row in fre_dp_SSI_list)

    for i in range(len(fre_dp_SSI_list)):
        if len(fre_dp_SSI_list[i])<max_len:
            # 计算需要补充的 np.nan 的数量
            num_nans = max_len - len(fre_dp_SSI_list[i])
            # 将原始列表转换为 numpy 数组
            original_array = np.array(fre_dp_SSI_list[i])
            # 创建一个包含 np.nan 的数组
            nan_array = np.full(num_nans, np.nan)
            # 将原始数组和 np.nan 数组合并
            expanded_array = np.concatenate((original_array[:len(original_array) // 2],
                                             nan_array[:num_nans // 2],
                                             original_array[len(original_array) // 2:],
                                             nan_array[num_nans // 2:]))
            fre_dp_SSI_list[i]=expanded_array
    padded_data = fre_dp_SSI_list
    fre_dp_SSI = np.array(padded_data)

    if np.all(np.isnan(fre_dp_SSI)):
        fre1 = np.full([np.floor(len(padded_data)).astype(int),1], np.nan)
        dp1 = np.full([np.floor(len(padded_data)).astype(int),1], np.nan)
    else:
        fre1, dp1 = OMA_rlt_Cluster(fre_dp_SSI)

    result = np.concatenate((fre1, dp1), axis=1)

    fre_path = outputconfig['vibration']['OMA1']['save_path']
    np.savetxt(fre_path, result, delimiter=',')

    # 绘制结果图
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 12  # 设置字体大小为12
    fig_width = 14 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = 8 * 0.393701  # cm 转换为英寸
    x_t = rlt['timestamp'].dt.tz_localize(None)  # 时间轴
    figure_path = outputconfig['vibration']['OMA1']['figure_path'][0]
    fig1 = plt.figure(figsize=(fig_width, fig_height))
    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    plt.plot(x_t, fre1[:,:4], label='频率',linewidth=0.4)
    # 设置 x 轴标签格式为 "日-月 时"
    plt.xlabel('时间 (月-日）', fontsize=12)
    plt.ylabel('频率 (Hz)', fontsize=12)
    title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
    plt.title(f"{title_name[0]}至{title_name[len(x_t) - 1]}日OMA1频率识别结果", fontsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=10)
    plt.xticks(rotation=30)  # x 轴标签旋转 30 度
    plt.tick_params(axis='x', labelsize=10)  # 调整字体大小
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置时间格式
    # 在右下角添加年份
    year = x_t.iloc[-1].year  # 获取时间序列的最后一年
    plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction', fontsize=12,
                 xytext=(0, -30), textcoords='offset points', ha='right', va='top')
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    plt.savefig(figure_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
    plt.close(fig1)  # 关闭当前图表以避免影响后面的图表

    figure_path = outputconfig['vibration']['OMA1']['figure_path'][1]
    fig1 = plt.figure(figsize=(fig_width, fig_height))
    dp1[dp1 < 0] = np.nan
    plt.plot(x_t, dp1*100, label='阻尼比',linewidth=0.3)
    # 设置 x 轴标签格式为 "日-月 时"
    plt.xlabel('时间 (月-日）', fontsize=12)
    plt.ylabel('阻尼比 (%)', fontsize=12)
    title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
    plt.title(f"{title_name[0]}至{title_name[len(x_t) - 1]}日OMA1阻尼比识别结果", fontsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=10)
    plt.xticks(rotation=30)  # x 轴标签旋转 30 度
    plt.tick_params(axis='x', labelsize=10)  # 调整字体大小
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置时间格式
    # 在右下角添加年份
    year = x_t.iloc[-1].year  # 获取时间序列的最后一年
    plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction', fontsize=12,
                 xytext=(0, -30), textcoords='offset points', ha='right', va='top')
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    plt.savefig(figure_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
    plt.close(fig1)  # 关闭当前图表以避免影响后面的图表
    ##############################补充绘制振型图

def oma_rlt2(rlt, rlt_combined):
    rlt['OMA2'] = rlt_combined['concat_ws(,, analysis_result[OMA2] AS OMA2)'].apply(convert_to_list)
    fre_dp_SSI_list = rlt['OMA2'].tolist()
    max_len = max(len(row) for row in fre_dp_SSI_list)
    for i in range(len(fre_dp_SSI_list)):
        if len(fre_dp_SSI_list[i]) < max_len:
            # 计算需要补充的 np.nan 的数量
            num_nans = max_len - len(fre_dp_SSI_list[i])
            # 将原始列表转换为 numpy 数组
            original_array = np.array(fre_dp_SSI_list[i])
            # 创建一个包含 np.nan 的数组
            nan_array = np.full(num_nans, np.nan)
            # 将原始数组和 np.nan 数组合并
            expanded_array = np.concatenate((original_array[:len(original_array) // 2],
                                             nan_array[:num_nans // 2],
                                             original_array[len(original_array) // 2:],
                                             nan_array[num_nans // 2:]))
            fre_dp_SSI_list[i] = expanded_array
    padded_data = fre_dp_SSI_list
    fre_dp_SSI = np.array(padded_data)

    if np.all(np.isnan(fre_dp_SSI)):
        fre1 = np.full([np.floor(len(padded_data)).astype(int), 1], np.nan)
        dp1 = np.full([np.floor(len(padded_data)).astype(int), 1], np.nan)
    else:
        fre1, dp1 = OMA_rlt_Cluster(fre_dp_SSI)

    result = np.concatenate((fre1, dp1), axis=1)

    fre_path = outputconfig['vibration']['OMA2']['save_path']
    np.savetxt(fre_path, result, delimiter=',')

    # 绘制结果图
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 12  # 设置字体大小为12
    fig_width = 14 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = 8 * 0.393701  # cm 转换为英寸
    x_t = rlt['timestamp'].dt.tz_localize(None)  # 时间轴
    figure_path = outputconfig['vibration']['OMA2']['figure_path'][0]
    fig1 = plt.figure(figsize=(fig_width, fig_height))
    plt.plot(x_t, fre1, label='频率',linewidth=0.2)
    # 设置 x 轴标签格式为 "日-月 时"
    plt.xlabel('时间 (月-日）', fontsize=12)
    plt.ylabel('频率 (Hz)', fontsize=12)
    title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
    plt.title(f"{title_name[0]}至{title_name[len(x_t) - 1]}日OMA2频率识别结果", fontsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=10)
    plt.xticks(rotation=30)  # x 轴标签旋转 30 度
    plt.tick_params(axis='x', labelsize=10)  # 调整字体大小
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置时间格式
    # 在右下角添加年份
    year = x_t.iloc[-1].year  # 获取时间序列的最后一年
    plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction', fontsize=12,
                 xytext=(0, -30), textcoords='offset points', ha='right', va='top')
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    plt.savefig(figure_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
    plt.close(fig1)  # 关闭当前图表以避免影响后面的图表

    figure_path = outputconfig['vibration']['OMA2']['figure_path'][1]
    fig1 = plt.figure(figsize=(fig_width, fig_height))
    dp1[dp1 < 0] = np.nan
    plt.plot(x_t, dp1 * 100, label='阻尼比',linewidth=0.2)
    # 设置 x 轴标签格式为 "日-月 时"
    plt.xlabel('时间 (月-日）', fontsize=12)
    plt.ylabel('阻尼比 (%)', fontsize=12)
    title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
    plt.title(f"{title_name[0]}至{title_name[len(x_t) - 1]}日OMA2阻尼比识别结果", fontsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=10)
    plt.xticks(rotation=30)  # x 轴标签旋转 30 度
    plt.tick_params(axis='x', labelsize=10)  # 调整字体大小
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置时间格式
    # 在右下角添加年份
    year = x_t.iloc[-1].year  # 获取时间序列的最后一年
    plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction', fontsize=12,
                 xytext=(0, -30), textcoords='offset points', ha='right', va='top')
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    plt.savefig(figure_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
    plt.close(fig1)  # 关闭当前图表以避免影响后面的图表

def oma_rlt3(rlt, rlt_combined):
    rlt['OMA3'] = rlt_combined['concat_ws(,, analysis_result[OMA3] AS OMA3)'].apply(convert_to_list)
    fre_dp_SSI_list = rlt['OMA3'].tolist()
    max_len = max(len(row) for row in fre_dp_SSI_list)

    for i in range(len(fre_dp_SSI_list)):
        if len(fre_dp_SSI_list[i]) < max_len:
            # 计算需要补充的 np.nan 的数量
            num_nans = max_len - len(fre_dp_SSI_list[i])
            # 将原始列表转换为 numpy 数组
            original_array = np.array(fre_dp_SSI_list[i])
            # 创建一个包含 np.nan 的数组
            nan_array = np.full(num_nans, np.nan)
            # 将原始数组和 np.nan 数组合并
            expanded_array = np.concatenate((original_array[:len(original_array) // 2],
                                             nan_array[:num_nans // 2],
                                             original_array[len(original_array) // 2:],
                                             nan_array[num_nans // 2:]))
            fre_dp_SSI_list[i] = expanded_array
    padded_data = fre_dp_SSI_list
    fre_dp_SSI = np.array(padded_data)

    if np.all(np.isnan(fre_dp_SSI)):
        fre1 = np.full([np.floor(len(padded_data)).astype(int), 1], np.nan)
        dp1 = np.full([np.floor(len(padded_data)).astype(int), 1], np.nan)
    else:
        fre1, dp1 = OMA_rlt_Cluster(fre_dp_SSI)

    result = np.concatenate((fre1, dp1), axis=1)

    fre_path = outputconfig['vibration']['OMA3']['save_path']
    np.savetxt(fre_path, result, delimiter=',')

    # 绘制结果图
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 12  # 设置字体大小为12
    fig_width = 14 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = 8 * 0.393701  # cm 转换为英寸
    x_t = rlt['timestamp'].dt.tz_localize(None)  # 时间轴  # 时间轴
    figure_path = outputconfig['vibration']['OMA3']['figure_path'][0]
    fig1 = plt.figure(figsize=(fig_width, fig_height))
    plt.plot(x_t, fre1, label='频率',linewidth=0.2)
    # 设置 x 轴标签格式为 "日-月 时"
    plt.xlabel('时间 (月-日）', fontsize=12)
    plt.ylabel('频率 (Hz)', fontsize=12)
    title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
    plt.title(f"{title_name[0]}至{title_name[len(x_t) - 1]}日OMA3频率识别结果", fontsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=10)
    plt.xticks(rotation=30)  # x 轴标签旋转 30 度
    plt.tick_params(axis='x', labelsize=10)  # 调整字体大小
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置时间格式
    # 在右下角添加年份
    year = x_t.iloc[-1].year  # 获取时间序列的最后一年
    plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction', fontsize=12,
                 xytext=(0, -30), textcoords='offset points', ha='right', va='top')
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    plt.savefig(figure_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
    plt.close(fig1)  # 关闭当前图表以避免影响后面的图表

    figure_path = outputconfig['vibration']['OMA3']['figure_path'][1]
    fig1 = plt.figure(figsize=(fig_width, fig_height))
    dp1[dp1 < 0] = np.nan
    plt.plot(x_t, dp1 * 100, label='阻尼比')
    # 设置 x 轴标签格式为 "日-月 时"
    plt.xlabel('时间 (月-日）', fontsize=12)
    plt.ylabel('阻尼比 (%)', fontsize=12,linewidth=0.2)
    title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
    plt.title(f"{title_name[0]}至{title_name[len(x_t) - 1]}日OMA3阻尼比识别结果", fontsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=10)
    plt.xticks(rotation=30)  # x 轴标签旋转 30 度
    plt.tick_params(axis='x', labelsize=10)  # 调整字体大小
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置时间格式
    # 在右下角添加年份
    year = x_t.iloc[-1].year  # 获取时间序列的最后一年
    plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction', fontsize=12,
                 xytext=(0, -30), textcoords='offset points', ha='right', va='top')
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    plt.savefig(figure_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
    plt.close(fig1)  # 关闭当前图表以避免影响后面的图表

def rms_rlt(rlt, rlt_combined):
    rlt['rms'] = rlt_combined['concat_ws(,, analysis_result[rms] AS rms)'].apply(convert_to_list)
    rms_path = outputconfig['vibration']['rms']['save_path']
    rms = pad_lists(rlt['rms'].tolist(), len(outputconfig['vibration']['rms']['channels']))
    save_data = pd.DataFrame(rms, index=None)
    for i in range(save_data.shape[1]):
        original_data = save_data.iloc[:,i].copy()
        save_data.iloc[:,i],_=AM.rmoutliers_gesd(original_data)
    np.savetxt(rms_path, save_data, delimiter=',')
    #################################保存汇总表#################################
    time_data = rlt['timestamp'].dt.strftime("%m.%d-%H")
    result = pd.DataFrame({
        '通道号': outputconfig['vibration']['rms']['channels'],  # 设置第一列为通道号
        '加速度均方根最大值(mg)': save_data.max().values,  # 最大值
        '最大值时刻(月.日-时)': save_data.idxmax().values,  # 最大值所在的行号
        '加速度均方根最小值(mg)': save_data.min().values,  # 最小值
        '最小值时刻(月.日-时)': save_data.idxmin().values  # 最小值所在的行号
    })
    # 用 time_data 的索引值替换 result 中的索引
    result['最大值时刻(月.日-时)'] = result['最大值时刻(月.日-时)'].apply(
        lambda x: time_data.loc[x] if pd.notna(x) and x in time_data.index else "通道数据缺失")
    result['最小值时刻(月.日-时)'] = result['最小值时刻(月.日-时)'].apply(
        lambda x: time_data.loc[x] if pd.notna(x) and x in time_data.index else "通道数据缺失")
    # 填充 NaN
    result = result.fillna('通道数据缺失')
    sum_data_path = outputconfig['vibration']['rms']['sum_table_path']
    result.to_csv(sum_data_path, index=False)

    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 8  # 设置字体大小为12
    fig_width = 7 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = 4 * 0.393701  # cm 转换为英寸
    pic_num=len(outputconfig['vibration']['rms']['figure_path'])
    x_t = rlt['timestamp'].dt.tz_localize(None)  # 时间轴
    rms = save_data.values.tolist()
    rms = pad_lists(rms, len(outputconfig['vibration']['rms']['channels']))
    for pic in range(pic_num):
        single_rms=[rms[i][pic] for i in range(len(rms))]
        figure_path = outputconfig['vibration']['rms']['figure_path'][pic]
        channel_num= outputconfig['vibration']['rms']['channels'][pic]
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.plot(x_t, single_rms, label='rms',linewidth=0.2)
        # 设置 x 轴标签格式为 "日-月 时"
        title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
        plt.title(f'{title_name[0]}至{title_name[len(x_t) - 1]}日主梁{channel_num}号加速度计均方根')
        plt.xlabel('时间 (月-日）')
        plt.ylabel('加速度均方根 (mg)')
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
        plt.subplots_adjust(left=0.18, right=0.83, top=0.85, bottom=0.23)
        plt.savefig(figure_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
        plt.close(fig)  # 关闭当前图表以避免影响后面的图表

def average_rlt(rlt, rlt_combined):
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 8  # 设置字体大小为12
    fig_width = 7 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = 4 * 0.393701  # cm 转换为英寸
    rlt['mean'] = rlt_combined['concat_ws(,, analysis_result[mean] AS mean)'].apply(convert_to_list)
    mean_path = outputconfig['vibration']['mean']['save_path']
    rms = pad_lists(rlt['mean'].tolist(), len(outputconfig['vibration']['mean']['channels']))
    save_data = pd.DataFrame(rms, index=None)
    for i in range(save_data.shape[1]):
        original_data = save_data.iloc[:,i].copy()
        save_data.iloc[:,i],_=AM.rmoutliers_gesd(original_data)

    np.savetxt(mean_path, save_data, delimiter=',')
    #################################保存汇总表#################################
    time_data = rlt['timestamp'].dt.strftime("%m.%d-%H")
    result = pd.DataFrame({
        '通道号': outputconfig['vibration']['mean']['channels'],  # 设置第一列为通道号
        '加速度均值最大值(mg)': save_data.max().values,  # 最大值
        '最大值时刻(月.日-时)': save_data.idxmax().values,  # 最大值所在的行号
        '加速度均值最小值(mg)': save_data.min().values,  # 最小值
        '最小值时刻(月.日-时)': save_data.idxmin().values  # 最小值所在的行号
    })
    # 用 time_data 的索引值替换 result 中的索引
    result['最大值时刻(月.日-时)'] = result['最大值时刻(月.日-时)'].apply(
        lambda x: time_data.loc[x] if pd.notna(x) and x in time_data.index else "通道数据缺失")
    result['最小值时刻(月.日-时)'] = result['最小值时刻(月.日-时)'].apply(
        lambda x: time_data.loc[x] if pd.notna(x) and x in time_data.index else "通道数据缺失")
    # 填充 NaN
    result = result.fillna('通道数据缺失')
    sum_data_path = outputconfig['vibration']['mean']['sum_table_path']
    result.to_csv(sum_data_path, index=False)

    pic_num = len(outputconfig['vibration']['mean']['figure_path'])
    x_t = rlt['timestamp'].dt.tz_localize(None)  # 时间轴
    rms = save_data.values.tolist()
    rms = pad_lists(rms, len(outputconfig['vibration']['mean']['channels']))
    for pic in range(pic_num):
        single_rms = [rms[i][pic] for i in range(len(rms))]
        figure_path = outputconfig['vibration']['mean']['figure_path'][pic]
        channel_num = outputconfig['vibration']['mean']['channels'][pic]
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.plot(x_t, single_rms, label='mean',linewidth=0.2)
        # 设置 x 轴标签格式为 "日-月 时"
        title_name = rlt['timestamp'].dt.strftime("%y.%m.%d")
        plt.title(f'{title_name[0]}至{title_name[len(x_t) - 1]}主梁{channel_num}号通道加速度均值')
        plt.xlabel('时间 (月-日）')
        plt.ylabel('加速度均值(mg)')
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
        #plt.tight_layout()  # 自动调整布局以防止标签重叠
        plt.subplots_adjust(left=0.15, right=0.91, top=0.87, bottom=0.25)
        plt.savefig(figure_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
        plt.close(fig)  # 关闭当前图表以避免影响后面的图表

def OMA_rlt_Cluster(SSI_data):
    # 输入SSI结果，类型为np矩阵
    # 将nan转为-1，以便聚类
    data = np.nan_to_num(SSI_data, nan=-1)
    # 将频率和阻尼比分开，聚合成两列
    idx = np.floor(0.5 * data.shape[1]).astype(int)
    flat_freqs = data[:, :idx].flatten(order='F')
    flat_dps = data[:, idx:].flatten(order='F')
    db = DBSCAN(eps=0.05, min_samples=np.floor(0.1 * len(data)).astype(int))
    idx_clt = db.fit_predict(np.column_stack((flat_freqs, flat_dps)))
    # 生成聚类标签矩阵
    cluster_labels = idx_clt.reshape(-1, len(data)).T
    max_label = int(np.nanmax(cluster_labels)) + 1
    n_rows = data.shape[0]
    # 初始化结果矩阵
    data = np.where(data == -1, np.nan, data)
    fre_result = np.full((n_rows, max_label), np.nan)
    dp_result = np.full((n_rows, max_label), np.nan)
    # 调整data每一行的阶次
    fre_data = data[:, :idx]
    dp_data = data[:, idx:]
    # 第一个循环：遍历每个类别索引
    for label in range(max_label):
        # 第二个循环：遍历每一行
        for i in range(n_rows):
            # 获取当前行中属于当前类别的索引
            indices = np.where(cluster_labels[i] == label)[0]
            if len(indices) > 0:
                fre_result[i, label] = fre_data[i, indices[0]]  # 取第一个匹配的原始数据
                dp_result[i, label] = dp_data[i, indices[0]]  # 取第一个匹配的原始数据

    # 判断哪些列全为NaN
    nan_columns = np.all(np.isnan(fre_result), axis=0)
    # 获取非NaN列的索引
    non_nan_columns = np.where(~nan_columns)[0]
    # 获取NaN列的索引
    nan_columns_idx = np.where(nan_columns)[0]

    # 将非NaN列的顺序调整到前面
    fre_reordered = fre_result[:, non_nan_columns]
    dp_reordered = dp_result[:, non_nan_columns]

    # 如果有NaN列，将其添加到最后
    if len(nan_columns_idx) > 0:
        fre_reordered = np.column_stack([fre_reordered, fre_result[:, nan_columns_idx]])
        dp_reordered = np.column_stack([dp_reordered, dp_result[:, nan_columns_idx]])

    return fre_reordered, dp_reordered

def convert_to_list(text):
    """将逗号分隔的字符串转换为浮点数列表"""
    if isinstance(text, str):
        return [float(item) for item in text.split(',')]
    elif isinstance(text, float):
        return [float(text)]
    else:
        return []

def save_array_to_results_folder(filename):
    # 获取当前脚本所在路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建相对路径到'../data/results/'文件夹
    results_dir = os.path.join(current_dir, 'rlt_table')
    # 确保目标文件夹存在
    os.makedirs(results_dir, exist_ok=True)
    # 构建保存文件的完整路径
    file_path = os.path.join(results_dir, filename)
    return file_path

def pad_lists(lst, target_length):
    result = []
    for sublist in lst:
        if not isinstance(sublist, list):  # 确保 sublist 是列表
            sublist = [sublist]  # 将单个值变成列表
        padded = sublist[:target_length] + [np.nan] * (target_length - len(sublist))
        result.append(padded)
    return result



from config import AccConfig


folder_path = r'D:\论文\04自动生成报告\监测数据自动化分析\result_folder_acc'

acc_conf = AccConfig()
ACC_rlt_read(folder_path, acc_conf)

