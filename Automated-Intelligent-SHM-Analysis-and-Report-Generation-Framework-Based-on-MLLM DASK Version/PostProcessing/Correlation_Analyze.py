import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

from config import OutputConfig, CorConfig

OutConfig = OutputConfig()
out_config = OutConfig.tasks
cor_conf = CorConfig()
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir,'correlation','rlt_table')
# 确保目标文件夹存在
os.makedirs(results_dir, exist_ok=True)
results_dir = os.path.join(current_dir,'correlation','rlt_figure')
os.makedirs(results_dir, exist_ok=True)
def plot_correlation_heatmap(df1, df2,fig_path,data_kind_1,data_kind_2,cor_channel_1,cor_channel_2,task1,task2, method="spearman"):
    task_infor={
        'mean':     '均值',
        'rms':      '均方根',
        'OMA1': ' ',
        'OMA2': ' ',
        'OMA3': ' ',
        'common_analysis':           '',
    }
    # 设置全局字体为宋体，字号为12
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 12  # 设置字体大小为12
    fig_width = 14 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = 14 * 0.393701
    """
    计算 df1 和 df2 之间所有列的相关性，并绘制热图。
    参数:
    - df1: 第一个 DataFrame
    - df2: 第二个 DataFrame
    - method: 相关性计算方法 (默认为 "pearson"，可选 "spearman", "kendall")
    返回:
    - 相关性矩阵 (DataFrame)
    """
    # 确保两个 DataFrame 行数一致
    if df1.shape[0] != df2.shape[0]:
        raise ValueError("df1 和 df2 必须具有相同的行数才能计算相关性")
    empty_columns_df1 = df1.columns[df1.isna().all()].tolist()
    empty_columns_df2 = df2.columns[df2.isna().all()].tolist()
    if len(empty_columns_df1)==len(cor_channel_1) or len(empty_columns_df2)==len(cor_channel_2):
        return 0
    for index in sorted(empty_columns_df1, reverse=True):
        cor_channel_1.remove(index+1)
    for index in sorted(empty_columns_df2, reverse=True):
        cor_channel_2.remove(index+1)
    df1 = df1.dropna(axis=1, how='all')
    df2 = df2.dropna(axis=1, how='all')
    df1, df2 = df1.align(df2, join="inner", axis=0)  # 对齐索引
    #df1, df2 = df1.dropna(), df2.dropna()
    # 计算相关性矩阵
    # 将df1和df2分成两半
    midpoint = len(df1) // 2
    df1_first_half, df1_second_half = df1[:midpoint], df1[midpoint:]
    df2_first_half, df2_second_half = df2[:midpoint], df2[midpoint:]
    if (df1_first_half.isna().all().all() or df1_second_half.isna().all().all() or
        df2_first_half.isna().all().all()or df2_second_half.isna().all().all()):
        return 0
    # 计算前一半的相关性矩阵
    correlation_matrix_first_half = pd.DataFrame(
        {col1: [df1_first_half[col1].corr(df2_first_half[col2], method=method) for col2 in df2_first_half.columns]
         for col1 in df1_first_half.columns},
        index=df2_first_half.columns
    )

    # 计算后一半的相关性矩阵
    correlation_matrix_second_half = pd.DataFrame(
        {col1: [df1_second_half[col1].corr(df2_second_half[col2], method=method) for col2 in df2_second_half.columns]
         for col1 in df1_second_half.columns},
        index=df2_second_half.columns
    )
    label_infor = {
        'displacement': '位移',
        'temperature': '温度',
        'strain': '应变',
        'vibration': '主梁加速度',
        'cable_vib': '拉索加速度',
        'wind_speed': '风速',
        'wind_direction': '风向',
        'inclination': '倾角',
        'settlement': '沉降',
        'GPS': 'GPS',
        'cable_force': '索力',
        'traffic':'交通荷载'
    }
    # 定义颜色边界和对应的颜色
    bounds = [-1, -0.75, -0.5, -0.25,0, 0.25, 0.5, 0.75, 1]
    colors = ['darkblue', 'cyan', 'lightblue', 'lightgreen','lightgreen', 'pink', 'lightcoral', 'darkred']
    # 创建自定义的颜色映射
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    # 使用BoundaryNorm来指定颜色的分界线
    norm = BoundaryNorm(bounds, cmap.N)
    # 绘制热力图
    fig=plt.figure(figsize=(fig_width, fig_height))
    # 绘制前一半的热力图
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(correlation_matrix_first_half, annot=True, cmap=cmap, norm=norm, cbar_kws={'ticks': bounds},
                     linewidths=0.5, linecolor='black')

    # 添加第二个相关系数矩阵的注释
    for i in range(len(correlation_matrix_first_half.columns)):
        for j in range(len(correlation_matrix_first_half.index)):
            # 在每个单元格上标注第二个相关系数矩阵的值
            ax.text(i + 0.5, j + 0.75, f'{correlation_matrix_second_half.iloc[j, i]:.2f}',
                    ha='center', va='center', color='black', fontsize=10)
    if 'OMA' not in [task1, task2]:
        xlabel = f'{label_infor[data_kind_1]}{task_infor[task1]}/通道号'
        ylabel = f'{label_infor[data_kind_2]}{task_infor[task2]}/通道号'
        if len(cor_channel_1)==1:
            title = (f'{label_infor[data_kind_1]}{task_infor[task1]}{cor_channel_1[0]}通道与'
                     f'{label_infor[data_kind_2]}{task_infor[task2]}{cor_channel_2[0]}-{cor_channel_2[-1]}通道的相关系数图')
        elif len(cor_channel_2)==1:
            title = (f'{label_infor[data_kind_1]}{task_infor[task1]}{cor_channel_1[0]}-{cor_channel_1[-1]}通道与{label_infor[data_kind_2]}{task_infor[task2]}'
                 f'{cor_channel_2[-1]}通道的相关系数图')
        else:
            title = (f'{label_infor[data_kind_1]}{task_infor[task1]}{cor_channel_1[0]}-{cor_channel_1[-1]}通道与{label_infor[data_kind_2]}{task_infor[task2]}'
                 f'{cor_channel_2[0]}-{cor_channel_2[-1]}通道的相关系数图')

    if 'OMA' in task1:
        xlabel = f'结构基频/阶数'
        ylabel = f'{label_infor[data_kind_2]}{task_infor[task2]}/通道号'
        if len(cor_channel_1)==1:
            title = (f'结构{cor_channel_1[0]}阶基频与'
                     f'{label_infor[data_kind_2]}{task_infor[task2]}{cor_channel_2[0]}-{cor_channel_2[-1]}通道的相关系数图')
        elif len(cor_channel_2)==1:
            title = (f'结构{cor_channel_1[0]}-{cor_channel_1[-1]}阶基频与{label_infor[data_kind_2]}{task_infor[task2]}'
                 f'{cor_channel_2[0]}通道的相关系数图')
        else:
            title = (f'结构{cor_channel_1[0]}-{cor_channel_1[-1]}阶基频与{label_infor[data_kind_2]}{task_infor[task2]}'
                 f'{cor_channel_2[0]}-{cor_channel_2[-1]}通道的相关系数图')

    if 'OMA' in task2:
        xlabel = f'{label_infor[data_kind_1]}{task_infor[task1]}/通道号'
        ylabel = f'结构基频/阶数'
        if len(cor_channel_1) == 1:
            title = (f'{label_infor[data_kind_1]}{task_infor[task1]}{cor_channel_1[0]}通道与结构{cor_channel_2[0]}-{cor_channel_2[-1]}阶基频'
                     f'的相关系数图')
        elif len(cor_channel_2) == 1:
            title = (f'{label_infor[data_kind_1]}{task_infor[task1]}{cor_channel_1[0]}-{cor_channel_1[-1]}通道与结构{cor_channel_2[0]}阶基频'
                     f'的相关系数图')
        else:
            title = (f'{label_infor[data_kind_1]}{task_infor[task1]}{cor_channel_1[0]}-{cor_channel_1[-1]}通道与结构{cor_channel_2[0]}-{cor_channel_2[-1]}阶基频'
                     f'的相关系数图')
    # 处理 traffic 特殊情况
    if 'common_analysis' in [task1, task2]:
        if 'common_analysis' in task1:
            xlabel = '交通荷载'
        if 'common_analysis' in task2:
            ylabel = '交通荷载'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # 设置 xticks 和 yticks
    ax.set_xticks(np.arange(len(cor_channel_1)) + 0.5)  # 设置 x 轴刻度位置
    ax.set_xticklabels(cor_channel_1, ha="right")  # 使用 cor_channel_1 作为 x 轴标签
    ax.set_yticks(np.arange(len(cor_channel_2)) + 0.5)   # 设置 y 轴刻度位置
    ax.set_yticklabels(cor_channel_2, rotation=0)  # 使用 cor_channel_2 作为 y 轴标签
    # 处理 traffic 特殊情况
    traf_tick=['每小时总车重/t','每小时总车数/辆']
    if 'common_analysis' in [task1, task2]:
        if 'common_analysis' in task1:
            tick=[]
            for i in cor_channel_1:
                tick=tick+[traf_tick[i-1]]
            ax.set_xticklabels(tick, ha="center")  # 使用 cor_channel_1 作为 x 轴标签
        if 'common_analysis' in task2:
            tick = []
            for i in cor_channel_1:
                tick = tick + [traf_tick[i - 1]]
            ax.set_yticklabels(tick, rotation=0)  # 使用 cor_channel_2 作为 y 轴标签

    plt.tight_layout()
    plt.savefig(fig_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
    plt.close(fig)

    return correlation_matrix_first_half, correlation_matrix_second_half

# 结果的对应关系表
def Corr(task_conf):
    # 设置全局字体为宋体，字号为12
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 6  # 设置字体大小为12
    fig_width = 7 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = 4 * 0.393701  # cm 转换为英寸
    label_infor={
        'displacement':     '位移',
        'temperature':      '温度',
        'strain':           '应变',
        'vibration':        '主梁加速度',
        'cable_vib':        '拉索加速度',
        'wind_speed':       '风速',
        'wind_direction':   '风向',
        'inclination':      '倾角',
        'settlement':       '沉降',
        'GPS':              'GPS',
        'cable_force':      '索力',
        'traffic':          '交通荷载'
    }
    task_infor={
        'mean':     '均值',
        'rms':      '均方根',
        'OMA1': ' ',
        'OMA2': ' ',
        'OMA3': ' ',
        'common_analysis':           '',
    }
    data_kind_1,data_kind_2= task_conf
    # 读取对应的分析结果
    cor_task_1=task_conf[data_kind_1][0]
    cor_channel_1=task_conf[data_kind_1][1]
    data_path_1=out_config[data_kind_1][cor_task_1]['save_path']
    data1=pd.read_csv(data_path_1,header=None)
    cor_task_2 = task_conf[data_kind_2][0]
    cor_channel_2 = task_conf[data_kind_2][1]
    data_path_2 = out_config[data_kind_2][cor_task_2]['save_path']
    data2 = pd.read_csv(data_path_2, header=None)
    origin_channel_1 = out_config[data_kind_1][cor_task_1]['channels']
    origin_channel_2 = out_config[data_kind_2][cor_task_2]['channels']
    if 'OMA' in cor_task_2:
        indices = [origin_channel_1.index(item) for item in cor_channel_1]
        data1 = data1.iloc[:, indices]
        indices=[i-1 for i in cor_channel_2]
        data2 = data2.iloc[:, indices]
    elif 'OMA' in cor_task_1:
        indices = [i-1 for i in cor_channel_1]
        data1 = data1.iloc[:, indices]
        indices=[origin_channel_2.index(item) for item in cor_channel_2]
        data2 = data2.iloc[:, indices]
    else:
        indices = [origin_channel_1.index(item) for item in cor_channel_1]
        data1=data1.iloc[:,indices]
        indices = [origin_channel_2.index(item) for item in cor_channel_2]
        data2=data2.iloc[:,indices]
    for i in range(len(cor_channel_1)):
        for j in range(len(cor_channel_2)):
            list1 = np.array(data1.iloc[:, i])
            list2 = np.array(data2.iloc[:, j])
            # 找出两个数组中有 NaN 的位置
            mask = np.isnan(list1) | np.isnan(list2)
            # 过滤掉含有 NaN 的数据
            clean_data1 = list1[~mask]
            clean_data2 = list2[~mask]
            # 数据拆分
            mid_index = len(clean_data1) // 2
            clean_data1_first_half = clean_data1[:mid_index]
            clean_data2_first_half = clean_data2[:mid_index]
            clean_data1_second_half = clean_data1[mid_index:]
            clean_data2_second_half = clean_data2[mid_index:]
            # 检查是否为空
            if len(clean_data1_first_half) == 0 or len(clean_data2_first_half) == 0 or len(clean_data1_second_half) == 0 or len(clean_data2_second_half) == 0:
                #print(f"数据列 {i} 和 {j} 为空，跳过拟合")
                coefficients_1 = [np.nan, np.nan]
                coefficients_2 = [np.nan, np.nan]  # 也可以返回其他默认值，如 [0, 0]
            else:
                # 对前一半数据拟合
                coefficients_1 = np.polyfit(clean_data1_first_half, clean_data2_first_half, 1)
                coefficients_2 = np.polyfit(clean_data1_second_half, clean_data2_second_half, 1)

            slope_1, intercept_1 = coefficients_1
            y_fit_1 = slope_1 * clean_data1_first_half + intercept_1
            slope_2, intercept_2 = coefficients_2
            y_fit_2 = slope_2 * clean_data1_second_half + intercept_2
            # 绘制数据点
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            # 绘制前一半数据的散点
            plt.scatter(clean_data1_first_half, clean_data2_first_half, edgecolors='blue', label='前半数据', s=2, marker='o', facecolors='none', linewidths=0.4, alpha=0.5)
            # 绘制后一半数据的散点
            plt.scatter(clean_data1_second_half, clean_data2_second_half, edgecolors='orange', label='后半数据', s=2, marker='o', facecolors='none', linewidths=0.4, alpha=0.5)
            # 绘制前一半数据的拟合线
            plt.plot(clean_data1_first_half, y_fit_1, color='green', linestyle='-', label=f'前半拟合k={round(slope_1,3)}')
            # 绘制后一半数据的拟合线
            plt.plot(clean_data1_second_half, y_fit_2, color='red', linestyle='-', label=f'后半拟合k={round(slope_2,3)}')
            # 设置图表信息
            if 'OMA' not in [cor_task_2, cor_task_1]:
                xlabel = f'{label_infor[data_kind_1]}{cor_channel_1[i]}号通道{task_infor[cor_task_1]}'
                ylabel = f'{label_infor[data_kind_2]}{cor_channel_2[j]}号通道{task_infor[cor_task_2]}'
                title = f'{xlabel}与{ylabel}相关性'

            if 'OMA' in cor_task_1:
                xlabel = f'结构第{cor_channel_1[i]}阶频率'
                ylabel = f'{label_infor[data_kind_2]}{cor_channel_2[j]}号通道{task_infor[cor_task_2]}'
                title = f'{xlabel}与{ylabel}相关性'

            if 'OMA' in cor_task_2:
                xlabel = f'{label_infor[data_kind_1]}{cor_channel_1[i]}号通道{task_infor[cor_task_1]}'
                ylabel = f'结构第{cor_channel_2[j]}阶频率'
                title = f'{xlabel}与{ylabel}相关性'
            # 处理 traffic 特殊情况
            if 'common_analysis' in [cor_task_1, cor_task_2]:
                def transform_label(channel):
                    return '每小时车流量' if channel == 2 else '每小时车重'
                if 'common_analysis' in cor_task_1:
                    xlabel = transform_label(cor_channel_1[i])
                if 'common_analysis' in cor_task_2:
                    ylabel = transform_label(cor_channel_2[j])
                title = f'{xlabel}与{ylabel}相关性'
            # 应用标签
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            # 生成6个均匀间隔的x刻度
            if len(clean_data1) > 0:
                xticks = np.linspace(np.min(clean_data1), np.max(clean_data1), 6)
            else:
                # 如果 clean_data1 为空，可以设置默认的 xticks，或者跳过绘制
                xticks = np.linspace(0, 10, 6)  # 设置一个默认范围
            ax.set_xticks(xticks)
            plt.yticks(fontsize=6)
            plt.xticks(fontsize=6)
            plt.tight_layout()  # 自动调整布局以防止标签重叠
            plt.subplots_adjust(left=0.15, right=0.93, top=0.85, bottom=0.35)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2,fontsize=6,
                       labelspacing=0.23,  # 标签之间的垂直间距，值越小越紧凑
                       handlelength=1,  # 图例标识符（线条）的长度，缩短使其更加紧凑
                       columnspacing=0.7)
            value_to_find = task_conf
            keys = [k for k, v in cor_conf.tasks_channels.items() if value_to_find == v][0]
            fig_path = out_config['correlation'][keys]['figure_path'][len(cor_channel_2) * i + j]

            plt.savefig(fig_path, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
            plt.close(fig)

    fig_path = out_config['correlation'][keys]['figure_path'][-1]
    plot_correlation_heatmap(data1, data2, fig_path, data_kind_1, data_kind_2,cor_channel_1,cor_channel_2,cor_task_1,cor_task_2)


if __name__ == '__main__':
    for task, task_config in cor_conf.tasks_channels.items():
        Corr(task_config)