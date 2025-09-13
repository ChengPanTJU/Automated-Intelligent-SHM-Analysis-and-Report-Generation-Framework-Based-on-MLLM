"""
交通荷载分析

Created on 2024

@author: Gong Fengzong
"""
from datetime import datetime

from config import FileConfig, TrafConfig, OutputConfig, traf_config
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 加载配置参数
file_conf = FileConfig()
traf_conf = TrafConfig()
OutConfig = OutputConfig()
outputconfig = OutConfig.tasks

# 设置全局字体为宋体，字号为12
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12  # 设置字体大小为12

traffic_path = traf_config.traffic_path
start_time = file_conf.start_time
end_time = file_conf.end_time
# 设置所需信息的列数
traffic_time = traf_config.traffic_time
time_format = traf_conf.time_format
total_weight = traf_conf.total_weight
lane_num = traf_conf.lane_num
speed = traf_conf.speed
axle_weight = traf_conf.axle_weight

# 读取该文件夹下所有文件
files = [f for f in os.listdir(traffic_path) if f.endswith(('.xlsx', '.xls', '.csv'))]

# 初始化一个空的 DataFrame
all_traffic_data = pd.DataFrame()

# 逐个读取文件并合并
for file in files:
    file_path = os.path.join(traffic_path, file)
    # 判断文件类型并读取
    if file.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)  # 可根据文件编码调整
    # 添加文件名列（可选）
    df['source_file'] = file
    # 合并数据
    all_traffic_data = pd.concat([all_traffic_data, df], ignore_index=True)

all_traffic_data = all_traffic_data.dropna(subset=[all_traffic_data.columns[total_weight]])
all_traffic_data = all_traffic_data[all_traffic_data.iloc[:, total_weight] > 0]  # 只保留大于 0 的车重
weight_col = all_traffic_data.columns[total_weight]
all_traffic_data[weight_col] = all_traffic_data[weight_col].astype(float) / 1000

# 基础分析，车重，车流量统计（每小时层面）
# 确保时间列格式正确
all_traffic_data[all_traffic_data.columns[traffic_time]] = pd.to_datetime(
    all_traffic_data[all_traffic_data.columns[traffic_time]].astype(str), format=time_format,
    errors='coerce'
)

start_time = pd.to_datetime(start_time, format='%Y-%m-%d %H', errors='coerce')
end_time = pd.to_datetime(end_time, format='%Y-%m-%d %H', errors='coerce')

# 过滤时间范围
all_traffic_data = all_traffic_data[
    (all_traffic_data.iloc[:, traffic_time] >= start_time) & (all_traffic_data.iloc[:, traffic_time] <= end_time)]
# 按小时统计车重和车辆数量
all_traffic_data['hourly_time'] = all_traffic_data.iloc[:, traffic_time].dt.floor('h')

summary = all_traffic_data.groupby('hourly_time').agg(
    total_weight_sum=(all_traffic_data.columns[total_weight], 'sum'),
    vehicle_count=(all_traffic_data.columns[total_weight], 'count')
).reset_index()
save_path=outputconfig['traffic']['common_analysis']['save_path']
df2=summary.copy()
df2.columns = ['时间', '数据1','数据2']
# 找出 df2 中缺失的时间
x_time_path = os.path.join(os.getcwd(), 'Post_processing', "rlt_table", f"rlt_time.csv")
x_time = pd.read_csv(x_time_path, header=None)
x_time.columns=['时间']
df1=x_time.copy()
# 找出 df2 中缺失的时间
df1['时间'] = pd.to_datetime(df1['时间']).dt.tz_localize(None)
df2['时间'] = pd.to_datetime(df2['时间'])
missing_times = df1[~df1['时间'].isin(df2['时间'])]
# 计算均值
mean_data1 = df2['数据1'].mean()
mean_data2 = df2['数据2'].mean()
# 创建缺失行，并填充均值
missing_rows = missing_times.copy()
missing_rows['数据1'] = mean_data1
missing_rows['数据2'] = mean_data2
# 补全 df2
df2_filled = pd.concat([df2, missing_rows]).sort_values(by='时间').reset_index(drop=True)

df2_filled.iloc[:,[1,2]].to_csv(save_path,index=False,header=None)

fig1 = plt.figure(figsize=(14 * 0.393701, 8 * 0.393701))
plt.plot(summary['hourly_time'], summary['total_weight_sum'], label='车重',linewidth=0.2)
# 设置 x 轴标签格式为 "日-月 时"
plt.xlabel('时间 (月-日）', fontsize=12)
plt.ylabel('重量 (t)', fontsize=12)
formatted_date1 = summary['hourly_time'][0].strftime("%y-%m-%d")
formatted_date2 = summary['hourly_time'][summary.shape[0]-1].strftime("%y-%m-%d")
plt.title(f"{formatted_date1}至{formatted_date2}日每小时车流总重量统计", fontsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tick_params(axis='x', labelsize=10)
plt.xticks(rotation=30)  # x 轴标签旋转 30 度
plt.tick_params(axis='x', labelsize=10)  # 调整字体大小
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置时间格式
# 在右下角添加年份
year = all_traffic_data['hourly_time'].iloc[-1].year  # 获取时间序列的最后一年
plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction', fontsize=12,
             xytext=(0, -30), textcoords='offset points', ha='right', va='top')
plt.tight_layout()  # 自动调整布局以防止标签重叠
fig_path=outputconfig['traffic']['common_analysis']['figure_path'][0]
plt.savefig(fig_path, format='png', dpi=300)
plt.close()

all_traffic_data = all_traffic_data[
    (all_traffic_data.iloc[:, traffic_time] >= start_time) & (all_traffic_data.iloc[:, traffic_time] <= end_time)]
# 按小时统计车重和车辆数量
all_traffic_data['hourly_time'] = all_traffic_data.iloc[:, traffic_time].dt.floor('h')

summary = all_traffic_data.groupby('hourly_time').agg(
    total_weight_sum=(all_traffic_data.columns[total_weight], 'sum'),
    vehicle_count=(all_traffic_data.columns[total_weight], 'count')
).reset_index()

fig1 = plt.figure(figsize=(14 * 0.393701, 8 * 0.393701))
plt.plot(summary['hourly_time'], summary['vehicle_count'], label='车流量',linewidth=0.2)
# 设置 x 轴标签格式为 "日-月 时"
plt.xlabel('时间 (月-日）', fontsize=12)
plt.ylabel('车流量 (辆)', fontsize=12)
formatted_date1 = summary['hourly_time'][0].strftime("%y-%m-%d")
formatted_date2 = summary['hourly_time'][summary.shape[0]-1].strftime("%y-%m-%d")
plt.title(f"{formatted_date1}至{formatted_date2}日每小时车流量统计", fontsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tick_params(axis='x', labelsize=10)
plt.xticks(rotation=30)  # x 轴标签旋转 30 度
plt.tick_params(axis='x', labelsize=10)  # 调整字体大小
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置时间格式
# 在右下角添加年份
year = all_traffic_data['hourly_time'].iloc[-1].year  # 获取时间序列的最后一年
plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction', fontsize=12,
             xytext=(0, -30), textcoords='offset points', ha='right', va='top')
plt.tight_layout()  # 自动调整布局以防止标签重叠
fig_path=outputconfig['traffic']['common_analysis']['figure_path'][2]
plt.savefig(fig_path, format='png', dpi=300)
plt.close()

# 车重概率密度分布
hist_values, bin_edges = np.histogram(all_traffic_data.iloc[:, total_weight], bins=50, density=True)  # 计算概率密度直方图
# 计算 bin 的中心点
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# 绘制直方图
fig1 = plt.figure(figsize=(14 * 0.393701, 8 * 0.393701))
plt.bar(bin_centers, hist_values, width=(bin_edges[1] - bin_edges[0]), alpha=0.6, color='blue',
        edgecolor='black', linewidth=1.5, label="Histogram (PDF)")
#plt.xlim(0, max(all_traffic_data.iloc[:, total_weight]))
# 设置 x 轴标签格式为 "日-月 时"
plt.xlabel('车重 (t)', fontsize=12)
plt.ylabel('概率密度', fontsize=12)
plt.title("车重概率密度分布", fontsize=12)
plt.tick_params(axis='y', labelsize=10)
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='x', labelsize=10)  # 调整字体大小
plt.tight_layout()  # 自动调整布局以防止标签重叠
fig_path=outputconfig['traffic']['common_analysis']['figure_path'][1]
plt.savefig(fig_path, format='png', dpi=300)
plt.close()

# 如果车道号存在，则进一步按照车道统计分析，各车道车流量总数
if pd.isna(lane_num):
    print("lane_num is NaN. Skipping lane-wise analysis.")
else:
    lane_summary = all_traffic_data.groupby(all_traffic_data.columns[lane_num]).agg(
        total_weight_sum=(all_traffic_data.columns[total_weight], 'sum'),  # 该车道的总车重
        vehicle_count=(all_traffic_data.columns[total_weight], 'count')  # 该车道的车辆总数
    ).reset_index()

    # 绘制每个车道的车辆数量直方图
    fig, ax = plt.subplots(figsize=(14 * 0.393701, 8 * 0.393701))
    ax.bar(lane_summary.iloc[:, 0], lane_summary["vehicle_count"], color='blue', alpha=0.7, edgecolor='black')
    # 显示数值标签
    for i, v in enumerate(lane_summary["vehicle_count"]):
        ax.text(lane_summary.iloc[i, 0], v + 5, str(v), ha='center', fontsize=10)
    # 美化图表
    # 设置 x 轴和 y 轴标签
    plt.xlabel('车道号', fontsize=12)
    plt.ylabel('车辆数量 (辆)', fontsize=12)
    # 设置每个车道的标题
    plt.title("各车道车辆数量统计", fontsize=12)
    # 设置 x 轴的刻度为整数
    plt.xticks(range(int(lane_summary.iloc[:, 0].min()), int(lane_summary.iloc[:, 0].max()) + 1))
    plt.tick_params(axis='y', labelsize=10)
    plt.tick_params(axis='x', labelsize=10)
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    fig_path = outputconfig['traffic']['lane']['figure_path'][1]
    plt.savefig(fig_path, format='png', dpi=300)
    plt.close()

    # 进一步统计各个车道的车辆重量概率密度分布
    lane_histograms = {}
    for lane in lane_summary[all_traffic_data.columns[lane_num]].unique():
        lane_data = all_traffic_data[all_traffic_data[all_traffic_data.columns[lane_num]] == lane]
        lane_hist_values, lane_bin_edges = np.histogram(lane_data.iloc[:, total_weight], bins=50, density=True)
        lane_bin_centers = (lane_bin_edges[:-1] + lane_bin_edges[1:]) / 2
        lane_histograms[lane] = (lane_hist_values, lane_bin_centers)

    # 打印每个车道的车重分布情况
    lane_num=len(lane_histograms.keys())
    plt.subplot(lane_num,1,1)
    lane_index=1
    plt.figure(figsize=(14 * 0.393701, 25 * 0.393701))  # 转换为英寸
    for lane, (lane_hist_values, lane_bin_centers) in lane_histograms.items():
        plt.subplot(lane_num, 1, lane_index)
        lane_index=lane_index+1
        plt.bar(lane_bin_centers, lane_hist_values, width=(lane_bin_centers[1] - lane_bin_centers[0]),
                alpha=0.6, color='blue', edgecolor='black', linewidth=1.5, label="Histogram (PDF)")
        #plt.xlim(0, max(all_traffic_data.iloc[:, total_weight]))
        # 设置 x 轴和 y 轴标签
        plt.xlabel('车重 (t)', fontsize=12)
        plt.ylabel('概率密度', fontsize=12)
        # 设置每个车道的标题
        plt.title(f"车道 {lane} 车重概率密度分布", fontsize=12)
        plt.tick_params(axis='y', labelsize=10)
        plt.tick_params(axis='x', labelsize=10)
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    fig_path = outputconfig['traffic']['lane']['figure_path'][0]
    plt.savefig(fig_path, format='png', dpi=300)
    plt.close()



# 如果车速存在，车速概率密度分布
if pd.isna(speed):
    print("Speed is NaN. Skipping Speed analysis.")
else:
    hist_values, bin_edges = np.histogram(all_traffic_data.iloc[:, speed], bins=30, density=True)  # 计算概率密度直方图
    # 计算 bin 的中心点
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 绘制直方图
    fig1 = plt.figure(figsize=(14 * 0.393701, 8 * 0.393701))
    plt.bar(bin_centers, hist_values, width=(bin_edges[1] - bin_edges[0]), alpha=0.6, color='blue',
            edgecolor='black', linewidth=1.5, label="Histogram (PDF)")
    plt.xlim(0, max(all_traffic_data.iloc[:, speed]))
    # 设置 x 轴标签格式为 "日-月 时"
    plt.xlabel('车速 (km/h)', fontsize=12)
    plt.ylabel('概率密度', fontsize=12)
    plt.title("车速概率密度分布", fontsize=12)
    plt.tick_params(axis='y', labelsize=10)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='x', labelsize=10)  # 调整字体大小
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    fig_path = outputconfig['traffic']['speed']['figure_path']
    plt.savefig(fig_path, format='png', dpi=300)
    plt.close()

# 如果车轴数存在，分析各轴车数量，各轴车的车重分布
if pd.isna(axle_weight).any():
    print("axle_weight is NaN. Skipping axle_weight analysis.")
else:
    axle_cols = [all_traffic_data.columns[i] for i in axle_weight]
    # 计算每辆车的轴数（非零轴重的数量）
    all_traffic_data['axle_count'] = (all_traffic_data[axle_cols] > 0).sum(axis=1)
    # 统计不同轴数的车辆数量
    axle_count_summary = all_traffic_data['axle_count'].value_counts().sort_index()
    if 1 in axle_count_summary.index:
        axle_count_summary = axle_count_summary.drop(1)
    # 遍历不同轴数，计算其车重分布并绘制概率密度直方图
    plt.figure(figsize=(14 * 0.393701, 4*len(axle_count_summary) * 0.393701))
    fig_index=1
    for axle_count in axle_count_summary.index:
        # 选择轴数匹配的车辆
        axle_group = all_traffic_data[all_traffic_data['axle_count'] == axle_count].copy(deep=True)
        # 计算这些车辆的总车重（所有轴重之和）
        axle_group.loc[:, 'total_axle_weight'] = axle_group[axle_cols].sum(axis=1)/1000
        # 计算车重概率密度
        hist_values, bin_edges = np.histogram(axle_group['total_axle_weight'], bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # 绘制直方图
        plt.subplot(len(axle_count_summary.index),1,fig_index)
        fig_index=fig_index+1
        plt.bar(bin_centers, hist_values, width=(bin_edges[1] - bin_edges[0]),
                alpha=0.6, color='blue', edgecolor='black', linewidth=1.5, label="Histogram (PDF)")
        plt.xlabel('总重量 (t)', fontsize=12)
        plt.ylabel('概率密度', fontsize=12)
        plt.title(f"{axle_count} 轴车车重概率密度分布", fontsize=12)
        plt.tick_params(axis='y', labelsize=10)
        plt.tick_params(axis='x', labelsize=10)
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    fig_path = outputconfig['traffic']['axle']['figure_path']
    plt.savefig(fig_path, format='png', dpi=300)
    plt.close()

# 分析完成
print('交通荷载分析完成!')

