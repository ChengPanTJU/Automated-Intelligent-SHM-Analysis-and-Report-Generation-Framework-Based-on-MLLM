"""
温度的自动化分析主程序

Created on 2024

@author: Gong Fengzong
"""
import os
import logging
logging.getLogger("distributed.worker.memory").setLevel(logging.ERROR)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

import generateFileList
from AnalysisFunction import Wdr_Analyze
from DataRead import Read_file
from PostProcessing import WDR_Post_processing
from config import FileConfig, WdrConfig, OutputConfig

OutConfig = OutputConfig()
outputconfig = OutConfig.tasks
# 加载配置参数
file_conf = FileConfig()
wdr_conf = WdrConfig()
# 文件读取函数映射
gen_filenames = {
    'gen_filenames': generateFileList.generate_filenames,  # 仅文件路径包含时间
    'gen_filenames_with_path': generateFileList.generate_filenames_with_paths,  # 文件夹路径包含时间
}

# 包装分析函数，输入为一个分区的df数组
def process_partition(df):
    # row-wise apply，传入额外参数lambda row: your_function(row, external_param), axis=1
    # 将字典重新包装成 AccConfig 实例（如果需要）
    wdr_conf_obj = WdrConfig()
    result_df = df.apply(lambda row: Wdr_Analyze.Wdr_analyze(row, wdr_conf_obj), axis=1)
    # 保留原来的行索引
    result_df.index = df.index
    # 合并原始 df 和新结果
    return result_df
def plot_raw_timeseries(
    file_path_list,
    fig_width,
    fig_height,
    channels,
    figure_path,
    missing_threshold=0.2
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    for ch_idx, channel in enumerate(channels):
        plotted = False  # 标记该通道是否已经画过

        for time_tag, csv_path in file_path_list:
            data, file_exists = Read_file.Read_csv1(csv_path)
            if not file_exists:
                continue

            # 防止通道索引越界
            if channel >= data.shape[1]:
                continue

            series = data.iloc[:, channel]

            # 缺失率判断
            if series.isna().mean() > missing_threshold:
                continue

            # ====== 绘图 ======
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            x_t = np.linspace(0, 60, len(series))

            ax.plot(x_t, series, linewidth=1)
            ax.set_xlabel('时间 (min)')
            ax.set_ylabel('风向（°）')
            ax.set_title(
                time_tag.strftime("%y.%m.%d-%H") + f" 时{channel}号风向计时程图"
            )

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.subplots_adjust(left=0.18, right=0.90, top=0.85, bottom=0.23)

            save_path = figure_path[ch_idx]
            plt.savefig(save_path, format='png', dpi=300)
            plt.close(fig)

            plotted = True
            break  # 找到一个文件即可，不再找后续

        if not plotted:
            print(f"⚠️ 风向 {channel} 号通道在所有时段均无有效数据，未绘制原始时程图")
def WDR_main(num_workers=4):
    # --- 配置多核并行 ---
    # num_workers = 4
    # 获取当前脚本所在路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) # 获取上一级目录
    # 构建相对路径到'../data/results/'文件夹
    folder_path = os.path.join(parent_dir, 'RowResults')
    rlt_path = os.path.join(folder_path, 'WDR_rlt_row.csv')

    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)  # 4个worker，每个worker 1线程
    client = Client(cluster)
    # --- 生成文件列表，保持原来逻辑 ---
    file_path = gen_filenames[file_conf.gen_filenames](file_conf.start_time, file_conf.end_time, file_conf.base_dir,
                                                       file_conf.filename_patterns['wind_direction'], file_conf.date_pattern,
                                                       file_conf.time_pattern)
    # 转为pandas数组
    df_pd = pd.DataFrame(file_path, columns=["Time", "file_path"])
    # 转为Dask的df数组
    ddf_dask = dd.from_pandas(df_pd, npartitions=num_workers)

    # 对每个分区应用Acc_Analyze
    meta = {task: object for task in wdr_conf.tasks_channels.keys()}
    # result_df = process_partition(df_pd, acc_conf)
    rlt_ddf = ddf_dask.map_partitions(process_partition, meta=meta)
    rlt_ddf = dd.concat([ddf_dask['Time'], rlt_ddf], axis=1)
    rlt_ddf['Time'] = rlt_ddf['Time'].dt.strftime('%Y/%m/%d %H:%M:%S')

    # 将结果分别保存到csv文件中用于后续绘图分析
    os.makedirs(folder_path, exist_ok=True)
    rlt_ddf.to_csv(rlt_path, index=True, single_file=True)
    # 关闭client
    client.close()
    cluster.close()

    # 结果后处理
    WDR_Post_processing.WDR_rlt_read(rlt_path, wdr_conf)
    ################## 绘制原始数据的时程图 ##################
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 8

    fig_width = 7 * 0.393701
    fig_height = 4 * 0.393701
    channels = outputconfig['wind_direction']['preprocess']['channels']
    figure_path = outputconfig['wind_direction']['preprocess']['raw_data_figure_path']
    plot_raw_timeseries(
        file_path_list=file_path,
        fig_width=fig_width,
        channels=channels,
        figure_path=figure_path,
        fig_height=fig_height
    )

    print('风向分析完成!')

if __name__ == "__main__":
    WDR_main(4)
