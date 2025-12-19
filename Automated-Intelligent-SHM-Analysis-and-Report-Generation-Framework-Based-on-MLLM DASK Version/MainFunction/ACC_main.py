"""
加速度的自动化分析主程序

Created on 2025.07

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

from DataRead import Read_file
from AnalysisFunction import Acc_Analyze
from PostProcessing import ACC_Post_processing
from config import FileConfig, AccConfig, OutputConfig
import generateFileList

OutConfig = OutputConfig()
outputconfig = OutConfig.tasks
# 加载配置参数
file_conf = FileConfig()
acc_conf = AccConfig()

# 文件读取函数映射
gen_filenames = {
    'gen_filenames': generateFileList.generate_filenames,  # 仅文件路径包含时间
    'gen_filenames_with_path': generateFileList.generate_filenames_with_paths,  # 文件夹路径包含时间
}


# 包装分析函数，输入为一个分区的df数组
def process_partition(df):
    # row-wise apply，传入额外参数lambda row: your_function(row, external_param), axis=1
    # 将字典重新包装成 AccConfig 实例（如果需要）
    acc_conf_obj = AccConfig()
    # acc_conf_obj.__dict__.update(acc_conf_dict)
    result_df = df.apply(lambda row: Acc_Analyze.Acc_analyze(row, acc_conf_obj), axis=1)
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
            ax.set_ylabel('位移（mm）')
            ax.set_title(
                time_tag.strftime("%y.%m.%d-%H") + f" 时{channel}号位移计时程图"
            )

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.subplots_adjust(left=0.18, right=0.90, top=0.85, bottom=0.23)

            save_path = figure_path[ch_idx]
            plt.savefig(save_path, format='png', dpi=300)
            plt.close(fig)

            plotted = True
            break  # 找到一个文件即可，不再找后续

        if not plotted:
            print(f"⚠️ 加速度 {channel} 号通道在所有时段均无有效数据，未绘制原始时程图")

def ACC_main(num_workers=4):
    # --- 配置多核并行 ---
    # num_workers = 4
    # 获取当前脚本所在路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) # 获取上一级目录
    # 构建相对路径到'../data/results/'文件夹
    folder_path = os.path.join(parent_dir, 'RowResults')
    rlt_path = os.path.join(folder_path, 'ACC_rlt_row.csv')

    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)  # 4个worker，每个worker 1线程
    client = Client(cluster)
    # --- 生成文件列表，保持原来逻辑 ---
    file_path = gen_filenames[file_conf.gen_filenames](file_conf.start_time, file_conf.end_time, file_conf.base_dir,
                                                       file_conf.filename_patterns['vibration'], file_conf.date_pattern,
                                                       file_conf.time_pattern)
    # 转为pandas数组
    df_pd = pd.DataFrame(file_path, columns=["Time", "file_path"])
    # 转为Dask的df数组
    ddf_dask = dd.from_pandas(df_pd, npartitions=num_workers)

    # 对每个分区应用Acc_Analyze
    meta = {task: object for task in acc_conf.tasks_channels.keys()}
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
    ACC_Post_processing.ACC_rlt_read(rlt_path, acc_conf)
    ##################绘制原始数据的时程图#####################################
    # 设置全局字体为宋体，字号为12
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 8  # 设置字体大小为12
    fig_width = 7 * 0.393701  # cm 转换为英寸 图幅大小
    fig_height = 4 * 0.393701  # cm 转换为英寸
    figure_path = outputconfig['vibration']['preprocess']['raw_data_figure_path']
    for i in range(len(file_path)):
        data, file_exists = Read_file.Read_csv1(file_path[i][1])
        channels = outputconfig['vibration']['preprocess']['channels']
        try:
            channel_data = data.iloc[:, channels]
        except Exception as e:
            file_exists = 0
        if file_exists:
            missing_ratio = data.isna().mean()
            missing_ratio = np.mean(missing_ratio)
            if missing_ratio > 0.2:
                file_exists = 0
        if file_exists:
            ###############绘制振型图#########################
            for task in outputconfig['vibration'].keys():
                if 'OMA' in task:
                    oma_channels = np.array(outputconfig['vibration'][task]['channels'])
                    import AnalysisFunction.Acc_Analyze as AA
                    oma_data = data.iloc[:, oma_channels]
                    freq_orders, Phi_orders = AA.ACC_SSI_withMS(oma_data, acc_conf)
                    if np.all(np.isnan(freq_orders)):
                        print(f'计算{task}时通道缺失过多，无法绘制振型图')
                    else:
                        order_path = os.path.join(os.getcwd(), 'Post_processing', "rlt_table", f"rlt_acc_{task}.csv")
                        fre_data = pd.read_csv(order_path, header=None)
                        fre_data = fre_data.mean()
                        fre_data = fre_data[:int(fre_data.shape[0] / 2)]
                        fre_data = np.sort(fre_data[~np.isnan(fre_data)])
                        nearest_indices = np.array([np.abs(freq_orders - a).argmin() for a in fre_data])
                        show_order = min(3, len(nearest_indices))
                        fig = plt.subplots(figsize=(14 * 0.393701, show_order * 3 * 0.393701))
                        for j in range(show_order):  #################绘制各阶振型
                            order = nearest_indices[j]
                            Phi_orders[order, :] = Phi_orders[order, :] / max(abs(Phi_orders[order, :]))
                            plt.subplot(show_order, 1, j + 1)
                            plt.plot([x for x in range(1, len(Phi_orders[order, :]) + 1)], Phi_orders[order, :])
                            plt.title(f'第{j + 1}阶振型({round(freq_orders[order], 2)}Hz)')
                            plt.ylim(-1.1, 1.1)
                            # 设置 x 轴刻度只显示整数
                            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                        plt.tight_layout()
                        save_path = outputconfig['vibration'][task]['figure_path'][-1]
                        plt.savefig(save_path, format='png', dpi=300)
                        plt.close(fig)

            ################## 绘制原始数据的时程图 ##################
            plt.rcParams['font.sans-serif'] = ['SimSun']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 8

            fig_width = 7 * 0.393701
            fig_height = 4 * 0.393701
            channels = outputconfig['vibration']['preprocess']['channels']
            figure_path = outputconfig['vibration']['preprocess']['raw_data_figure_path']
            plot_raw_timeseries(
                file_path_list=file_path,
                fig_width=fig_width,
                channels=channels,
                figure_path=figure_path,
                fig_height=fig_height
            )
    print('主梁加速度分析完成!')


if __name__ == "__main__":
    ACC_main(4)

