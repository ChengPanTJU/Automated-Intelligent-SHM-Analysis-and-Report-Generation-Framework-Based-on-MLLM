"""
加速度的自动化分析主程序

Created on 2025

@author: Pan Cheng & Gong Fengzong
Email：2310450@tongji.edu.cn
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import MapType, FloatType, StringType, ArrayType
import generateFileList
from AnalysisFunction import Acc_Analyze
from DataRead import Read_file
from Post_processing import ACC_Post_processing
from config import FileConfig, AccConfig, OutputConfig

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

# 创建Spark会话
spark = SparkSession.builder \
    .appName("AutoAnalysis") \
    .master("local[*]") \
    .getOrCreate()

# 创建数据文件列表，并转为dataframe
file_path = gen_filenames[file_conf.gen_filenames](file_conf.start_time, file_conf.end_time, file_conf.base_dir,
                                                   file_conf.filename_patterns['vibration'], file_conf.date_pattern,
                                                   file_conf.time_pattern)
file_list_df = spark.createDataFrame(file_path, ["timestamp", "file_path"])

# 将自定义分析函数注册为 UDF
analyze_file_udf = F.udf(lambda File_path: Acc_Analyze.Acc_analyze(File_path, acc_conf),
                         MapType(StringType(), ArrayType(FloatType())))
result_df = file_list_df.withColumn("analysis_result", analyze_file_udf("file_path"))

# 提取结果
select_columns = [
    F.col("timestamp"),  # 保留原始的 id 列
    F.col("file_path")  # 保留原始的 timestamp 列
]
for task in acc_conf.tasks_channels:
    # 对每个任务（即字典中的 key），使用 getItem 动态获取对应的列
    select_columns.append(F.concat_ws(",", F.col("analysis_result").getItem(task).alias(task)))

df_with_selected_columns = result_df.select(*select_columns)

# df_with_selected_columns.show(truncate=False)
df_with_selected_columns.coalesce(4).write.csv("result_folder_acc", header=True, mode='overwrite')


# 分析结果
# 获取当前脚本所在路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建相对路径到'../data/results/'文件夹
folder_path = os.path.join(current_dir, 'result_folder_acc')
os.makedirs(folder_path, exist_ok=True)
# 将结果分别保存到csv文件中用于后续绘图分析
ACC_Post_processing.ACC_rlt_read(folder_path, acc_conf)

# 关闭当前会话
spark.stop()

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
        missing_ratio=np.mean(missing_ratio)
        if missing_ratio>0.2:
            file_exists = 0
    if file_exists:
        ###############绘制振型图#########################
        for task in outputconfig['vibration'].keys():
            if 'OMA' in task:
                oma_channels = np.array(outputconfig['vibration'][task]['channels'])
                import AnalysisFunction.Acc_Analyze as AA
                oma_data=data.iloc[:,oma_channels]
                freq_orders, Phi_orders = AA.ACC_SSI_withMS(oma_data, acc_conf)
                if np.all(np.isnan(freq_orders)):
                    print(f'计算{task}时通道缺失过多，无法绘制振型图')
                else:
                    order_path = os.path.join(os.getcwd(), 'Post_processing', "rlt_table", f"rlt_acc_{task}.csv")
                    fre_data = pd.read_csv(order_path,header=None)
                    fre_data = fre_data.mean()
                    fre_data = fre_data[:int(fre_data.shape[0] / 2)]
                    fre_data = np.sort(fre_data[~np.isnan(fre_data)])
                    nearest_indices = np.array([np.abs(freq_orders - a).argmin() for a in fre_data])
                    show_order = min(3, len(nearest_indices))
                    fig = plt.subplots(figsize=(14 * 0.393701 , show_order*3 * 0.393701 ))
                    for j in range(show_order):#################绘制各阶振型
                        order = nearest_indices[j]
                        Phi_orders[order, :] = Phi_orders[order, :] / max(abs(Phi_orders[order, :]))
                        plt.subplot(show_order, 1,j + 1)
                        plt.plot([x for x in range(1, len(Phi_orders[order, :]) + 1)], Phi_orders[order, :])
                        plt.title(f'第{j + 1}阶振型({round(freq_orders[order], 2)}Hz)')
                        plt.ylim(-1.1, 1.1)
                        # 设置 x 轴刻度只显示整数
                        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.tight_layout()
                    save_path=outputconfig['vibration'][task]['figure_path'][-1]
                    plt.savefig(save_path,format='png', dpi=300)
                    plt.close()

        for j in range(len(figure_path)):
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            x_t=np.linspace(0,60,len(data))
            channel=outputconfig['vibration']['preprocess']['channels'][j]
            plt.plot(x_t,channel_data[channel],linewidth=1)
            plt.xlabel('时间 (min)')
            plt.ylabel('加速度（mg）')
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.subplots_adjust(left=0.18, right=0.90, top=0.85, bottom=0.23)
            plt.title(file_path[i][0].strftime("%y.%m.%d-%H")+f"时主梁{channel}号加速度计时程图")
            plt.savefig(figure_path[j], format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
            plt.close()
        break
print('主梁加速度分析完成!')


