"""
温度的自动化分析主程序

Created on 2024

@author: Gong Fengzong
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import MapType, FloatType, StringType, ArrayType
import generateFileList
from AnalysisFunction import Dis_Analyze
from DataRead import Read_file
from Post_processing import DIS_Post_processing
from config import FileConfig, DisConfig, OutputConfig

OutConfig = OutputConfig()
outputconfig = OutConfig.tasks

# 加载配置参数
file_conf = FileConfig()
dis_conf = DisConfig()
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
                                                   file_conf.filename_patterns['displacement'], file_conf.date_pattern,
                                                   file_conf.time_pattern)
file_list_df = spark.createDataFrame(file_path, ["timestamp", "file_path"])

# 将自定义分析函数注册为 UDF
analyze_file_udf = F.udf(lambda File_path: Dis_Analyze.Dis_analyze(File_path, dis_conf),
                         MapType(StringType(), ArrayType(FloatType())))
result_df = file_list_df.withColumn("analysis_result", analyze_file_udf("file_path"))

# 提取结果
select_columns = [
    F.col("timestamp"),  # 保留原始的 id 列
    F.col("file_path")  # 保留原始的 timestamp 列
]
for task in dis_conf.tasks_channels:
    # 对每个任务（即字典中的 key），使用 getItem 动态获取对应的列
    select_columns.append(F.concat_ws(",", F.col("analysis_result").getItem(task).alias(task)))

df_with_selected_columns = result_df.select(*select_columns)

# df_with_selected_columns.show(truncate=False)
df_with_selected_columns.coalesce(8).write.csv("result_folder_dis", header=True, mode='overwrite')


# 分析结果
# 获取当前脚本所在路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建相对路径到'../data/results/'文件夹
folder_path = os.path.join(current_dir, 'result_folder_dis')
os.makedirs(folder_path, exist_ok=True)
# 将结果分别保存到csv文件中用于后续绘图分析
DIS_Post_processing.DIS_rlt_read(folder_path, dis_conf)
# 关闭当前会话
spark.stop()

##################绘制原始数据的时程图#####################################
# 设置全局字体为宋体，字号为12
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 8  # 设置字体大小为12
fig_width = 7 * 0.393701  # cm 转换为英寸 图幅大小
fig_height = 4 * 0.393701  # cm 转换为英寸
figure_path = outputconfig['displacement']['preprocess']['raw_data_figure_path']
for i in range(len(file_path)):
    data, file_exists = Read_file.Read_csv1(file_path[i][1])
    channels = outputconfig['displacement']['preprocess']['channels']
    try:
        data = data.iloc[:, channels]
    except Exception as e:
        file_exists = 0
    if file_exists:
        missing_ratio = data.isna().mean()
        missing_ratio=np.mean(missing_ratio)
        if missing_ratio>0.2:
            file_exists = 0
    if file_exists:
        for j in range(len(figure_path)):
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            x_t=np.linspace(0,60,len(data))
            channel=outputconfig['displacement']['preprocess']['channels'][j]
            plt.plot(x_t,data[channel],linewidth=1)
            plt.xlabel('时间 (min)')
            plt.ylabel('位移/(mm)')
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.subplots_adjust(left=0.18, right=0.90, top=0.85, bottom=0.23)
            plt.title(file_path[i][0].strftime("%y.%m.%d-%H")+f"时{channel}号位移计时程图")
            plt.savefig(figure_path[j], format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
            plt.close()
        break
print('位移分析完成!')
