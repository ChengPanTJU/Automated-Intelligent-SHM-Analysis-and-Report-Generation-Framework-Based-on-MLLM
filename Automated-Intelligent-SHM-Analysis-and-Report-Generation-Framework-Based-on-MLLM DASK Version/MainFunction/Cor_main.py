"""
各类型数据分析完成后，进行后处理分析，包括各类型数据的相关性分析等任务
直接从Post_processing文件夹中的结果文件进行读取

"""

from config import FileConfig, CorConfig
from PostProcessing import Correlation_Analyze
import os

def Cor_main(num_workers=4):
    # 加载配置参数
    file_conf = FileConfig()
    cor_conf = CorConfig()

    # 任务与分析函数的映射
    analysis_functions = {
        'Correlation1': Correlation_Analyze.Corr,          # 相关性分析
        'Correlation2': Correlation_Analyze.Corr,     # 相关性分析
        'Correlation3': Correlation_Analyze.Corr,     # 相关性分析
        'Correlation4': Correlation_Analyze.Corr,     # 相关性分析
        'Correlation5': Correlation_Analyze.Corr,     # 相关性分析
        'Correlation6': Correlation_Analyze.Corr,     # 相关性分析
        'Correlation7': Correlation_Analyze.Corr,     # 相关性分析
        'Correlation8': Correlation_Analyze.Corr,     # 相关性分析
        }

    # 执行分析，读取分析数据类型和结果类型
    for task, task_config in cor_conf.tasks_channels.items():
        if task in analysis_functions.keys():
            # 获取任务对应的分析函数，函数需要输入二级字典，用于表示任务的相关配置
            analysis_func = analysis_functions[task](task_config)
        else:
            print(f"任务 '{task}' 没有对应的相关性分析函数！")

    print('相关性分析任务完成!')

if __name__ == "__main__":
    Cor_main()
