"""
监测数据自动化分析主程序

Created on 2025

@author: Pan Cheng & Gong Fengzong
Email：2310450@tongji.edu.cn
"""

import Pic_Insert
import Pic_identifier_Insert
import Word_Insert
from config import FileConfig, OutputConfig
import os
from _00_common_model import ensure_model
from _01_MLLM4pic import run_MLLM4pic
from _02_MLLM4single_type_sum import run_MLLM4single_type_sum
from _03_MLLM4corandreg_sum import run_MLLM4corandreg_sum
from _04_MLLM4all_sum import run_MLLM4all_sum

def clear_files_only(folder_path):
    if not os.path.exists(folder_path):
        print("文件夹不存在！")
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
        except Exception as e:
            print(f"删除 {file_path} 失败: {e}")

OutConfig = OutputConfig()
out_config = OutConfig.tasks
# 加载配置参数
file_conf = FileConfig()
# 任务与分析函数的映射
analysis_module = {
    'temperature':      'TMP_main',     # 温度分析模块
    'strain':           'STR_main',     # 应变分析模块
    'traffic':          'TRAF_main',    # 交通流分析模块
    'vibration':        'ACC_main',     # 加速度分析模块
    'cable_vib':        'VIC_main',     # 拉索加速度分析模块
    'displacement':     'DIS_main',     # 位移分析模块
    'wind_speed':       'WSD_main',     # 风速分析模块
    'wind_direction':   'WDR_main',     # 风向分析模块
    'inclination':      'IAN_main',     # 倾角分析模块
    'settlement':       'SET_main',     # 沉降分析模块
    'GPS':              'GPS_main',     # GPS分析模块
    'cable_force':      'CBF_main',     # 索力分析模块
    'humidity':         'HUM_main',     # 湿度分析模块
    'correlation':      'Cor_main',     # 相关性分析模块
    'assessment1':      'ASS1_main',    # 评估分析模块1
    'assessment2':      'ASS2_main',    # 评估分析模块2
    }

clear_files_only(r'Post_processing\rlt_figure')
clear_files_only(r'Post_processing\rlt_table')
clear_files_only(r'Post_processing\ass1_model')
clear_files_only(r'Post_processing\ass2_model')

for task, channels in file_conf.filename_patterns.items():
    if task in analysis_module.keys():
        # 获取任务对应的分析模块
        module_name = analysis_module[task]
        analysis_func = __import__(module_name)  # 动态导入模块
        # 执行分析
        analysis_func
    else:
        print(f"任务 '{task}' 没有对应的分析模块！")

# 后续各类型数据间的分析

# 输出word文档
model, processor = ensure_model()

# 顺序调用，传入已加载对象，避免 b/c/d 重复加载
run_MLLM4pic(model, processor)
run_MLLM4single_type_sum(model, processor)
run_MLLM4corandreg_sum(model, processor)
run_MLLM4all_sum(model, processor)
Pic_identifier_Insert.identifier_insert()
Pic_Insert.pic_insert()
Word_Insert.word_insert()

