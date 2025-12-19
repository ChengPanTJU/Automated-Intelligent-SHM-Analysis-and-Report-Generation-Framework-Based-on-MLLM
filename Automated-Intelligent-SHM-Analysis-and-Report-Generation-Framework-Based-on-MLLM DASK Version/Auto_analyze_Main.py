import matplotlib
import Pic_Insert
import Pic_identifier_Insert
import Word_Insert
from _00_common_model import ensure_model
from _01_MLLM4pic import run_MLLM4pic
from _02_MLLM4single_type_sum import run_MLLM4single_type_sum
from _03_MLLM4corandreg_sum import run_MLLM4corandreg_sum
from _04_MLLM4all_sum import run_MLLM4all_sum
from config import FileConfig, OutputConfig
import importlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*renamed to.*ensure_all_finite.*",
    category=FutureWarning,
)

num_workers=8

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

import os
import shutil

def clear_folder_only(folder_path):
    if not os.path.exists(folder_path):
        print(f"{folder_path} 文件夹不存在！")
        return

    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"删除 {path} 失败: {e}")


if __name__ == "__main__":
    OutConfig = OutputConfig()
    out_config = OutConfig.tasks
    # 加载配置参数
    file_conf = FileConfig()
    # 任务与分析函数的映射
    # 任务与分析函数的映射
    analysis_module = {
        'temperature': 'TMP_main',  # 温度分析模块
        'strain': 'STR_main',  # 应变分析模块
        'traffic': 'TRAF_main',  # 交通流分析模块
        'vibration': 'ACC_main',  # 加速度分析模块
        'cable_vib': 'VIC_main',  # 拉索加速度分析模块
        'displacement': 'DIS_main',  # 位移分析模块
        'wind_speed': 'WSD_main',  # 风速分析模块
        'wind_direction': 'WDR_main',  # 风向分析模块
        'inclination': 'IAN_main',  # 倾角分析模块
        'settlement': 'SET_main',  # 沉降分析模块
        'GPS': 'GPS_main',  # GPS分析模块
        'cable_force': 'CBF_main',  # 索力分析模块
        'humidity': 'HUM_main',  # 湿度分析模块
        'correlation': 'Cor_main',  # 相关性分析模块
        'assessment1': 'ASS1_main',  # 评估分析模块1
        'assessment2': 'ASS2_main',  # 评估分析模块2
    }
    '''
    clear_folder(r'PostProcessing\vibration')
    clear_folder(r'PostProcessing\GPS')
    clear_folder(r'PostProcessing\wind_speed')
    clear_folder(r'PostProcessing\wind_direction')
    clear_folder(r'PostProcessing\temperature')
    clear_folder(r'PostProcessing\humidity')
    clear_folder(r'PostProcessing\strain')
    clear_folder(r'PostProcessing\displacement')
    clear_folder(r'PostProcessing\traffic')
    clear_folder(r'PostProcessing\LLM_result')
    clear_folder(r'PostProcessing\assessment2')
    clear_folder(r'PostProcessing\assessment1')
    clear_folder(r'PostProcessing\correlation')
    clear_folder(r'PostProcessing\inclination')
    clear_folder_only(r'RowResults')'''

    for task, channels in file_conf.filename_patterns.items():
        if task in analysis_module.keys():
            # 获取任务对应的分析模块
            module_name = analysis_module[task]
            module = importlib.import_module(f"MainFunction.{module_name}")
            # 获取函数名
            analysis_func = getattr(module, module_name)
            analysis_func(num_workers)
        else:
            print(f"任务 '{task}' 没有对应的分析模块！")
    # 输出word文档
    model, processor = ensure_model()
    run_MLLM4pic(model, processor)
    run_MLLM4single_type_sum(model, processor)
    run_MLLM4corandreg_sum(model, processor)
    run_MLLM4all_sum(model, processor)
    Pic_identifier_Insert.identifier_insert()
    Pic_Insert.pic_insert()
    Word_Insert.word_insert()