"""
定义配置文件
Created on 2025

@author: Pan Cheng & Gong Fengzong
Email：2310450@tongji.edu.cn
"""

import os

import pandas as pd


#分析配置文件
class FileConfig:
    def __init__(self):
        self.bridge_name='湾头大桥'
        self.word_file_name = f'{self.bridge_name}评估报告.docx'        # 需要保存的报告名称
        self.start_time = '2019-06-01 00'               # 数据开始时间
        self.end_time = '2020-06-01 00'                 # 数据结束
        self.base_dir = r'G:\宁波数据\明洲大桥\data'        # 数据文件基础路径
        self.date_pattern = '%Y-%m-%d'                  # 日期格式
        self.time_pattern = '%H'                        # 小时格式
        # 数据类型对应的文件名格式
        self.gen_filenames = 'gen_filenames_with_path'      # 选择文件读取函数_with_path
        self.filename_patterns = {
            'wind_speed':   '{date} {hour}-UAN.csv',        # 风速数据文件名格式
            'wind_direction':'{date} {hour}-UAN.csv',       # 风向数据文件名格式
            'GPS':          '{date} {hour}-GPS.csv',        # GPS数据文件名格式
            ##'inclination':  '{date} {hour}-IAN.csv',        # 倾角数据文件名格式
            ##'settlement':   '{date} {hour}-SET.csv',        # 沉降数据文件名格式
            ##'cable_force':   '{date} {hour}-CBF.csv',       # 索力数据文件名格式
            'vibration':    '{date} {hour}-VIB.csv',        # 振动数据文件名格式
            ##'cable_vib':    '{date} {hour}-VIC.csv',        # 拉索振动数据文件名格式
            'temperature':  '{date} {hour}-TMP.csv',        # 温度数据文件名格式，一般是结构温度
            #'humidity':     '{date} {hour}-RHS.csv',        # 湿度数据文件名格式，一般与温度数据同步出现
            'strain':       '{date} {hour}-RSG.csv',        # 应变数据文件名格式
            'displacement': '{date} {hour}-DPM.csv',        # 位移数据文件名格式
            'traffic':      '',                             # 交通荷载文件单独读取
            'correlation':  '',                             # 相关性分析
            #'assessment1':  '',                             # 基于数据的状态评估,自回归
            'assessment2':  '',                             # 基于数据的状态评估,普通线性回归
        }

class WsdConfig:
    def __init__(self):
        self.fs = 10                  # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {        # 每个任务对应的通道
            'rms': [4],                # rms任务分析通道8（只能输入单通道）
            'mean': [4],
            'preprocess': [4],   # 数据质量评估
        }

        self.Pre_conf = {           # 预处理的参数设置
            'up_lim': 20,           # 离群点上界
            'low_lim': 0          # 离群点下界
        }

class CbfConfig:
    def __init__(self):
        self.fs = 0.1                  # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {        # 每个任务对应的通道
            'rms': [2*i-1 for i in range(1,37)],                # rms任务分析通道8（只能输入单通道）
            'mean': [2*i-1 for i in range(1,37)],
            'preprocess': [2*i-1 for i in range(1,37)],   # 数据质量评估
            #'max_min': [2*i-1 for i in range(1,37)],
        }

class WdrConfig:
    def __init__(self):
        self.fs = 0.1  # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {  # 每个任务对应的通道（只能输入单通道）
            'rms': [5],                # rms任务分析通道8（只能输入单通道）
            'mean': [5],
            'preprocess': [5],   # 数据质量评估
        }

class IanConfig:
    def __init__(self):
        self.fs = 0.00055555               # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {    # 每个任务对应的通道（只能输入单通道）
            'preprocess': [1,3,5,7,9,11],     # 数据质量预评估
            'rms': [1,3,5,7,9,11],
            'mean': [1,3,5,7,9,11],           # 主梁顶板温度
            'max_min':[1,3,5,7,9,11],
        }
        self.Pre_conf = {     # 预处理的参数设置
            'up_lim': 90,     # 离群点上界
            'low_lim': -90    # 离群点下界
        }

class SetConfig:
    def __init__(self):
        self.fs = 1 # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {  # 每个任务对应的通道（只能输入单通道）
            'preprocess': [i+1 for i in range(22)],
            'mean': [i+1 for i in range(22)],        # 主梁顶板应变
            'rms': [i+1 for i in range(22)],
            'max_min':[i+1 for i in range(22)],
        }

class GpsConfig:
    def __init__(self):
        self.fs = 10 # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {  # 每个任务对应的通道（只能输入单通道）
            'preprocess': [i+1 for i in range(9)],
            'mean': [i+1 for i in range(9)],        # 主梁顶板应变
            'rms': [i+1 for i in range(9)],
            #'max_min': [i + 1 for i in range(9)],
        }

class AccConfig:
    def __init__(self):
        self.fs = 50                   # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {        # 每个任务对应的通道
            'rms': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],                # rms任务分析通道
            #'mean': [1,2,3,4,5,6,7,8],
            'OMA1': [4,6,8,10,11,12],          # 主梁模态分析
            'OMA2': [16,17,18,19,20,21],                  # 拱肋模态分析
            #'OMA3': [4, 5, 6, 7, 8],            # 主塔模态分析
            'preprocess': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],   # 数据质量评估
        }

        self.Pre_conf = {           # 预处理的参数设置
            'up_lim': 500,           # 离群点上界
            'low_lim': -500          # 离群点下界
        }

        self.SSI_conf = {                # SSI方法的参数设置
            'order': 60,                # SSI的阶数
            'err': [0.02, 0.1, 0.9],     # 稳定点判断条件
            'order_num': 10,             # 统计的模态阶数
            'filter_band': [0.1, 5],    # 滤波区间
        }

class VicConfig:
    def __init__(self):
        self.fs = 64                   # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {        # 每个任务对应的通道
            'rms': [1,2,3,4,5,6,7,8,9,10],                # rms任务分析通道8（只能输入单通道）
            'mean': [1,2,3,4,5,6,7,8,9,10],
            'OMA': [1,2,3,4,5,6,7,8,9,10],          # 拉索模态分析，一根拉索一个结果
            'preprocess': [1,2,3,4,5,6,7,8,9,10],   # 数据质量评估
        }

        self.Pre_conf = {           # 预处理的参数设置
            'up_lim': 500,           # 离群点上界
            'low_lim': -500          # 离群点下界
        }

        self.SSI_conf = {                # SSI方法的参数设置
            'order': 60,                # SSI的阶数
            'err': [0.02, 0.1, 0.9],     # 稳定点判断条件
            'order_num': 10,             # 统计的模态阶数
            'filter_band': [0.1, 10],    # 滤波区间
        }
        self.ACF_conf = {
            'max_lag': 800,                # ACF（用于识别阻尼比）的时间滞后个数
            'filter_band': [[0.45, 0.5],
                            [0.61, 0.65],
                            [1.63, 1.7]],  # 滤波区间
        }

class TmpConfig:
    def __init__(self):
        self.fs = 1           # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {    # 每个任务对应的通道（只能输入单通道）
            'preprocess': [6,7,8,9,10,11,12,13,14,15,16],     # 数据质量预评估
            'rms': [6,7,8,9,10,11,12,13,14,15,16],
            'mean': [6,7,8,9,10,11,12,13,14,15,16],           # 主梁顶板温度
        }

        self.Pre_conf = {     # 预处理的参数设置
            'up_lim': 90,     # 离群点上界
            'low_lim': -20    # 离群点下界
        }

class HumConfig:
    def __init__(self):
        self.fs = 1             # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {    # 每个任务对应的通道（只能输入单通道）
            'preprocess': [2],     # 数据质量预评估
            'rms': [2],
            'mean': [2],           # 主梁顶板温度
        }

        self.Pre_conf = {     # 预处理的参数设置
            'up_lim': 95,     # 离群点上界
            'low_lim': 5    # 离群点下界
        }

class StrConfig:
    def __init__(self):
        self.fs = 1  # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {  # 每个任务对应的通道（只能输入单通道）
            'preprocess': [i+1 for i in range(28)],
            'mean': [i+1 for i in range(28)],
            'rms': [i+1 for i in range(28)],
            #'max_min': [i + 1 for i in range(16)],
        }

class DisConfig:
    def __init__(self):
        self.fs = 1 # 采样频率
        self.dt = 1 / self.fs
        self.tasks_channels = {  # 每个任务对应的通道（只能输入单通道）
            'preprocess': [1, 2,3,4],
            'mean': [1, 2,3,4],        # 主梁顶板应变
            'rms': [1, 2,3,4],
            #'max_min': [1, 2,3,4],
        }

class TrafConfig:
    def __init__(self):
        self.traffic_path = r"G:\宁波数据\明洲大桥\称重2019-2021"  #文件路径
        self.traffic_time =2                                            # 时间列的列数
        self.time_format = '%Y-%m-%d %H:%M:%S.%f'                   # 时间的格式'%Y-%m-%d %H:%M:%S.%f'   '%Y年%m月%d日 %H:%M:%S'
        self.lane_num = 1                                              # 车道号列数
        self.speed=41                                               #车速列数
        self.total_weight = 6                                         # 称重的列数
        self.axle_weight = [24, 25, 26, 27, 28, 29, 30, 31]             # 轴重列

class CorConfig:
    def __init__(self):
        # 初始化相关性分析任务
        self.tasks_channels = {
            'Correlation1': {
                'temperature': ['mean', [ 8]],
                'strain': ['mean', [6,7,8,9,10]],
            },  # 相关性分析1
            'Correlation2': {
                'temperature': ['mean', [10]],
                'displacement': ['mean', [1, 2,3,4]],
            },  # 相关性分析2
            'Correlation3': {
                'traffic': ['common_analysis', [1, 2]],
                'displacement': ['rms', [1, 2, 3, 4]],
            },  # 相关性分析2
            'Correlation4': {
                'traffic': ['common_analysis', [1, 2]],
                'strain': ['rms', [6,7,8,9,10]],
            },  # 相关性分析2
            'Correlation5': {
                'traffic': ['common_analysis', [1, 2]],
                'vibration': ['rms', [4,6,8,10,11,12]],
            },  # 相关性分析2
            'Correlation6': {
                'traffic': ['common_analysis', [1, 2]],
                'vibration': ['OMA1', [1, 2, 3, 4]],
            },  # 相关性分析2
        }

class Ass1Config:
    def __init__(self):
        # 初始化评估任务
        self.tasks_channels = {
            'strain': ['mean', [5,6,7,8],'dl'],
            'temperature': ['mean', [ 7,8,9,10],'reg'],
            'vibration': ['rms', [4,6,8,10,11,12],'reg'],
            'displacement':['rms',[1,2,3,4],'dl'],
            'traffic': ['common_analysis', [1, 2], 'dl'],
        }

class Ass2Config:
    def __init__(self):
        self.tasks_channels = {
            'regression1': {
                'strain': ['mean', [9,10],'dl'],
                'temperature': ['mean', [ 7]],
                'traffic': ['common_analysis', [1, 2]]
            },  # 评估任务1，默认第一项是被拟合对象，之后的项是拟合项
            'regression2': {
                'vibration': ['OMA1', [1, 2, 3],'dl'],
                'temperature': ['mean', [ 7,8,10]],
                'traffic': ['common_analysis', [1, 2]]
            },  # 评估任务3，默认第一项是被拟合对象，之后的项是拟合项分析中有一项为主梁加速度的频率时，OMA对应的通道号为频率阶数
            'regression3': {
                'displacement': ['mean', [1, 2, 3,4], 'dl'],
                'traffic': ['common_analysis', [1, 2]],
                'temperature': ['mean', [ 7,8,9,10]],
            },  # 评估任务2，默认第一项是被拟合对象，之后的项是拟合项
            'regression4': {
                'GPS': ['mean', [3,6,9], 'dl'],
                'traffic': ['common_analysis', [1, 2]],
                'displacement': ['mean', [1, 2, 3,4]],
            },  # 评估任务2，默认第一项是被拟合对象，之后的项是拟合项
        }

tmp_config = TmpConfig()    # 实例化TmpConfig类
str_config = StrConfig()    # 实例化StrConfig类
traf_config = TrafConfig()  # 实例化TrafConfig类
acc_config = AccConfig()    # 实例化AccConfig类
dis_config = DisConfig()    # 实例化DisConfig类
vic_config = VicConfig()    # 实例化VicConfig类
wsd_config = WsdConfig()    # 实例化WsdConfig类
wdr_config = WdrConfig()    # 实例化WdrConfig类
gps_config = GpsConfig()    # 实例化GpsConfig类
ian_config = IanConfig()    # 实例化IanConfig类
set_config = SetConfig()    # 实例化SetConfig类
cbf_config = CbfConfig()    # 实例化CbfConfig类
hum_config = HumConfig()    # 实例化CbfConfig类
cor_config = CorConfig()    # 实例化CorConfig类
ass1_config = Ass1Config()  # 实例化CorConfig类
ass2_config = Ass2Config()  # 实例化CorConfig类

class OutputConfig:
    def __init__(self):
        self.tmp_config = tmp_config
        self.str_config = str_config
        self.traf_config = traf_config
        self.acc_config = acc_config
        self.dis_config = dis_config
        self.vic_config = vic_config
        self.wsd_config = wsd_config
        self.wdr_config = wdr_config
        self.gps_config = gps_config
        self.ian_config = ian_config
        self.set_config = set_config
        self.cbf_config = cbf_config
        self.hum_config = hum_config
        self.cor_config = cor_config
        self.ass1_config = ass1_config
        self.ass2_config = ass2_config
        # 获取当前脚本所在路径
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建相对路径到'../data/results/'文件夹，对应一级文件夹
        self.results_dir = os.path.join(self.current_dir, 'Post_processing')
        # 任务字典
        acctasks = {}
        for task_name, channels in self.acc_config.tasks_channels.items():
            if task_name == 'preprocess':
                acctasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('acc', task_name),
                    'figure_path': self.get_figure_path2('acc', task_name),  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('acc', 'raw_data', channels),
                    'figure_identifier': 'figure_acc_time_history'
                }
            elif 'OMA' in task_name:
                acctasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('acc', task_name),
                    'figure_path': self.get_figure_path3('acc', task_name),  # 输入数据类型标识符，任务名称
                    'figure_identifier': f'figure_acc_{task_name}'
                }
            else:
                acctasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('acc', task_name),
                    'figure_path': self.get_figure_path1('acc', task_name, channels),  # 输入数据类型标识符，任务名称
                    'sum_table_path': self.get_save_path3('acc', task_name),
                    'table_identifier': f'table_acc_{task_name}',
                    'figure_identifier': f'figure_acc_{task_name}',
                }
        acctasks['word_identifier']='acc_summary'

        wsdtasks = {}
        for task_name, channels in self.wsd_config.tasks_channels.items():
            if task_name == 'preprocess':
                wsdtasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('wsd', task_name),
                    'figure_path': self.get_figure_path2('wsd', task_name),  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('wsd', 'raw_data', channels),
                    'figure_identifier': 'figure_wsd_time_history'
                }
            else:
                wsdtasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('wsd', task_name),
                    'figure_path': self.get_figure_path1('wsd', task_name, channels),  # 输入数据类型标识符，任务名称
                    'sum_table_path': self.get_save_path3('wsd', task_name),
                    'table_identifier': f'table_wsd_{task_name}',
                    'figure_identifier': f'figure_wsd_{task_name}'
                }
        wsdtasks['word_identifier'] = 'wsd_summary'

        gpstasks = {}
        for task_name, channels in self.gps_config.tasks_channels.items():
            if task_name == 'preprocess':
                gpstasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('gps', task_name),
                    'figure_path': self.get_figure_path2('gps', task_name)  ,  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('gps', 'raw_data', channels),
                    'figure_identifier':'figure_gps_time_history'
                }
            else:
                gpstasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('gps', task_name),
                    'figure_path': self.get_figure_path1('gps', task_name,channels),  # 输入数据类型标识符，任务名称
                    'sum_table_path': self.get_save_path3('gps', task_name),
                    'table_identifier': f'table_gps_{task_name}',
                    'figure_identifier': f'figure_gps_{task_name}'
                }
        gpstasks['word_identifier'] = 'gps_summary'

        cbftasks = {}
        for task_name, channels in self.cbf_config.tasks_channels.items():
            if task_name == 'preprocess':
                cbftasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('cbf', task_name),
                    'figure_path': self.get_figure_path2('cbf', task_name),  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('cbf', 'raw_data', channels),
                    'figure_identifier': 'figure_cbf_time_history'
                }
            else:
                cbftasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('cbf', task_name),
                    'figure_path': self.get_figure_path1('cbf', task_name, channels),  # 输入数据类型标识符，任务名称
                    'sum_table_path': self.get_save_path3('cbf', task_name),
                    'table_identifier': f'table_cbf_{task_name}',
                    'figure_identifier': f'figure_cbf_{task_name}'
                }
        cbftasks['word_identifier'] = 'cbf_summary'

        iantasks = {}
        for task_name, channels in self.ian_config.tasks_channels.items():
            if task_name=='preprocess':
                iantasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('ian', task_name),
                    'figure_path': self.get_figure_path2('ian', task_name)  ,  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('ian', 'raw_data', channels),
                    'figure_identifier': 'figure_ian_time_history'
                }
            else:
                iantasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('ian', task_name),
                    'figure_path': self.get_figure_path1('ian', task_name,channels),  # 输入数据类型标识符，任务名称
                    'sum_table_path': self.get_save_path3('ian', task_name),
                    'table_identifier': f'table_ian_{task_name}',
                    'figure_identifier': f'figure_ian_{task_name}'
                }
        iantasks['word_identifier'] = 'ian_summary'

        settasks = {}
        for task_name, channels in self.set_config.tasks_channels.items():
            if task_name == 'preprocess':
                settasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('set', task_name),
                    'figure_path': self.get_figure_path2('set', task_name)  ,  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('set', 'raw_data', channels),
                    'figure_identifier':'figure_set_time_history'
                }
            else:
                settasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('set', task_name),
                    'figure_path': self.get_figure_path1('set', task_name,channels),  # 输入数据类型标识符，任务名称
                    'sum_table_path': self.get_save_path3('set', task_name),
                    'table_identifier': f'table_set_{task_name}',
                    'figure_identifier': f'figure_set_{task_name}'
                }
        settasks['word_identifier'] = 'set_summary'

        wdrtasks = {}
        for task_name, channels in self.wdr_config.tasks_channels.items():
            if task_name == 'preprocess':
                wdrtasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('wdr', task_name),
                    'figure_path': self.get_figure_path2('wdr', task_name),  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('wdr', 'raw_data', channels),
                    'figure_identifier': 'figure_wdr_time_history'
                }
            elif task_name=='mean':
                wdrtasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('wdr', task_name),
                    'figure_path': self.get_figure_path5('wdr', task_name, channels),  # 输入数据类型标识符，任务名称
                    'figure_identifier': f'figure_wdr_{task_name}'
                }
            else:
                wdrtasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('wdr', task_name),
                    'figure_path': self.get_figure_path1('wdr', task_name, channels),  # 输入数据类型标识符，任务名称
                    'figure_identifier': f'figure_wdr_{task_name}'
                }
        wdrtasks['word_identifier'] = 'wdr_summary'

        victasks = {}
        for task_name, channels in self.vic_config.tasks_channels.items():
            if task_name == 'preprocess':
                victasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('vic', task_name),
                    'figure_path': self.get_figure_path2('vic', task_name),  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('vic', 'raw_data', channels),
                    'figure_identifier': 'figure_vic_time_history'
                }
            elif 'OMA' in task_name:
                victasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path2('vic', task_name, channels),
                    'figure_path': self.get_figure_path4('vic', task_name, channels),  # 输入数据类型标识符，任务名称
                    'figure_identifier': f'figure_vic_{task_name}'
                }
            else:
                victasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('vic', task_name),
                    'figure_path': self.get_figure_path1('vic', task_name, channels),  # 输入数据类型标识符，任务名称
                    'sum_table_path': self.get_save_path3('vic', task_name),
                    'table_identifier': f'table_vic_{task_name}',
                    'figure_identifier': f'figure_vic_{task_name}'
                }
        victasks['word_identifier'] = 'vic_summary'

        tmptasks = {}
        for task_name, channels in self.tmp_config.tasks_channels.items():
            if task_name=='preprocess':
                tmptasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('tmp', task_name),
                    'figure_path': self.get_figure_path2('tmp', task_name)  ,  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('tmp', 'raw_data', channels),
                    'figure_identifier': 'figure_tmp_time_history'
                }
            else:
                tmptasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('tmp', task_name),
                    'figure_path': self.get_figure_path1('tmp', task_name,channels),  # 输入数据类型标识符，任务名称
                    'sum_table_path': self.get_save_path3('tmp', task_name),
                    'table_identifier': f'table_tmp_{task_name}',
                    'figure_identifier': f'figure_tmp_{task_name}'
                }
        tmptasks['word_identifier'] = 'tmp_summary'

        humtasks = {}
        for task_name, channels in self.hum_config.tasks_channels.items():
            if task_name=='preprocess':
                humtasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('hum', task_name),
                    'figure_path': self.get_figure_path2('hum', task_name)  ,  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('hum', 'raw_data', channels),
                    'figure_identifier': 'figure_hum_time_history'
                }
            else:
                humtasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('hum', task_name),
                    'figure_path': self.get_figure_path1('hum', task_name,channels),  # 输入数据类型标识符，任务名称
                    'sum_table_path': self.get_save_path3('hum', task_name),
                    'table_identifier': f'table_hum_{task_name}',
                    'figure_identifier': f'figure_hum_{task_name}'
                }
        humtasks['word_identifier'] = 'hum_summary'

        strtasks = {}
        for task_name, channels in self.str_config.tasks_channels.items():
            if task_name == 'preprocess':
                strtasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('str', task_name),
                    'figure_path': self.get_figure_path2('str', task_name),  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('str', 'raw_data', channels),
                    'figure_identifier': 'figure_str_time_history'
                }
            else:
                strtasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('str', task_name),
                    'figure_path': self.get_figure_path1('str', task_name,channels),  # 输入数据类型标识符，任务名称
                    'sum_table_path': self.get_save_path3('str', task_name),
                    'table_identifier': f'table_str_{task_name}',
                    'figure_identifier': f'figure_str_{task_name}'
                }
        strtasks['word_identifier'] = 'str_summary'

        distasks = {}
        for task_name, channels in self.dis_config.tasks_channels.items():
            if task_name == 'preprocess':
                distasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('dis', task_name),
                    'figure_path': self.get_figure_path2('dis', task_name)  ,  # 输入数据类型标识符，任务名称
                    'raw_data_figure_path': self.get_figure_path1('dis', 'raw_data', channels),
                    'figure_identifier':'figure_dis_time_history'
                }
            else:
                distasks[task_name] = {
                    'channels': channels,
                    'save_path': self.get_save_path('dis', task_name),
                    'figure_path': self.get_figure_path1('dis', task_name,channels),  # 输入数据类型标识符，任务名称
                    'sum_table_path': self.get_save_path3('dis', task_name),
                    'table_identifier': f'table_dis_{task_name}',
                    'figure_identifier': f'figure_dis_{task_name}'
                }
        distasks['word_identifier'] = 'dis_summary'

        cortasks = {}
        for task_name, data_infor in self.cor_config.tasks_channels.items():
            index=next((i for i, k in enumerate(self.cor_config.tasks_channels) if k == task_name), -1)+1
            cortasks[task_name] = {
                list(data_infor.keys())[0]: list(data_infor.values())[0],
                list(data_infor.keys())[1]: list(data_infor.values())[1],
                'figure_path': self.get_figure_path6( task_name,list(data_infor.values())[0][1],list(data_infor.values())[1][1]),  # 输入数据类型标识符，任务名称
                'figure_name': self.get_figure_name( list(data_infor.keys())[0],list(data_infor.keys())[1],index),
            }
        cortasks['figure_identifier'] = {'figure_identifier': 'figure_corr','figure_path': ''}
        cortasks['word_identifier'] = 'cor_summary'

        traffictasks = {}
        traffictasks['common_analysis'] = {
                'channels':[1,2],
                'save_path': self.get_save_path('traf','common_analysis'),
                'figure_path': [self.get_figure_path2('traf', 'weight_hour'),self.get_figure_path2('traf', 'weight_probability'),
                                self.get_figure_path2('traf', 'count')], # 输入数据类型标识符，任务名称
                'figure_identifier': 'figure_traf_common_analysis'
            }
        if  pd.notna(traf_config.lane_num):
            traffictasks['lane'] = {
                'figure_path': [self.get_figure_path2('traf', f'lane_prob'),self.get_figure_path2('traf', f'lane_count')],
                'figure_identifier': 'figure_traf_lane'
                }
        if  pd.notna(traf_config.speed):
            traffictasks['speed'] = {
                'figure_path': self.get_figure_path2('traf', 'speed'),
                'figure_identifier': 'figure_traf_speed'
                }
        if  pd.notna(traf_config.axle_weight).any():
            traffictasks['axle'] = {
                'figure_path': self.get_figure_path2('traf', f'axle_prob'),
                'figure_identifier': 'figure_traf_axle'
                }
        traffictasks['word_identifier'] = 'traf_summary'

        ass1tasks = {}
        for task_name, channels in self.ass1_config.tasks_channels.items():
            index = next((i for i, k in enumerate(self.ass1_config.tasks_channels) if k == task_name), -1) + 1
            ass1tasks[task_name]={
                'channels':channels,
                'save_path': self.get_save_path4(task_name, channels[0],channels[1],channels[2]),
                'figure_path': self.get_figure_path1(f'ass1_{task_name}', channels[0],channels[1]),  # 输入数据类型标识符，任务名称
                'figure_name':self.get_figure_name2(task_name, channels[0], index,'自回归'),  # 输入数据类型标识符，任务名称
            }
        ass1tasks['figure_identifier'] = {'figure_identifier': 'figure_ass1','figure_path': ''}
        ass1tasks['word_identifier'] = 'ass1_summary'

        ass2tasks = {}
        for task_name, data_infor in self.ass2_config.tasks_channels.items():
            data_list = list(data_infor.keys())
            index=next((i for i, k in enumerate(self.ass2_config.tasks_channels) if k == task_name), -1)+1
            ass2tasks[task_name] = {
                'save_path': self.get_save_path5(data_infor),
                'figure_path': self.get_figure_path1( f'ass2_{data_list[0]}',data_infor[data_list[0]][0],data_infor[data_list[0]][1]),  # 输入数据类型标识符，任务名称
                'figure_name': self.get_figure_name2( data_list[0],data_infor[data_list[0]][0],index,'多元线性回归'),
            }
        ass2tasks['figure_identifier'] = {'figure_identifier': 'figure_ass2','figure_path': ''}
        ass2tasks['word_identifier'] = 'ass2_summary'

        self.tasks = {
            'vibration': acctasks,
            'temperature': tmptasks,
            'strain': strtasks,
            'traffic': traffictasks,
            'displacement': distasks,
            'cable_vib': victasks,
            'wind_speed': wsdtasks,
            'wind_direction': wdrtasks,
            'inclination': iantasks,
            'settlement': settasks,
            'GPS': gpstasks,
            'cable_force': cbftasks,
            'correlation': cortasks,
            'humidity': humtasks,
            'assessment1':ass1tasks,
            'assessment2': ass2tasks,
        }
###################保存csv#########################################
    def get_save_path(self, type, task_name):
        """
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
        file_name = f"rlt_{type}_{task_name}.csv"
        dir2 = f"rlt_table"
        path = os.path.join(self.results_dir, dir2)
        save_path = os.path.join(path, file_name)

        return save_path

    def get_save_path2(self, type, task_name,channels):
        """
        针对拉索加速度的OMA
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
        csv_path_list=[]
        for i in range(len(channels)):
            file_name = f"rlt_{type}_{task_name}_{channels[i]}.csv"
            dir2 = f"rlt_table"
            path = os.path.join(self.results_dir, dir2)
            save_path = os.path.join(path, file_name)
            csv_path_list.append(save_path)
        return csv_path_list

    def get_save_path3(self, type, task_name):
        """
        用于生成任务的汇总表
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
        file_name = f"rlt_{type}_{task_name}_sum.csv"
        dir2 = f"rlt_table"
        path = os.path.join(self.results_dir, dir2)
        save_path = os.path.join(path, file_name)
        return save_path

    def get_save_path4(self, data_type, task_name,channels,method):
        """
        用于ass1评估任务的save_path生成
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
        model_path_list=[]
        if method=='dl':
            for i in range(len(channels)):
                file_name = f"ass1_{data_type}_{task_name}_{channels[i]}.pth"
                dir2 = f"ass1_model"
                path = os.path.join(self.results_dir, dir2)
                save_path = os.path.join(path, file_name)
                model_path_list.append(save_path)
            return model_path_list
        else:
            for i in range(len(channels)):
                file_name = f"ass1_{data_type}_{task_name}_{channels[i]}.pkl"
                dir2 = f"ass1_model"
                path = os.path.join(self.results_dir, dir2)
                save_path = os.path.join(path, file_name)
                model_path_list.append(save_path)
            return model_path_list

    def get_save_path5(self,data_infor):
        """
        用于ass2评估任务的save_path生成
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
        data_type_list=list(data_infor.keys())
        model_path_list=[]
        target_data=data_type_list[0]
        target_task=data_infor[target_data][0]
        target_channels=data_infor[target_data][1]
        method=data_infor[target_data][2]
        source_info = []
        for i in data_type_list[1:]:
            source_info.append(i)
            source_info.append(data_infor[i][0])
            source_info.append(data_infor[i][1])
        if method=='dl':
            for i in range(len(target_channels)):
                file_name = f"ass2_{target_data}_{target_task}_{target_channels[i]}_{source_info}.pth"
                dir2 = f"ass2_model"
                path = os.path.join(self.results_dir, dir2)
                save_path = os.path.join(path, file_name)
                model_path_list.append(save_path)
            return model_path_list
        else:
            for i in range(len(target_channels)):
                file_name = f"ass2_{target_data}_{target_task}_{target_channels[i]}_{source_info}.pkl"
                dir2 = f"ass2_model"
                path = os.path.join(self.results_dir, dir2)
                save_path = os.path.join(path, file_name)
                model_path_list.append(save_path)
            return model_path_list

#################保存图片#########################################################
    def get_figure_name(self, type1, type2,index):
        """
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
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
        data_kind1=label_infor[type1]
        data_kind2 = label_infor[type2]
        fig_name = f"子图{index}：{data_kind1}与{data_kind2}相关性分析结果"
        return fig_name

    def get_figure_name2(self, data_type,task_name,index,method):
        """
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
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
            'traffic': '交通荷载',
        }
        task_infor={
            'mean':'均值',
            'rms':'均方根',
            'OMA1':'频率',
            'common_analysis': ''
        }
        data_kind=label_infor[data_type]
        task_kind = task_infor[task_name]
        fig_name = f"子图{index}：{data_kind}{task_kind}数据评估结果({method})"
        return fig_name

    def get_figure_path1(self, type, task_name, channels):
        """
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
        fig_path_list=[]
        for i in range(len(channels)):
            file_name = f"figure_{type}_{task_name}_{channels[i]}.png"
            dir2 = f"rlt_figure"
            path = os.path.join(self.results_dir, dir2)
            save_path = os.path.join(path, file_name)
            fig_path_list.append(save_path)

        return fig_path_list

    def get_figure_path2(self, type, task_name):
        """
        专门针对图像中不存在通道号的任务
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
        file_name = f"figure_{type}_{task_name}.png"
        dir2 = f"rlt_figure"
        path = os.path.join(self.results_dir, dir2)
        save_path = os.path.join(path, file_name)

        return save_path

    def get_figure_path3(self, type, task_name):
        """
        专门针对主梁加速度数据的oma
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
        file_name1 = f"figure_{type}_{task_name}_fre.png"
        file_name2 = f"figure_{type}_{task_name}_dp.png"
        file_name3 = f"figure_{type}_{task_name}_phi.png"
        dir2 = f"rlt_figure"
        path = os.path.join(self.results_dir, dir2)
        save_path1 = os.path.join(path, file_name1)
        save_path2 = os.path.join(path, file_name2)
        save_path3 = os.path.join(path, file_name3)

        return [save_path1,save_path2,save_path3]

    def get_figure_path4(self, type, task_name, channels):
        """
        针对VIC的OMA量身定制
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
        fig_path_list = []
        for i in range(len(channels)):
            file_name1 = f"figure_{type}_{task_name}_{channels[i]}_fre.png"
            file_name2 = f"figure_{type}_{task_name}_{channels[i]}_dp.png"
            dir2 = f"rlt_figure"
            path = os.path.join(self.results_dir, dir2)
            save_path1 = os.path.join(path, file_name1)
            save_path2 = os.path.join(path, file_name2)
            fig_path_list.append(save_path1)
            fig_path_list.append(save_path2)
        return fig_path_list

    def get_figure_path5(self, type, task_name, channels):
        """
        针对WDR的mean量身定制
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
        fig_path_list = []
        for i in range(len(channels)):
            file_name1 = f"figure_{type}_{task_name}_{channels[i]}_raw.png"
            file_name2 = f"figure_{type}_{task_name}_{channels[i]}_rose.png"
            dir2 = f"rlt_figure"
            path = os.path.join(self.results_dir, dir2)
            save_path1 = os.path.join(path, file_name1)
            save_path2 = os.path.join(path, file_name2)
            fig_path_list.append(save_path1)
            fig_path_list.append(save_path2)
        return fig_path_list

    def get_figure_path6(self,  task_name, channel1,channel2):
        """
        针对ass1
        根据任务名称生成保存结果的文件路径和文件名。
        假设结果文件名为：<任务名称>_results.csv
        """
        # 生成文件保存路径
        fig_path_list = []
        for i in range(len(channel1)):
            for j in range(len(channel2)):
                file_name = f"figure_{task_name}_{channel1[i]}_{channel2[j]}.png"
                dir2 = f"rlt_figure"
                path = os.path.join(self.results_dir, dir2)
                save_path = os.path.join(path, file_name)
                fig_path_list.append(save_path)
        file_name = f"figure_{task_name}.png"
        dir2 = f"rlt_figure"
        path = os.path.join(self.results_dir, dir2)
        save_path = os.path.join(path, file_name)
        fig_path_list.append(save_path)
        return fig_path_list


if __name__ == '__main__':
    OutConfig = OutputConfig()
    out_config = OutConfig.tasks
    print(out_config['assessment2']['regression1']['figure_name'])



