"""
Created on 2024

@author: Gong Fengzong
"""
import pandas as pd
from config import TrafConfig
import os


def Traf_analyze(file_path, traf_conf):
    # 读取CSV文件（仅定义前4列，假设第三列是时间，格式为'yyyy-MM-dd HH:mm:ss.SSS'）
    # 如果文件不存在，则给各任务赋值为nan
    if not os.path.exists(file_path):
        return []
    data = pd.read_csv(file_path)
    data[traf_conf.time_col] = pd.to_datetime(data[traf_conf.time_col], format=traf_conf.time_format)
    data['date'] = data[traf_conf.time_col].dt.date
    data['hour'] = data[traf_conf.time_col].dt.hour

    # 按日期和小时分组，并统计每小时的车辆数量和流量
    hourly_stats = data.groupby(['date', 'hour']).agg(
        Number=(traf_conf.id_col, 'count'),  # 假设ID的计数代表车辆数量
        Mass=(traf_conf.load_col, 'sum')   # 假设方向字段代表车辆质量
    ).reset_index()

    # 获取数据中的最早和最晚月份
    start_month = data[traf_conf.time_col].min().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_month = (data[traf_conf.time_col].max().replace(day=1, hour=23, minute=59, second=59,
                                                        microsecond=0) + pd.offsets.MonthEnd(0))

    # 生成完整的时间范围（以小时为单位）
    time_range = pd.date_range(start=start_month, end=end_month, freq='H')

    # 创建完整的时间列表 DataFrame
    full_time_df = pd.DataFrame({'date': time_range.date, 'hour': time_range.hour})

    # 将统计结果合并到完整时间列表中
    merged_stats = pd.merge(full_time_df, hourly_stats, on=['date', 'hour'], how='left')
    merged_stats['Number'] = merged_stats['Number'].fillna(0).astype(int)
    merged_stats['Mass'] = merged_stats['Mass'].fillna(0)

    # 获取当前脚本所在的根目录
    current_script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
    folder_path = os.path.join(project_root, 'result_folder_traffic')
    os.makedirs(folder_path, exist_ok=True)

    # 保存每小时的统计结果
    first_date = data.loc[0, traf_conf.time_col]
    file_name = first_date.strftime('%Y-%m')  # 提取格式为 'YYYY-MM'
    hourly_file_name = f"{file_name}.csv"
    hourly_file_path = os.path.join(folder_path, hourly_file_name)
    merged_stats.to_csv(hourly_file_path, index=False)

    return merged_stats


if __name__ == '__main__':
    filename = r'G:/2 Data/MinzhouBridge Data/WIM/2019.9.csv'
    traf_conf = TrafConfig()
    hourly_traf = Traf_analyze(filename, traf_conf)
