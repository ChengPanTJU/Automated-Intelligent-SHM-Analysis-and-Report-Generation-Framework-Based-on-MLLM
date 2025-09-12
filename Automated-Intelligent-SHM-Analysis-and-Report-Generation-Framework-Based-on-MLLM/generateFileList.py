"""
生成数据文件列表，并判断缺失文件

Created on 2024

@author: Gong Fengzong
"""
import os
from datetime import datetime, timedelta


def generate_filenames(start_time: str, end_time: str, folder_path: str, filename_pattern: str, date_pattern: str,
                       time_pattern: str) -> list:
    """
    根据给定的开始时间和结束时间生成文件名列表

    :param start_time: 开始时间，格式为 'YYYY-MM-DD HH' (24小时制)
    :param end_time: 结束时间，格式为 'YYYY-MM-DD HH' (24小时制)
    :param filename_pattern: 文件名的命名规则，例如 'data_{date}-{hour}-VIB.csv'
    :param date_pattern: 日期格式化模式，例如 '%Y-%m-%d'
    :param time_pattern: 小时格式化模式，例如 '%H'

    :return: 生成的文件名列表
    """
    # 转换时间字符串为 datetime 对象
    start_time = datetime.strptime(start_time, '%Y-%m-%d %H')
    end_time = datetime.strptime(end_time, '%Y-%m-%d %H')

    # 检查开始时间是否在结束时间之前
    if start_time > end_time:
        raise ValueError("Start time must be before end time.")

    # 初始化文件名列表
    filenames = []

    # 生成文件名
    current_time = start_time
    while current_time <= end_time:
        # 使用动态日期和小时格式化模式来生成文件名
        file_name = filename_pattern.format(
            date=current_time.strftime(date_pattern),
            hour=current_time.strftime(time_pattern)
        )
        full_name = os.path.join(folder_path, file_name)
        filenames.append((current_time, full_name))
        # 增加一小时
        current_time += timedelta(hours=1)

    return filenames


def generate_filenames_with_paths(start_time: str, end_time: str, base_dir: str, filename_pattern: str,
                                  date_pattern: str, time_pattern: str) -> list:
    """
    根据给定的开始时间和结束时间生成文件名和路径列表
    :param start_time: 开始时间，格式为 'YYYY-MM-DD HH' (24小时制)
    :param end_time: 结束时间，格式为 'YYYY-MM-DD HH' (24小时制)
    :param base_dir: 基础文件夹路径
    :param filename_pattern: 文件名的命名规则，例如 'data_{date}-{hour}-VIB.csv'
    :param date_pattern: 日期格式化模式，例如 '%Y-%m-%d'
    :param time_pattern: 小时格式化模式，例如 '%H'
    :return: 生成的文件路径列表
    """
    # 转换时间字符串为 datetime 对象
    start_time = datetime.strptime(start_time, '%Y-%m-%d %H')
    end_time = datetime.strptime(end_time, '%Y-%m-%d %H')

    # 检查开始时间是否在结束时间之前
    if start_time > end_time:
        raise ValueError("Start time must be before end time.")

    # 初始化文件路径列表
    file_paths = []

    # 生成文件路径
    current_time = start_time
    while current_time <= end_time:
        # 格式化日期和小时
        date_str = current_time.strftime(date_pattern)  # 生成日期字符串
        hour_str = current_time.strftime(time_pattern)  # 生成小时字符串

        # 使用动态日期和小时格式化模式来生成文件名
        file_name = filename_pattern.format(date=date_str, hour=hour_str)

        # 生成文件夹路径，以日期为文件夹名
        folder_path = os.path.join(base_dir, date_str)

        # 确保文件夹存在
        os.makedirs(folder_path, exist_ok=True)

        # 生成完整的文件路径
        full_file_path = os.path.join(folder_path, file_name)

        # 添加到文件路径列表
        file_paths.append((current_time, full_file_path))

        # 增加一小时
        current_time += timedelta(hours=1)

    return file_paths


def check_files_exist(file_paths: list) -> list:
    """
    检查文件是否存在，并返回不存在的文件列表。

    :param file_paths: 文件路径列表
    :return: 不存在的文件路径列表
    """
    # 存放不存在的文件路径
    missing_files = []

    # 遍历所有文件路径，检查每个文件是否存在
    for time_list, file_path in file_paths:
        if not os.path.exists(file_path):  # 如果文件不存在
            missing_files.append(file_path)

    return missing_files


# 测试
if __name__ == '__main__':
    # 输入
    start_time = '2024-11-01 00'
    end_time = '2024-11-01 03'
    date_pattern = '%Y-%m-%d'  # 日期格式
    time_pattern = '%H'        # 小时格式
    filename_pattern = '{date} {hour}-VIB.csv'  # 文件名格式
    base_dir = '/data'

    # 调用生成文件名函数
    file_names = generate_filenames(start_time, end_time, base_dir, filename_pattern, date_pattern, time_pattern)
    # 输出生成的文件名
    print(file_names)

    file_path = generate_filenames_with_paths(start_time, end_time, base_dir, filename_pattern,
                                               date_pattern, time_pattern)
    # 检查文件是否存在
    missing_files = check_files_exist(file_path)
    print(missing_files)

    missing_files = check_files_exist(file_names)
    print(missing_files)
