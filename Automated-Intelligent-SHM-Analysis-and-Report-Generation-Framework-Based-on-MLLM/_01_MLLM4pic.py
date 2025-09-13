# -*- coding: utf-8 -*-
import os
from config import FileConfig, OutputConfig, AccConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
from _00_common_model import ensure_model


OutConfig = OutputConfig()
outputconfig = OutConfig.tasks
file_conf = FileConfig()

def process_single_figure(fig_path, prompt_text,model=None, processor=None):
    model, processor = ensure_model(model, processor)
    """处理单个图像，返回模型输出字符串"""
    if not os.path.exists(fig_path):
        return ""
    # 构造消息
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": fig_path.replace("\\", "/")},
            {"type": "text", "text": prompt_text},
        ],
    }]
    # 生成输入文本以及处理视觉信息
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    # 模型生成
    generated_ids = model.generate(**inputs, max_new_tokens=4950)
    # 裁剪 prompt 部分
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] + '\n'

prompts = {
    "A": """你是一个顶级的结构健康监测数据分析引擎。你的任务是从给定的时程图中提取关键的**宏观模式和特征**，并将其输出为一个半结构化的JSON对象。你**不应**尝试识别或定位单个的异常数据点（如尖峰或离群值），但需要关注数据质量（如长时间中断）和整体模式的变化。你的输出必须精准、全面且简洁，尤其对于数据质量与特殊模式的描述应严格控制在15字内。
请分析下方提供的时程图，并严格按照指定的半结构化JSON格式输出分析结果。**不要添加任何解释性文字、开场白或结束语，只输出一个完整的JSON对象。**
**输出JSON格式定义与说明:**
{
  "源信息": {
    "通道名称": "字符串",
    "监测指标": "字符串",
    "单位": "字符串"
  },
  "时间范围": {
    "开始日期": "字符串",
    "结束日期": "字符串"
  },
  "统计特征": {
    "波动范围": [
      "浮点数",
      "浮点数"
    ],
  "模式分析": {
    "整体趋势": "字符串",
    "周期性": [
      {
        "类型": "字符串",
        "描述": "字符串"
      }
    ]
  },
  "数据质量与特殊模式": [
    {
      "特征类型": "字符串",
      "时间或范围": "字符串",
      "描述": "字符串"
    }
  ]
}
""",
    "B": """角色:
你是一个高效的结构健康监测数据分析引擎，专精于从运营模态分析(OMA)结果图中提取关键信息。
任务:
请分析提供的频率识别结果图。你的任务是识别出图中所有独立的模态簇，并为每个模态提取其中心频率和频率波动范围。请严格按照指定的精简中文JSON格式输出结果。不要添加任何多余的解释或文字，只输出一个JSON对象。
输出中文JSON格式定义与说明:
Json
{
  "源信息": {
    "分析任务": "字符串",
    "监测指标": "字符串",
    "单位": "字符串"
  },
  "时间范围": {
    "开始日期": "字符串",
    "结束日期": "字符串"
  },
  "已识别模态": [
    {
      "模态序号": "整数",
      "中心频率": "浮点数",
      "频率波动范围": [
        "浮点数",
        "浮点数"
      ]
    }
  ]
}
""",
    "C": """角色:
你是一个高效的结构健康监测数据分析引擎，专精于从运营模态分析(OMA)结果图中提取结构整体的阻尼比特征。
任务:
由于不同模态的阻尼比识别结果在数值上通常会重叠，难以仅从视觉上进行精确区分，你的任务是不再区分具体模态，而是将图中所有数据点视为一个整体，提取并总结其总体的统计特征，特别是整体波动范围。请严格按照指定的精简中文JSON格式输出结果。不要添加任何多余的解释或文字，只输出一个JSON对象。
输出中文JSON格式定义与说明:
code
Json
{
  "源信息": {
    "分析任务": "字符串",
    "监测指标": "字符串",
    "单位": "字符串"
  },
  "时间范围": {
    "开始日期": "字符串",
    "结束日期": "字符串"
  },
  "整体阻尼特征": {
    "平均值": "浮点数",
    "波动范围": [
      "浮点数",
      "浮点数"
    ],
    "离散度描述": "字符串"
  }
}
""",
    "D": """角色:
你是一个高效的结构健康监测数据分析引擎，专精于从振型图中提取和描述模态信息。
任务:
请分析下方提供的振型图。图中包含多个子图，每个子图代表一个独立的振动模态。你的任务是为每个模态提取其阶次、关联频率，并提供一个详细的文字描述来概括其振型特征。不要提取数值向量。 文字描述应清晰地概括振动的对称性、主要波峰/波谷的位置等关键视觉信息。请严格按照指定的精简中文JSON格式输出结果。不要添加任何多余的解释或文字，只输出一个JSON对象。
输出中文JSON格式定义与说明:
code
Json
{
  "源信息": {
    "分析任务": "字符串"
  },
  "识别出的振型": [
    {
      "模态阶次": "整数",
      "关联频率": "浮点数",
      "频率单位": "字符串",
      "振型描述": "字符串"
    }
  ]
}
""",
    "E": """角色:
你是一个高效的结构健康监测数据分析引擎，专精于解读多子图的车辆荷载分布直方图。
任务:
请分析下方提供的“各轴型车车重概率密度分布图”。图中包含多个子图，每个子图代表一种轴型车的车重分布。你的任务是遍历每一个子图，并为每种轴型车提取以下关键信息：
该轴型车的整体车重范围。
其车重最集中的主要区间（即峰值所在的区间）。
对分布形状的定性描述。
任何显著的、具有工程意义的解读（例如，识别双峰分布代表的空载/满载状态）。
请严格按照指定的中文JSON格式输出结果。不要添加任何多余的解释或文字，只输出一个JSON对象。
输出中文JSON格式定义与说明:
{
  "源信息": {
    "分析任务": "字符串"
  },
  "各轴型分布详情": [
    {
      "轴型": "字符串",
      "重量范围": [
        "浮点数",
        "浮点数"
      ],
      "主要重量区间": [
        "浮点数",
        "浮点数"
      ],
      "分布形态描述": "字符串",
      "工程解读": "字符串"
    }
  ]
}
""",
    "F": """角色:
你是一个高效的数据提取和分析引擎，专精于解读条形统计图并进行初步的数据分析。
任务:
请分析下方提供的条形统计图。你的任务是：
为图中的每一个条形，准确提取其代表的类别和对应的数值。
对整体数据进行分析和总结，包括计算总和，并对数据的分布特征进行解读。
请严格按照指定的中文JSON格式输出结果。不要添加任何多余的解释或文字，只输出一个JSON对象。
输出中文JSON格式定义与说明:
Json
{
  "源信息": {
    "分析任务": "字符串",
    "类别轴名称": "字符串",
    "数值轴名称": "字符串"
  },
  "统计数据": [
    {
      "类别": "字符串",
      "数值": "整数"
    }
  ],
  "分析摘要": {
    "总计数值": "整数",
    "分布解读": "字符串"
  }
}
""",
    "G": """"角色:
你是一个高效的结构健康监测数据分析引擎，专精于解读多子图的车辆荷载分布直方图。
任务:
请分析下方提供的“各车道车重概率密度分布图”。图中包含多个子图，每个子图代表一个车道的车重分布。你的任务是遍历每一个子图，并为每个车道提取以下关键信息：
该车道上车辆的整体车重范围。
其车重最集中的主要区间（即峰值所在的区间）。
对分布形状的定性描述（如单峰、多峰等）。
任何显著的、具有工程意义的解读（例如，推断该车道的主要通行车辆类型）。
请严格按照指定的中文JSON格式输出结果。不要添加任何多余的解释或文字，只输出一个JSON对象。
输出中文JSON格式定义与说明:
Json
{
  "源信息": {
    "分析任务": "字符串"
  },
  "各车道分布详情": [
    {
      "车道": "字符串",
      "重量范围": [
        "浮点数",
        "浮点数"
      ],
      "主要重量区间": [
        "浮点数",
        "浮点数"
      ],
      "分布形态描述": "字符串",
      "工程解读": "字符串"
    }
  ]
}
""",
    "H": """角色:
你是一个高效的数据分析引擎，专精于解读车速分布直方图。
任务:
请分析下方提供的车速概率密度分布图。你的任务是提取并总结该分布的关键统计特征，包括：
车辆行驶的主要速度范围。
对分布形状的定性描述（如单峰、多峰、偏态等）。
最主要的峰值速度区间。
请严格按照指定的中文JSON格式输出结果。不要添加任何多余的解释或文字，只输出一个JSON对象。
输出中文JSON格式定义与说明:
Json
{
  "源信息": {
    "分析任务": "字符串",
    "监测指标": "字符串",
    "单位": "字符串"
  },
  "分布特征": {
    "主要速度范围": [
      "浮点数",
      "浮点数"
    ],
    "分布形态描述": "字符串",
    "峰值速度区间": [
      "浮点数",
      "浮点数"
    ]
  }
}
""",
    "I": """角色:
你是一个高效的数据分析引擎，专精于解读车辆总重分布直方图，并从中分析交通组成。
任务:
请分析下方提供的车重概率密度分布图。你的任务是提取并总结该分布的关键特征，特别是要识别出代表不同车辆类型的主要峰值和次要峰值，并基于此对整体交通组成进行解读。请严格按照指定的中文JSON格式输出结果。不要添加任何多余的解释或文字，只输出一个JSON对象。
输出中文JSON格式定义与说明:
Json
{
  "源信息": {
    "分析任务": "字符串",
    "监测指标": "字符串",
    "单位": "字符串"
  },
  "分布特征": {
    "重量范围": [
      "浮点数",
      "浮点数"
    ],
    "分布形态描述": "字符串",
    "峰值分布": [
      {
        "峰值位置(t)": [
          "浮点数",
          "浮点数"
        ],
        "描述": "字符串"
      }
    ]
  },
  "交通组成解读": "字符串"
}
""",
    "J": """角色:
你是一个高效的数据分析引擎，专精于解读风向玫瑰图。
任务:
请分析下方提供的风向频率玫瑰图。你的任务是：
准确提取每个方向（N, NE, E, SE, S, SW, W, NW）对应的频率数值。
基于数值识别出该时段内的主导风向和次主导风向。
对整体风况特征进行简要的文字总结。
请严格按照指定的中文JSON格式输出结果。不要添加任何多余的解释或文字，只输出一个JSON对象。
输出中文JSON格式定义与说明:
Json
{
  "源信息": {
    "分析任务": "字符串",
    "监测设备": "字符串",
    "时间范围": "字符串"
  },
  "风向数据": [
    {
      "方向": "字符串",
      "频率": "浮点数"
    }
  ],
  "风况摘要": {
    "主导风向": "字符串",
    "次主导风向": [
      "字符串"
    ],
    "总体特征描述": "字符串"
  }
}
""",
    "K": """角色:
你是一个资深的结构健康监测工程师，专精于利用时间序列预测模型作为基准，来评估结构的长期行为稳定性。
任务:
请分析下方提供的结构状态评估图。在此图中，模型的置信区间（灰色区域）定义了结构的“预期正常波动范围”。你的任务是进行一次多层次的评估：
首先，从整体上量化真实值（绿色线）在置信区间内的覆盖率。
然后，识别并分类所有显著的偏离模式。你需要区分三种情况：
持续性偏离： 真实值一次性、长时间连续处于置信区间（灰色区域）之外。
频发性偏离： 真实值多次、有规律地、短暂地突破置信区间（灰色区域）。
偶然性偏离： 孤立的、无规律的单个瞬时突破置信区间（灰色区域）的毛刺。
最后，基于以上量化和定性分析，对结构在该评估期内的行为稳定性做出最终判断。
若图中未显示真实值（绿色线）则记录为数据缺失，默认结构正常
请严格按照指定的中文JSON格式输出结果。不要添加任何多余的解释或文字，只输出一个JSON对象。
输出中文JSON格式定义与说明:
code
Json
{
  "源信息": {
    "评估对象": "字符串",
    "监测指标": "字符串",
    "通道号": "字符串",
    "数值单位": "字符串"
  },
  "基线模型信息": {
    "模型类型": "字符串",
    "评估数据范围": "字符串"
  },
  "结构稳定性评估": {
    "置信区间覆盖率": "字符串",
    "主要偏离模式": [
      {
        "偏离类型": "字符串",
        "发生时段": "字符串",
        "偏离描述": "字符串"
      }
    ],
    "综合评估结论": "字符串"
  }
}
""",
    "L": """角色:
你是一个资深的数据分析专家，专精于解读和分析相关系数矩阵图，并能进行高度的归纳总结。
任务:
请详细分析下方提供的相关系数图,图中每一格中均有两个数据，上方字号更大的数据是前半时间的相关系数，下方字号更小的数据是后半时间的相关系数。你的任务是在内部完成所有必要的比较和分析，但最终只输出一个仅包含源信息和综合摘要的JSON对象。
不要包含任何逐项的细节列表。
在构建综合摘要时，请进行高层次的归纳，并明确回答以下核心问题：
整体趋势： 总体来看，这两类数据是正相关为主，还是负相关为主，或是混合存在？
整体稳定性： 绝大多数通道的相关性强度是否保持稳定？（一个“显著变化”被定义为相关系数从一个强度区间移动到了另一个不同的强度区间）
关键发现1： 哪个变量对表现出最强的相关性？
关键发现2： 哪个变量对的相关性变化最显著？
在回答时，请务必写明完整的变量名称和通道号，例如‘温度均值8通道与应变均值8通道’。
相关性强度区间定义:
[-1.00, -0.75]: "强负相关"
[-0.75, -0.50]: "较强负相关"
[-0.50, -0.25]: "较弱负相关"
[-0.25, 0.25]: "无相关"
[ 0.25, 0.50]: "较弱正相关"
[ 0.50, 0.75]: "较强正相关"
[ 0.75, 1.00]: "强正相关"
请严格按照指定的中文JSON格式输出结果。
输出中文JSON格式定义与说明:
Json
{
"源信息": {
"分析任务": "字符串",
"变量一": "字符串",
"变量二": "字符串"
},
"综合摘要": {
"整体相关性趋势": "字符串",
"整体稳定性评估": "字符串",
"最强相关变量对": "字符串",
"变化最显著变量对": "字符串"
}
}
""",
    "M": """角色:
你是一个高效的数据提取引擎，专精于从线性回归散点图中快速提取核心信息。
任务:
请分析下方提供的相关性图。你的任务是简明扼要地提取以下核心信息：
图表分析的两个变量。
比较前后两个时间段数据散点的离散程度，并必须从以下预设选项中选择一个作为答案：["前半段更大", "后半段更大", "两段时间类似"]。
判断总体相关性（正/负/无）。
提取前后两个时间段的拟合线斜率，并判断相关性强度的变化。
请严格按照指定的中文JSON格式输出结果，并确保所有文字描述都尽可能简短。不要添加任何多余的解释或文字，只输出一个JSON对象。
输出中文JSON格式定义与说明:
Json
{
  "分析对象": {
    "X轴": "字符串",
    "Y轴": "字符串"
  },
  "散点离散度比较": "字符串",
  "相关性分析": {
    "总体": "字符串",
    "前半段斜率": "浮点数",
    "后半段斜率": "浮点数",
    "变化评估": "字符串"
  }
}

""",
}

# 根据文件名选择 prompt key
def get_prompt_key(filename):
    name = filename.lower()
    if "rose" in name:
        return "J"
    elif ("rms" in name and "ass" not in name) or ("mean" in name and "ass" not in name) \
       or "figure_traf_count" in name or "figure_traf_weight_hour" in name:
        return "A"
    elif "fre" in name:
        return "B"
    elif "dp" in name:
        return "C"
    elif "phi" in name:
        return "D"
    elif "figure_traf_axle_prob" in name:
        return "E"
    elif "figure_traf_lane_count" in name:
        return "F"
    elif "figure_traf_lane_prob" in name:
        return "G"
    elif "figure_traf_speed" in name:
        return "H"
    elif "figure_traf_weight_probability" in name:
        return "I"
    elif "rose" in name:
        return "J"
    elif "ass" in name:
        return "K"
    elif re.match(r"figure_correlation\d+_\d+_\d+(\.png)?", name):
        return "M"
    elif re.match(r"figure_correlation\d+(\.png)?", name):
        return "L"
    return None
# 遍历每个任务及其对应的单任务
def run_MLLM4pic(model=None, processor=None):
    model, processor = ensure_model(model, processor)
    for task, channels in file_conf.filename_patterns.items():
        for single_task, task_info in outputconfig[task].items():
            out_put_result = ''
            # figure_path 可能为单个字符串或列表
            # 根据任务和任务类型选择对应的 prompt
            if single_task in ['rms', 'mean','regression','Correlation']:
                path_list = task_info['figure_path'] if isinstance(task_info['figure_path'], list) else [
                    task_info['figure_path']]
                i=0
                for fig_path in path_list:
                    i=i+1
                    prompt_key = get_prompt_key(os.path.basename(fig_path))
                    if prompt_key is None:
                        print(f"⚠️ 未匹配到提示词：{fig_path}")
                        continue  # 跳过未匹配的图片
                    prompt_to_use = prompts[prompt_key]
                    if i<10:
                    # 调用处理函数
                        out_put_result += process_single_figure(fig_path, prompt_to_use,model, processor)
                # 保存输出结果到文件
                save_path = os.path.join('Post_processing', 'LLM_result', task, f"{single_task}.txt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as file:
                    file.write(out_put_result)
                print(f'{task}的{single_task}分析完成')

            elif single_task != 'preprocess' and single_task != 'figure_identifier'and single_task != 'word_identifier':
                path_list = task_info['figure_path'] if isinstance(task_info['figure_path'], list) else [
                    task_info['figure_path']]
                for fig_path in path_list:
                    prompt_key = get_prompt_key(os.path.basename(fig_path))
                    if prompt_key is None:
                        print(f"⚠️ 未匹配到提示词：{fig_path}")
                        continue  # 跳过未匹配的图片
                    prompt_to_use = prompts[prompt_key]
                    # 调用处理函数
                    out_put_result += process_single_figure(fig_path, prompt_to_use,model, processor)
                # 保存输出结果到文件
                save_path = os.path.join('Post_processing', 'LLM_result', task, f"{single_task}.txt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as file:
                    file.write(out_put_result)
                print(f'{task}的{single_task}分析完成')

            else:
                continue


if __name__ == "__main__":
    run_MLLM4pic()