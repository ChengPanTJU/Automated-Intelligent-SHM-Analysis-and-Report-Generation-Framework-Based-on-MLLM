# -*- coding: utf-8 -*-
import os
from config import FileConfig, OutputConfig, AccConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from _00_common_model import ensure_model


# 定义各种 prompt
PROMPT_EVAL = """
角色:
你是一位顶级的结构健康监测首席诊断专家。你的办公桌上放着多份由不同专业组（如沉降组、振动组）提交的、关于结构在同一时段内不同方面的评估文本。你的最终任务是融会贯通所有报告，形成一份全局性的、面向最高决策层的结构总体健康状况诊断文本。
任务:
你将收到一个文本块作为输入，其中包含多份关于结构不同监测指标（如沉降、加速度频率等）的独立评估文本。你的核心任务是进行一次彻底的跨领域综合分析，寻找现象之间的关联。具体要求如下：
全面理解输入： 解析所有独立的评估文本，理解结构在每个监测维度上的稳定性和异常表现。
关联性分析与共性问题识别：
寻找时间上的共性： 识别出在多个不同监测报告中被反复提及的关键时间节点或时段,但应该忽略单个测点偶然的异常，视为预测模型的训练问题而非结构异常问题，只关注多个测点（至少三个）同时出现的异常。例如，如果沉降和加速度都在“11月20日”后出现持续性异常，这必须被视为一个最高优先级的全局性事件。
构建全局状态画像： 基于以上关联分析，对结构的总体健康状况给出一个明确、综合的判断。
最终输出： 不要输出JSON，而是生成一份完整的、简洁的、结构清晰的总体诊断文本。请直接开始撰写报告内容，不要添加任何开场白或自我介绍。输出限定在150字内
输出诊断报告结构要求:
结构总体健康状况诊断:
（在此处，用一到两句话，给出对结构在本监测时段内总体健康状况的最核心、最直接的判断。）
关键发现:
（在此处，详细阐述你的诊断依据。你需要整合并关联所有报告的信息，重点突出跨监测指标的共性问题。忽略单个测点偶然的异常，视为预测模型的训练问题而非结构异常问题，只关注多个测点（至少三个）同时出现的异常））
各监测维度状态概述:
沉降: （用一句话概括沉降的总体状态，如：“表现为局部不稳定，1号和3号通道后期异常。”）
加速度频率: （用一句话概括...如：“整体不稳定，多个关键时间点出现模式改变。”）
（根据实际数据动态生成）"""
PROMPT_CORR = """
角色:
你是一位顶级的结构健康监测首席诊断工程师。你的任务是将多份独立的、关于结构在同一时段内不同影响因素（如温度、荷载）与不同结构响应（如应变、位移、频率、沉降）之间相关性关系的研究文本，融会贯通，形成一份全局性的、高度概括的结构行为模式诊断文本。
任务:
你将收到一个文本块作为输入，其中包含多份关于不同变量对之间相关性的独立分析文本。你的核心任务是进行一次彻底的跨领域综合分析，寻找现象之间的关联与主导因素。具体要求如下：
全面理解输入： 解析所有独立的分析文本，理解结构对不同影响因素的响应模式、强度和稳定性。
按影响因素进行归纳总结：
提炼温度效应： 综合所有与“温度”相关的文本，总结温度对结构不同响应（位移、应变、频率等）的总体影响模式。
提炼荷载效应： 综合所有与“交通荷载”相关的文本，总结荷载对结构响应的总体影响模式。（如果没有荷载效应描述则忽略）
识别关键行为模式与问题：
识别主导因素： 判断哪个影响因素（温度或荷载）对结构的整体响应影响更大、相关性更强？
定位不稳定关系： 找出所有文本中被标记为“不稳定”或“发生显著变化”的关系，这是结构行为变化的关键信号。
最终输出： 不要输出JSON，而是生成一份完整的、简洁的、结构清晰的诊断文本。请直接开始撰写文本内容，不要添加任何开场白或自我介绍。要求输出在150字以内
输出诊断报告结构要求:
结构行为模式总体诊断:
（在此处，用一到两句话，给出关于结构响应主导因素的最核心结论。）
主要影响因素分析:
温度效应综合分析: （综合所有温度相关报告，用一两句话描述温度对位移、应变、频率的总体影响。）
交通荷载效应综合分析: （综合所有荷载相关报告，用一两句话描述荷载对沉降等响应的影响。）
关键不稳定关系汇总:
（在此处，明确列出所有被识别为“不稳定”或“发生显著变化”的变量关系，这是最重要的风险信号。例如：“本次分析识别出以下关键不稳定关系：1. 温度-应变关系（多个通道增强或反转）；2. 交通荷载-沉降关系（普遍减弱）；3. 温度-频率关系（局部减弱）。”）
综合建议与行动计划:
（基于全局诊断，提出一两条建议。）"""

OutConfig = OutputConfig()
outputconfig = OutConfig.tasks
file_conf = FileConfig()

def run_MLLM4corandreg_sum(model=None, processor=None):
    model, processor = ensure_model(model, processor)
    for task, channels in file_conf.filename_patterns.items():
        save_path = os.path.join('PostProcessing', 'LLM_result', task)
        if task=='correlation':
            cor_sum_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if
                              f.endswith('.txt') and ('Correlation' in f and 'sum' in f )]
            if len(cor_sum_files) > 0:
                merged_content = ''
                for file in cor_sum_files:
                    with open(file, 'r', encoding='utf-8') as f:
                        merged_content += f.read() + '\n'
                messages = [
                    {"role": "user", "content": [{"type": "text", "text":  PROMPT_CORR + "\n\n" + merged_content}, ], }
                ]
                # 生成输入文本以及处理视觉信息
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)
                # 模型生成
                generated_ids = model.generate(**inputs, max_new_tokens=4096)
                # 裁剪 prompt 部分
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                save_path = os.path.join('PostProcessing', 'LLM_result', task, "cor_sum.txt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as file:
                    file.write(output_text[0])
        elif task=='assessment2':
            cor_sum_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if
                              f.endswith('.txt') and ('regression' in f and 'sum' in f )]
            if len(cor_sum_files) > 0:
                merged_content = ''
                for file in cor_sum_files:
                    with open(file, 'r', encoding='utf-8') as f:
                        merged_content += f.read() + '\n'
                messages = [
                    {"role": "user", "content": [{"type": "text", "text":  PROMPT_EVAL + "\n\n" + merged_content}, ], }
                ]
                # 生成输入文本以及处理视觉信息
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)
                # 模型生成
                generated_ids = model.generate(**inputs, max_new_tokens=4096)
                # 裁剪 prompt 部分
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                save_path = os.path.join('PostProcessing', 'LLM_result', task, "reg_sum.txt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as file:
                    file.write(output_text[0])
if __name__ == "__main__":
    run_MLLM4corandreg_sum()