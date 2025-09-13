# -*- coding: utf-8 -*-
import os
import re

from config import FileConfig, OutputConfig, AccConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from _00_common_model import ensure_model


# 定义各种 prompt
PROMPT_TEXT = """角色:
你是一位顶级的结构健康首席诊断专家，负责撰写最终的结构健康年度/季度诊断报告。你的办公桌上汇总了关于一座桥梁在某一监测时段内的所有分析材料，包括各监测指标的总结、多维度相关性分析，以及基于模型的结构状态评估。你的任务是将所有这些信息融会贯通，形成一份全局性的、有理有据、结论明确的最终诊断报告。
任务:
你将收到一个包含多份独立分析报告的文本块。你的核心任务是进行一次终极的综合诊断。具体要求如下：
全面理解并整合所有输入信息，识别出结构在各个方面的表现。
（核心任务）寻找跨报告的证据链，进行因果推断：
关联“评估异常”与“相关性变化”： 结构状态评估中发现的异常偏离（如沉降不稳定），是否与相关性分析中发现的关系变化（如荷载-沉降关系减弱）在时间上吻合？
关联“指标趋势”与“评估异常”： 监测指标总结中发现的长期趋势（如索力持续上升），是否能解释状态评估中出现的偏离？
识别主导影响因素： 综合所有信息，判断当前结构状态的主要驱动因素是环境效应（如温度）、荷载效应，还是结构自身性能退化？
（核心任务）进行风险评级与定位：
基于所有证据，对结构的总体健康状况给出一个明确的定性评级（例如：健康、基本健康、需关注、预警、危险）。
明确指出风险最高的物理区域（如1号测点附近）和最关键的风险指标（如沉降、索力）。
最终输出： 不要输出JSON，而是生成一份结构清晰、语言精炼的最终诊断报告。请直接开始撰写报告，不要添加任何开场白或自我介绍。输出严格限定在800字以内
输出诊断报告结构要求:
1. 结构总体健康诊断:
监测时段： [自动识别并填写]
总体健康评级： [给出明确的评级]
核心诊断结论： （用一到两句话，高度概括结构当前最核心的状态和问题。）
2. 关键发现与综合诊断分析:
（在此处，详细阐述你的诊断依据，重点展示你构建的证据链。）
主导影响因素分析： （分析并指出是温度、荷载还是结构自身原因在主导当前结构的行为模式。）
关键异常事件与时间节点： （整合所有报告，识别并描述在多个维度上共现的异常事件及其关键时间点，如“11月20日”。）
风险区域与指标定位： （明确指出问题最集中的结构区域和最值得关注的监测指标。）
3. 数据质量总体评估:
（综合所有报告中关于数据质量的描述，给出一个关于整个监测系统可靠性的总体评价。）
4. 综合建议与行动优先级:
（基于全局诊断，提出有明确优先级的、具体的行动建议。）
最高优先级： （通常是针对最紧急风险的建议，如“立即对XX区域进行现场勘查”。）
次要优先级： （针对长期趋势或数据质量问题的建议，如“完善温度修正模型”、“修复数据采集系统”。）"""
OutConfig = OutputConfig()
outputconfig = OutConfig.tasks
file_conf = FileConfig()
def clean_gpt_output(text):
    """
    清理 GPT 输出文本中的不必要符号，如 * 和 -
    """
    # 这里将 * 和 - 单独出现或连续出现都替换为空
    cleaned_text = re.sub(r'[\*\-]+', '', text)
    # 去除多余空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def run_MLLM4all_sum(model=None, processor=None):
    model, processor = ensure_model(model, processor)
    save_path = os.path.join('Post_processing', 'LLM_result')
    reg_sum_files = [os.path.join(root, f) for root, _, files in os.walk(save_path) for f in files if f.endswith('.txt') and ('reg_sum' in f )]
    reg_content = ''
    cor_content = ''
    type_content = ''
    if len(reg_sum_files) > 0:
        for file in reg_sum_files:
            with open(file, 'r', encoding='utf-8') as f:
                reg_content += f.read() + '\n'
    cor_sum_files = [os.path.join(root, f) for root, _, files in os.walk(save_path) for f in files if f.endswith('.txt') and ('cor_sum' in f )]
    if len(cor_sum_files) > 0:
        for file in cor_sum_files:
            with open(file, 'r', encoding='utf-8') as f:
                cor_content += f.read() + '\n'
    type_sum_files = [os.path.join(root, f) for root, _, files in os.walk(save_path) for f in files if f.endswith('.txt') and ('type_sum' in f )]
    if len(type_sum_files) > 0:
        for file in type_sum_files:
            with open(file, 'r', encoding='utf-8') as f:
                type_content += f.read() + '\n'
    merged_content=reg_content+"\n\n###\n\n"+cor_content+"\n\n###\n\n"+type_content
    merged_content=clean_gpt_output(merged_content)
    messages = [
        {"role": "user", "content": [{"type": "text", "text":  PROMPT_TEXT + "\n\n" +merged_content}, ], }
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
    save_path = os.path.join('Post_processing', 'LLM_result', "all_sum.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(output_text[0])

if __name__ == "__main__":
    run_MLLM4all_sum()
