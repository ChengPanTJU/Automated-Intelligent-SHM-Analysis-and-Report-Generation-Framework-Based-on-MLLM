# -*- coding: utf-8 -*-
import os
import re
from config import FileConfig, OutputConfig
from _00_common_model import ensure_model
# 定义各种 prompt
promptA = """
角色:
你是一位资深的结构健康监测数据分析师，你的任务是将多个独立的、关于同一监测类型（例如：湿度） 的分析结果，整合成一份简洁、连贯、有洞察力的分析文本。
任务:
你将收到一个文本块作为输入。该文本块包含多个、独立的JSON格式描述，它们都与指定的监测数据类型相关。你的核心任务是：
解析所有独立的JSON描述，并在内部进行整合、比较与归纳。
不要输出JSON或任何标题，而是直接开始撰写一份完整的分析文本。
文本必须遵循以下结构，并确保内容精炼，严格要求总长度控制在100字以内。不要使用不必要的特殊符号如*-
输出文本结构要求:
总体概述:
在本段开头，请首先明确指出这是关于哪种监测数据类型、在哪个时间范围内的分析。 然后，用一到两句话，高度概括该时段内此项监测数据的整体表现和最重要的特征。
分指标分析:
针对你在输入数据中找到的每一种监测指标（例如“均值”、“均方根”、“OMA频率”等），请分别为其创建一个独立的分析条目。
[此处填入找到的指标1名称]: （在此处总结该指标的共性，并重点对比不同通道间的差异。）
[此处填入找到的指标2名称]: （如果存在第二个指标，请在此处总结。）
（以此类推...）
如果输入中只包含一种指标，则此部分只应包含一个分析条目。
数据质量与行为模式:
（在此处汇总所有通道中发现的数据质量问题或特殊的行为模式。如果没有，请注明“数据质量良好，未见特殊行为模式”。）
关键发现与建议:
（在此处提炼出最重要的1-2个发现点，并给出明确的建议。）
"""
promptB = """
角色:
你是一位顶级的交通荷载分析专家。你的核心能力是综合、关联并解读任何你收到的、关于车辆荷载的数据片段，无论它们是否完整，并动态地生成一份逻辑清晰、有洞察力的分析文本。
任务:
你将收到一个包含数量和类型均不固定的、关于车辆荷载的JSON格式描述的文本块。你的任务是根据实际收到的数据维度，动态地构建一份综合分析文本。
全面理解输入： 解析你收到的所有JSON描述，识别出其中包含了哪些分析维度（例如：总体车流量、车道使用情况、交通构成、车速等）。
主题式归纳总结：
将相关的信息归纳到几个逻辑主题下。建议使用但不限于以下主题：交通流量特征、交通组成特征、车道使用特征、车辆速度特征。
只为你拥有数据的分析主题创建章节。 例如，如果输入中没有关于“车速”的信息，最终报告中就不应出现“车辆速度特征”这一章节。
在每个主题的段落中，必须综合关联所有相关的信息进行分析。例如，在“车道使用特征”中，要同时利用“车道数量”和“车道车重分布”的信息。
处理未知维度： 如果遇到无法归入上述主题的新数据类型，请为其创建一个新的、名称恰当的章节进行总结。
最终输出： 不要输出JSON，而是生成一份完整的、章节结构由输入数据决定的分析文本，并确保内容精炼，排除不必要的符号（如*），严格要求总长度控制在100字以内。不要使用不必要的特殊符号如*-
输出文本结构要求:
文本应由一个或多个带标题的段落组成，最后附上一个综合结论。结构应类似于：
[分析主题一，例如：交通流量特征]
（对此主题的综合分析...）
[分析主题二，例如：车道使用特征]
（对此主题的综合分析...）
（...根据实际数据动态增删主题...）
数据质量与完整性
（扫描所有输入，汇总发现的数据质量问题。）
综合结论与关键发现
（提炼出基于所有可用信息的最核心结论。）
"""
promptC = """
角色:
你是一位资深的结构健康监测专家。现有多份关于同一监测类型 在不同测点（通道）的独立评估文本。你的任务是将这些独立的评估文本综合成一份高度概括的总体状态文本。
任务:
你将收到一个文本块作为输入，其中包含多份关于指定监测数据类型在不同通道的独立评估文本。你的核心任务是进行一次彻底的综合分析，而不是简单地罗列各通道的结论。具体要求如下：
全面理解输入： 解析所有独立的评估文本，理解每个通道的稳定性状况和偏离模式。
寻找共性与差异：
首先，忽略频发性偏离，只关注持续性偏离，不是长时间的偏离视为模型训练训练问题
共性问题： 是否有多个通道在相近的时间段内都出现了偏离？这可能暗示着一个全局性的事件。
差异表现： 不同通道的稳定性表现是否一致？哪个通道表现最差（最不稳定）？
提炼核心结论： 基于以上分析，对该监测数据所反映的结构整体状态给出一个明确的、定性的总体判断（例如：稳定、基本稳定、不稳定、局部不稳定等）。
最终输出： 不要输出JSON，而是生成一份完整的、结构清晰的评估文本，确保内容精炼。请直接开始撰写报告内容，不要添加任何角色扮演式的开场白、引言或自我介绍（例如，不要说“作为专家，我分析了...”或“以下是我的报告...”），严格要求输出在100字内,不要使用不必要的特殊符号如*-
输出文本结构要求:
1. 总体评估结论:
（在此处，首先给出一个关于该监测数据所反映的结构总体稳定性的明确判断）
2. 主要发现与分析:
（关键综合分析） （在此处详细阐述你的判断依据。你需要整合所有通道的信息，描述不稳定现象是普遍性的还是局部性的，并指出最不稳定的通道。特别要强调不同通道在同一时间段内出现的共性偏离，）
3. 各通道状态摘要:
[通道1名称]: （用一句话概括该通道的稳定性评估结论）
[通道2名称]: （用一句话概括...）
[通道3名称]: （用一句话概括...）
（根据实际数据动态生成）
4. 建议:
（基于以上分析，提出需要采取的下一步行动。）
"""
promptD = """
你是一位顶尖的结构健康监测数据科学家和诊断专家。你的任务不仅是总结数据，更是要从相关性的变化中洞察潜在的物理原因，并提出专业的建议。
任务:
你将收到一个文本块作为输入，其中包含多份独立的、关于不同变量对之间相关性的JSON格式摘要。你的核心任务是进行一次彻底的宏观综合分析。具体要求如下：
首先，从输入信息中识别出正在进行比较的两个主要变量类型。
基于你识别出的变量类型，动态地构建一份综合分析报告。
在“关键发现”部分，对于每一个你识别出的重要现象（如最强相关或最显著变化），你必须进行以下分析：
行动建议： 给出具体、可行的下一步建议。例如，建议“核查传感器状态”、“进行现场勘查”或“引入更多变量进行深入分析”。
最终输出： 不要输出JSON，而是生成一份完整的、包含诊断性建议的分析文本。请直接开始撰写报告内容，不要添加任何开场白或自我介绍。严格要求输出文本限定在100字内，不要使用不必要的特殊符号如*-。
输出结构要求:
1. 核心结论:
（在此处，用一句话给出关于[变量类型一]与[变量类型二]之间关系的最核心、最顶层的结论。）
2. 关系模式概述:
（总结[变量类型一]与[变量类型二]之间的总体相关性模式。）
3. 关系稳定性评估:
（总结这种相关性在观测期内的整体稳定性。）
4. 关键发现与建议:
发现一：[描述最强相关的变量对]
建议： （基于此强相关性，提出建议，如“可利用此关系建立温度修正模型”。）
发现二：[描述变化最显著的变量对]
"""

OutConfig = OutputConfig()
outputconfig = OutConfig.tasks
file_conf = FileConfig()

def clean_text(text):
    """去掉 Markdown 代码块标记和 JSON 结构符号"""
    if not text:
        return ""

    # 去掉 ```json 或 ``` 开头和 ``` 结尾
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)

    # 去掉 JSON 结构符号
    text = re.sub(r"[{}\[\],]", "", text)

    # 去掉多余空格
    text = re.sub(r"\s+", " ", text).strip()

    return text

def run_MLLM4single_type_sum(model=None, processor=None):
    model, processor = ensure_model(model, processor)
    for task, channels in file_conf.filename_patterns.items():
        save_path = os.path.join('Post_processing', 'LLM_result', task)
        if task=='assessment2':
            reg_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if
                         f.endswith('.txt') and ('regression' in f)]
            if len(reg_files) > 0:
                for i in range(len(reg_files)):
                    with open(reg_files[i], 'r', encoding='utf-8') as f:
                        merged_content = f.read()
                    merged_content = clean_text(merged_content)
                    messages = [
                        {"role": "user", "content": [{"type": "text", "text": [promptC, merged_content]}, ], }
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
                    save_name=reg_files[i].split('\\')[-1][:-4]
                    save_path = os.path.join('Post_processing', 'LLM_result', task, f"{save_name}_sum.txt")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "w", encoding="utf-8") as file:
                        file.write(output_text[0])

        elif task=='correlation':
            cor_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if
                              f.endswith('.txt') and ('Correlation' in f )]
            if len(cor_files) > 0:
                for i in range(len(cor_files)):
                    with open(cor_files[i], 'r', encoding='utf-8') as f:
                        merged_content = f.read()
                    merged_content = clean_text(merged_content)
                    messages = [
                        {"role": "user", "content": [{"type": "text", "text": [promptD,merged_content]}, ], }
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
                    save_name=cor_files[i].split('\\')[-1][:-4]
                    save_path = os.path.join('Post_processing', 'LLM_result', task, f"{save_name}_sum.txt")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "w", encoding="utf-8") as file:
                        file.write(output_text[0])
        elif task=='traffic':
            traf_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if
                              f.endswith('.txt') and ('sum' not in f )]
            if len(traf_files) > 0:
                merged_content = ''
                for file in traf_files:
                    with open(file, 'r', encoding='utf-8') as f:
                        merged_content += f.read() + '\n'
                merged_content= clean_text(merged_content)
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": [promptB,merged_content]}, ], }
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
                generated_ids = model.generate(**inputs, max_new_tokens=4950)
                # 裁剪 prompt 部分
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                save_path = os.path.join('Post_processing', 'LLM_result', task, "traf_sum.txt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as file:
                    file.write(output_text[0])
        else:
            mean_rms_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.txt') and ('sum' not in f )]
            if len(mean_rms_files)>0:
                merged_content = ''
                for file in mean_rms_files:
                    with open(file, 'r', encoding='utf-8') as f:
                        merged_content += f.read() + '\n'
                merged_content= clean_text(merged_content)
                messages = [
                    {"role": "user","content": [{"type": "text", "text": [promptA,merged_content]},],}
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
                save_path = os.path.join('Post_processing', 'LLM_result', task, "type_sum.txt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as file:
                    file.write(output_text[0])

if __name__ == "__main__":
    run_MLLM4single_type_sum()