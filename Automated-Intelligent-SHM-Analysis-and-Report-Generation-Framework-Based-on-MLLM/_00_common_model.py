from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

_MODEL_PATH = r"D:\论文\04自动生成报告\Qwen-2.5-vl-3B\MLLM_run\model_file_from_Gemini_dataset"

def ensure_model(model=None, processor=None):
    """
    若传入的 model/processor 为 None，则在本进程内加载一份；
    若已传入（来自 a.py），则直接复用，避免重复加载。
    返回: (model, processor)
    """
    if model is not None and processor is not None:
        return model, processor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        _MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(_MODEL_PATH, use_fast=False)
    return model, processor
