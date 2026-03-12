"""
模型加载器
=========
核心功能: 统一管理各类预训练模型的加载, 支持全精度/半精度/量化加载

技术架构:
1. 根据配置自动选择加载策略 (全精度、FP16、BF16、4bit量化、8bit量化)
2. 集成BitsAndBytes量化配置 (QLoRA的核心依赖)
3. 支持Flash Attention 2加速 (需要兼容的GPU和模型)
4. 自动处理模型特殊配置 (trust_remote_code, device_map等)


- 模型加载时的dtype选择直接影响显存占用和计算精度
- BF16 vs FP16: BF16动态范围更大(与FP32相同), 训练更稳定; FP16精度更高但易溢出
- 4bit量化原理: 将FP32/FP16权重映射到4bit表示, 推理时反量化回高精度计算
- NF4 (NormalFloat4): 假设权重服从正态分布, 量化点等概率分布, 比FP4更适合神经网络
"""

import logging
from typing import Dict, Any, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

# 模型类型到AutoModel类的映射
MODEL_TYPE_MAP = {
    "causal_lm": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM,
}

# torch dtype字符串到实际类型的映射
DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class ModelLoader:
    """
    模型加载器 - 负责预训练模型和Tokenizer的加载

    支持的加载模式:
    1. 全精度加载 (FP32): 最高精度, 显存需求最大
    2. 半精度加载 (FP16/BF16): 显存减半, 精度损失可忽略
    3. 4bit量化加载 (QLoRA): 显存大幅降低, 适合消费级GPU
    4. 8bit量化加载: 介于半精度和4bit之间的平衡

    使用示例:
        loader = ModelLoader(config)
        model, tokenizer = loader.load()
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.tokenizer_config = config.get("tokenizer", {})
        self.finetuning_config = config.get("finetuning", {})

    def _get_torch_dtype(self) -> torch.dtype:
        """
        解析配置中的数据类型

        选择建议:
        - fp32: 调试使用, 生产不推荐 (显存翻倍)
        - bf16: Ampere (A100/3090) 及以上GPU推荐 (训练稳定)
        - fp16: 较老GPU (V100等) 使用 (注意可能需要loss scaling)
        """
        dtype_str = self.model_config.get("torch_dtype", "bf16")
        return DTYPE_MAP.get(dtype_str, torch.bfloat16)

    def _build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        构建量化配置 (QLoRA核心)

        QLoRA量化技术详解:
        1. NF4 (NormalFloat4): 假设权重正态分布, 4bit量化点按等概率分布
           - 比均匀量化(FP4)更适合神经网络权重
        2. 双重量化 (Double Quantization): 对量化常数本身再做一次量化
           - 每个参数额外节省约0.37bit
        3. 分页优化器: 当GPU显存不足时, 自动将优化器状态卸载到CPU
           - 避免OOM, 但增加CPU-GPU数据传输开销

        Returns:
            BitsAndBytesConfig 或 None (非量化模式返回None)
        """
        method = self.finetuning_config.get("method", "full")

        if method != "qlora":
            return None

        qlora_config = self.finetuning_config.get("qlora", {})
        bits = qlora_config.get("bits", 4)
        quant_type = qlora_config.get("quant_type", "nf4")
        double_quant = qlora_config.get("double_quant", True)
        compute_dtype_str = qlora_config.get("compute_dtype", "bf16")
        compute_dtype = DTYPE_MAP.get(compute_dtype_str, torch.bfloat16)

        logger.info(
            f"Building quantization config: {bits}bit, type={quant_type}, "
            f"double_quant={double_quant}, compute_dtype={compute_dtype}"
        )

        if bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(
                f"Unsupported quantization bits: {bits}. Use 4 or 8.")

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        加载Tokenizer

        注意事项:
        - 某些模型需要 trust_remote_code=True (如Qwen, ChatGLM)
        - 需要确保pad_token存在 (部分模型默认没有, 如LLaMA)
        - padding_side影响生成行为 (训练用right, 生成用left)
        """
        model_name = self.model_config.get("model_name_or_path")
        tokenizer_name = (
            self.tokenizer_config.get("tokenizer_name_or_path") or model_name
        )
        trust_remote_code = self.model_config.get("trust_remote_code", True)

        logger.info(f"Loading tokenizer from: {tokenizer_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            padding_side="right",  # 训练时使用右填充
        )

        # 确保有pad_token (LLaMA等模型默认没有)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Set pad_token to eos_token: '{tokenizer.eos_token}'")

        return tokenizer

    def load_model(
        self,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> PreTrainedModel:
        """
        加载预训练模型

        加载流程:
        1. 确定模型类型 (CausalLM / Seq2SeqLM)
        2. 配置数据类型和设备映射
        3. 如果是QLoRA, 应用量化配置
        4. 可选启用Flash Attention 2

        Args:
            quantization_config: 量化配置 (QLoRA时传入)

        Returns:
            加载好的PreTrainedModel
        """
        model_name = self.model_config.get("model_name_or_path")
        model_type = self.model_config.get("model_type", "causal_lm")
        trust_remote_code = self.model_config.get("trust_remote_code", True)
        use_flash_attention = self.model_config.get(
            "use_flash_attention", False)

        model_class = MODEL_TYPE_MAP.get(model_type)
        if model_class is None:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported: {list(MODEL_TYPE_MAP.keys())}"
            )

        torch_dtype = self._get_torch_dtype()

        # 构建加载参数
        load_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "device_map": "auto",  # 自动设备映射 (多GPU自动分配)
        }

        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config

        if use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")

        logger.info(
            f"Loading model: {model_name} (type={model_type}, dtype={torch_dtype})"
        )

        model = model_class.from_pretrained(**load_kwargs)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model loaded. Total parameters: {total_params:,} "
            f"({total_params / 1e9:.2f}B)"
        )

        return model

    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        一站式加载模型和Tokenizer

        Returns:
            (model, tokenizer) 元组
        """
        tokenizer = self.load_tokenizer()
        quantization_config = self._build_quantization_config()
        model = self.load_model(quantization_config=quantization_config)

        # 调整模型embedding大小以匹配tokenizer (如果添加了特殊token)
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
            logger.info(
                f"Resized token embeddings to {len(tokenizer)}"
            )

        return model, tokenizer
