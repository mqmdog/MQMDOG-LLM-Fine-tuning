"""
通用工具函数
============
提供训练过程中常用的辅助功能
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    设置全局随机种子, 确保实验可复现

    需要同时设置:
    1. Python random模块
    2. NumPy
    3. PyTorch CPU和GPU
    4. CUDA相关的确定性设置

    面试要点:
    - 完全可复现还需要: 相同硬件、相同CUDA版本、deterministic算法
    - torch.backends.cudnn.deterministic=True 会降低部分操作性能
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 使用确定性算法 (可能略微降低性能)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to: {seed}")


def print_trainable_parameters(model: PreTrainedModel) -> dict:
    """
    打印模型的可训练参数统计

    对于PEFT模型, 可以清楚看到:
    - 总参数量 vs 可训练参数量
    - 参数效率比 (可训练/总参数)

    示例输出:
    Total params: 6,738,415,616 | Trainable: 4,194,304 | Ratio: 0.0623%

    Returns:
        包含参数统计的字典
    """
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    ratio = trainable_params / all_params * 100 if all_params > 0 else 0

    stats = {
        "total_params": all_params,
        "trainable_params": trainable_params,
        "trainable_ratio": ratio,
    }

    logger.info(
        f"Total params: {all_params:,} | "
        f"Trainable: {trainable_params:,} | "
        f"Ratio: {ratio:.4f}%"
    )

    return stats


def get_device_info() -> dict:
    """
    获取当前设备信息

    包括:
    - GPU数量和型号
    - 显存大小
    - CUDA版本
    - PyTorch版本
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = []

        for i in range(torch.cuda.device_count()):
            gpu_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1e9, 2
                ),
            }
            info["gpus"].append(gpu_info)

        logger.info(
            f"CUDA {info['cuda_version']} | "
            f"{info['gpu_count']} GPU(s): "
            f"{', '.join(g['name'] for g in info['gpus'])}"
        )
    else:
        logger.info("No CUDA GPU available, using CPU")

    # 检查bfloat16支持
    if torch.cuda.is_available():
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    else:
        info["bf16_supported"] = False

    return info


def estimate_memory_usage(
    model_params_billion: float,
    method: str = "lora",
    dtype: str = "bf16",
    batch_size: int = 4,
    seq_length: int = 512,
) -> dict:
    """
    估算训练显存需求

    显存组成:
    1. 模型权重: params × bytes_per_param
    2. 梯度: 与模型权重相同 (全参数微调) 或仅PEFT参数部分
    3. 优化器状态: Adam需要2倍参数量 (一阶和二阶动量)
    4. 激活值: 与batch_size和seq_length成正比
    5. KV Cache (推理): 2 × num_layers × hidden_dim × seq_length × batch_size

    经验公式 (全参数微调, Adam, FP16):
    总显存 ≈ 模型参数量 × 18 bytes (2权重 + 2梯度 + 8优化器 + 6激活)

    面试要点:
    - LoRA显存 ≈ 模型权重 + LoRA参数×(2梯度+8优化器) + 激活值
    - QLoRA显存 ≈ 模型权重/2(4bit) + LoRA参数×(2梯度+8优化器) + 激活值
    - 梯度检查点可减少约30%激活值显存, 但增加约20%计算时间

    Returns:
        显存估算详情字典
    """
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    param_bytes = bytes_per_param.get(dtype, 2)
    total_params = model_params_billion * 1e9

    if method == "full":
        # 全参数微调: 权重 + 梯度 + 优化器状态(2x)
        weight_mem = total_params * param_bytes
        gradient_mem = total_params * param_bytes
        optimizer_mem = total_params * 8  # Adam: 2 × FP32 = 8 bytes/param
        trainable_params = total_params
    elif method in ("lora", "adalora"):
        # LoRA: 全量权重(冻结) + LoRA参数的(梯度+优化器)
        lora_ratio = 0.01  # LoRA参数约占1%
        weight_mem = total_params * param_bytes
        gradient_mem = total_params * lora_ratio * param_bytes
        optimizer_mem = total_params * lora_ratio * 8
        trainable_params = total_params * lora_ratio
    elif method == "qlora":
        # QLoRA: 4bit权重 + LoRA参数的(梯度+优化器)
        lora_ratio = 0.01
        weight_mem = total_params * 0.5  # 4bit
        gradient_mem = total_params * lora_ratio * 2  # BF16
        optimizer_mem = total_params * lora_ratio * 8
        trainable_params = total_params * lora_ratio
    else:
        weight_mem = total_params * param_bytes
        gradient_mem = 0
        optimizer_mem = 0
        trainable_params = 0

    # 激活值估算 (简化)
    activation_mem = batch_size * seq_length * 4096 * 4  # 粗略估算

    total_mem = weight_mem + gradient_mem + optimizer_mem + activation_mem

    result = {
        "method": method,
        "model_params_B": model_params_billion,
        "trainable_params": int(trainable_params),
        "weight_mem_gb": round(weight_mem / 1e9, 2),
        "gradient_mem_gb": round(gradient_mem / 1e9, 2),
        "optimizer_mem_gb": round(optimizer_mem / 1e9, 2),
        "activation_mem_gb": round(activation_mem / 1e9, 2),
        "total_estimated_gb": round(total_mem / 1e9, 2),
    }

    logger.info(
        f"Memory estimate ({method}, {model_params_billion}B params): "
        f"~{result['total_estimated_gb']} GB"
    )

    return result
