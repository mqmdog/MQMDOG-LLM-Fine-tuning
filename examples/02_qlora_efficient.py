"""
示例2: QLoRA微调 - 资源高效的微调方案
=====================================
演示如何在消费级GPU上使用QLoRA微调大语言模型

应用场景: 资源受限环境下的模型微调
- 4bit量化大幅降低显存需求 (7B模型: ~6GB vs ~28GB)
- NF4量化 + 双重量化 + 分页优化器 = 消费级GPU也能微调

运行方式:
    python examples/02_qlora_efficient.py --config configs/qlora_config.yaml

技术要点:
1. NF4 (NormalFloat4): 利用权重正态分布特性的信息论最优量化
2. 双重量化: 对量化常数本身再量化, 额外节省~0.37bit/param
3. 分页优化器: GPU OOM时自动将优化器状态卸载到CPU内存
"""

from src.training.trainer import FineTuneTrainer
from src.data.data_loader import DataManager
from src.models.peft_config import PeftConfigFactory
from src.models.model_loader import ModelLoader
from src.utils.common import set_seed, print_trainable_parameters, get_device_info, estimate_memory_usage
from src.utils.logger import setup_logger
from src.utils.config_parser import ConfigParser
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="QLoRA Efficient Fine-tuning")
    parser.add_argument("--config", type=str,
                        default="configs/qlora_config.yaml")
    parser.add_argument("--base_config", type=str,
                        default="configs/base_config.yaml")
    args = parser.parse_args()

    config = ConfigParser.load(args.config, args.base_config)

    log_config = config.get("logging", {})
    logger = setup_logger(
        name="llm_finetune",
        log_dir=log_config.get("log_dir", "outputs/logs"),
    )

    logger.info("=" * 70)
    logger.info("  QLoRA Efficient Fine-tuning")
    logger.info("=" * 70)

    # 环境准备
    set_seed(config.get("training", {}).get("seed", 42))
    device_info = get_device_info()

    # 显存预估
    model_name = config.get("model", {}).get("model_name_or_path", "")
    # 简单估算模型大小 (实际项目可从config读取)
    mem_estimate = estimate_memory_usage(
        model_params_billion=0.5,  # Qwen2.5-0.5B
        method="qlora",
        dtype="bf16",
        batch_size=config.get("training", {}).get(
            "per_device_train_batch_size", 4),
        seq_length=config.get("tokenizer", {}).get("max_length", 1024),
    )
    logger.info(f"Estimated memory: {mem_estimate}")

    # 加载模型 (QLoRA: 4bit量化加载)
    logger.info("Loading 4-bit quantized model...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load()

    # 应用LoRA (QLoRA = 4bit模型 + LoRA)
    logger.info("Applying LoRA to quantized model...")
    peft_factory = PeftConfigFactory(config)
    model = peft_factory.apply_peft(model)
    print_trainable_parameters(model)

    # 准备数据
    logger.info("Preparing datasets...")
    template_name = config.get("data", {}).get("template", "alpaca")
    data_manager = DataManager(tokenizer, config, template_name=template_name)
    train_dataset, eval_dataset = data_manager.prepare_datasets()

    # 训练
    logger.info("Starting QLoRA training...")
    trainer = FineTuneTrainer(model, tokenizer, config)
    metrics = trainer.train(train_dataset, eval_dataset)

    # 保存
    output_dir = config.get("training", {}).get(
        "output_dir", "outputs/models/qlora-chat")
    trainer.save_model(output_dir)

    logger.info("=" * 70)
    logger.info("  QLoRA Fine-tuning Complete!")
    logger.info(f"  Model saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
