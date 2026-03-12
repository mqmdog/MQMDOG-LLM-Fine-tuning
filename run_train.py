"""
LLM Fine-tuning 主入口脚本
==========================
统一的训练入口, 支持通过配置文件和命令行参数控制训练流程

使用方式:
    # LoRA微调
    python run_train.py --config configs/lora_config.yaml

    # QLoRA微调
    python run_train.py --config configs/qlora_config.yaml

    # 全参数微调
    python run_train.py --config configs/full_finetune_config.yaml

    # DPO对齐训练
    python run_train.py --config configs/dpo_config.yaml --mode dpo

    # 命令行覆盖参数
    python run_train.py --config configs/lora_config.yaml \
        --override training.learning_rate=1e-4 \
        --override training.num_epochs=5
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

from src.utils.config_parser import ConfigParser
from src.utils.logger import setup_logger
from src.utils.common import set_seed, print_trainable_parameters, get_device_info
from src.models.model_loader import ModelLoader
from src.models.peft_config import PeftConfigFactory
from src.data.data_loader import DataManager
from src.training.trainer import FineTuneTrainer
from src.training.dpo_trainer import DPOFineTuneTrainer
from src.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Fine-tuning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to task config file (e.g., configs/lora_config.yaml)",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base config file (default: configs/base_config.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sft",
        choices=["sft", "dpo"],
        help="Training mode: sft (supervised fine-tuning) or dpo (alignment)",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values (format: key=value, e.g., training.learning_rate=1e-4)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation (skip training)",
    )
    return parser.parse_args()


def parse_overrides(override_list):
    """解析命令行覆盖参数"""
    overrides = {}
    for item in override_list:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        # 尝试转换类型
        try:
            value = json.loads(value)  # 处理数字、布尔值等
        except json.JSONDecodeError:
            pass  # 保持字符串
        overrides[key] = value
    return overrides


def run_sft(config, logger):
    """执行SFT训练流程"""

    # 1. 加载模型
    logger.info("[Step 1/5] Loading model and tokenizer...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load()

    # 2. 应用PEFT（比如LoRA）
    logger.info("[Step 2/5] Applying fine-tuning method...")
    method = config.get("finetuning", {}).get("method", "full")
    peft_factory = PeftConfigFactory(config)
    model = peft_factory.apply_peft(model)
    print_trainable_parameters(model)

    # 3. 准备数据
    logger.info("[Step 3/5] Preparing datasets...")
    template_name = config.get("data", {}).get("template", "alpaca")
    data_manager = DataManager(tokenizer, config, template_name=template_name)
    train_dataset, eval_dataset = data_manager.prepare_datasets()

    # 4. 训练
    logger.info("[Step 4/5] Starting training...")
    trainer = FineTuneTrainer(model, tokenizer, config)
    metrics = trainer.train(train_dataset, eval_dataset)

    # 5. 保存
    logger.info("[Step 5/5] Saving model...")
    output_dir = config.get("training", {}).get("output_dir", "outputs/models")
    trainer.save_model(output_dir)

    # 保存训练配置
    ConfigParser.save(config, os.path.join(output_dir, "training_config.yaml"))

    return model, tokenizer, metrics


def run_dpo(config, logger):
    """执行DPO对齐训练流程"""

    # 1. 加载策略模型
    logger.info("[Step 1/5] Loading policy model...")
    model_loader = ModelLoader(config)
    policy_model, tokenizer = model_loader.load()

    # 应用PEFT到策略模型
    peft_factory = PeftConfigFactory(config)
    policy_model = peft_factory.apply_peft(policy_model)

    # 2. 加载参考模型
    logger.info("[Step 2/5] Loading reference model...")
    ref_model_path = config.get("model", {}).get(
        "ref_model_name_or_path",
        config.get("model", {}).get("model_name_or_path"),
    )
    ref_config = config.copy()
    ref_config["model"] = {**config["model"],
                           "model_name_or_path": ref_model_path}
    ref_config["finetuning"] = {"method": "full"}
    ref_loader = ModelLoader(ref_config)
    ref_model, _ = ref_loader.load()

    # 3. 准备数据
    logger.info("[Step 3/5] Preparing DPO datasets...")
    template_name = config.get("data", {}).get("template", "dpo")
    data_manager = DataManager(tokenizer, config, template_name=template_name)
    train_dataset, eval_dataset = data_manager.prepare_datasets(dpo_mode=True)

    # 4. DPO训练
    logger.info("[Step 4/5] Starting DPO training...")
    dpo_trainer = DPOFineTuneTrainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
    )
    metrics = dpo_trainer.train(train_dataset, eval_dataset)

    # 5. 保存
    logger.info("[Step 5/5] Saving model...")
    output_dir = config.get("training", {}).get(
        "output_dir", "outputs/models/dpo")
    dpo_trainer.save_model(output_dir)

    return policy_model, tokenizer, metrics


def run_evaluation(model, tokenizer, config, logger):
    """运行评估"""
    logger.info("Running post-training evaluation...")
    evaluator = Evaluator(model, tokenizer, config)

    test_prompts = [
        "Explain the concept of gradient descent in machine learning.",
        "Write a Python function to reverse a linked list.",
        "What are the key differences between CNN and RNN?",
    ]

    results = evaluator.evaluate_generation(
        prompts=test_prompts,
        max_new_tokens=256,
        temperature=0.7,
    )

    for i, (prompt, text) in enumerate(
        zip(test_prompts, results["generated_texts"])
    ):
        logger.info(f"\n[Test {i+1}] {prompt}")
        logger.info(f"Response: {text[:300]}")

    output_dir = config.get("training", {}).get("output_dir", "outputs/models")
    Evaluator.save_results(
        results,
        os.path.join(output_dir, "eval_results.json"),
    )


def main():
    args = parse_args()

    # 加载配置
    config = ConfigParser.load(args.config, args.base_config)

    # 应用命令行覆盖
    if args.override:
        overrides = parse_overrides(args.override)
        config = ConfigParser.override(config, overrides)

    # 验证配置
    if not ConfigParser.validate(config):
        sys.exit(1)

    # 配置日志
    log_config = config.get("logging", {})
    logger = setup_logger(
        name="llm_finetune",
        log_dir=log_config.get("log_dir", "outputs/logs"),
        log_level=log_config.get("log_level", "INFO"),
    )

    # 打印启动信息
    logger.info("=" * 70)
    logger.info("  LLM Fine-tuning Framework")
    logger.info(f"  Mode: {args.mode.upper()}")
    logger.info(f"  Config: {args.config}")
    logger.info(
        f"  Method: {config.get('finetuning', {}).get('method', 'unknown')}")
    logger.info(
        f"  Model: {config.get('model', {}).get('model_name_or_path', 'unknown')}")
    logger.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # 环境准备
    set_seed(config.get("training", {}).get("seed", 42))
    device_info = get_device_info()

    # 执行训练
    if not args.eval_only:
        if args.mode == "sft":
            model, tokenizer, metrics = run_sft(config, logger)
        elif args.mode == "dpo":
            model, tokenizer, metrics = run_dpo(config, logger)

        logger.info(f"Training metrics: {metrics}")

        # 训练后评估
        run_evaluation(model, tokenizer, config, logger)
    else:
        logger.info("Eval-only mode: loading model for evaluation...")
        model_loader = ModelLoader(config)
        model, tokenizer = model_loader.load()
        peft_factory = PeftConfigFactory(config)
        model = peft_factory.apply_peft(model)
        run_evaluation(model, tokenizer, config, logger)

    logger.info("=" * 70)
    logger.info("  All Done!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
