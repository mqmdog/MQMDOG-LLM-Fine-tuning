"""
示例1: LoRA微调对话模型
======================
演示如何使用LoRA方法微调一个对话/指令遵循模型

应用场景: 文本生成与对话系统
- 将基座模型(如Qwen-0.5B)微调为能够遵循指令的对话模型
- 使用Alpaca格式的指令数据进行监督微调(SFT)

运行方式:
    python examples/01_lora_chat.py --config configs/lora_config.yaml

技术要点:
1. LoRA在注意力层Q/K/V/O和MLP层插入低秩矩阵
2. 仅训练约0.1%的参数即可达到接近全参数微调的效果
3. 训练后的adapter可独立保存, 体积小 (几十MB vs 几GB)
"""

from src.evaluation.evaluator import Evaluator
from src.training.trainer import FineTuneTrainer
from src.data.data_loader import DataManager
from src.models.peft_config import PeftConfigFactory
from src.models.model_loader import ModelLoader
from src.utils.common import set_seed, print_trainable_parameters, get_device_info
from src.utils.logger import setup_logger
from src.utils.config_parser import ConfigParser
import os
import sys
import argparse
import logging

# 将项目根目录加入PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # ==================== 1. 解析参数和配置 ====================
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for Chat Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base config file",
    )
    args = parser.parse_args()

    # 加载配置
    config = ConfigParser.load(args.config, args.base_config)

    # 配置日志
    log_config = config.get("logging", {})
    logger = setup_logger(
        name="llm_finetune",
        log_dir=log_config.get("log_dir", "outputs/logs"),
        log_level=log_config.get("log_level", "INFO"),
    )

    logger.info("=" * 70)
    logger.info("  LoRA Fine-tuning for Chat Model")
    logger.info("=" * 70)

    # ==================== 2. 环境准备 ====================
    set_seed(config.get("training", {}).get("seed", 42))
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")

    # ==================== 3. 加载模型和Tokenizer ====================
    logger.info("Loading model and tokenizer...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load()

    # ==================== 4. 应用LoRA ====================
    logger.info("Applying LoRA adapter...")
    peft_factory = PeftConfigFactory(config)
    model = peft_factory.apply_peft(model)
    print_trainable_parameters(model)

    # ==================== 5. 准备数据 ====================
    logger.info("Preparing datasets...")
    template_name = config.get("data", {}).get("template", "alpaca")
    data_manager = DataManager(tokenizer, config, template_name=template_name)
    train_dataset, eval_dataset = data_manager.prepare_datasets()

    logger.info(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval samples: {len(eval_dataset)}")

    # ==================== 6. 训练 ====================
    logger.info("Starting training...")
    trainer = FineTuneTrainer(model, tokenizer, config)
    metrics = trainer.train(train_dataset, eval_dataset)
    logger.info(f"Training metrics: {metrics}")

    # ==================== 7. 保存模型 ====================
    output_dir = config.get("training", {}).get(
        "output_dir", "outputs/models/lora-chat")
    trainer.save_model(output_dir)

    # ==================== 8. 评估 ====================
    logger.info("Running evaluation...")
    evaluator = Evaluator(model, tokenizer, config)

    # 测试生成
    test_prompts = [
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain what is machine learning in simple terms.\n\n### Response:\n",
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a Python function to calculate fibonacci numbers.\n\n### Response:\n",
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat are the benefits of exercise?\n\n### Response:\n",
    ]

    gen_results = evaluator.evaluate_generation(
        prompts=test_prompts,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

    for i, (prompt, response) in enumerate(
        zip(test_prompts, gen_results["generated_texts"])
    ):
        instruction = prompt.split("### Instruction:\n")[1].split("\n")[0]
        logger.info(f"\n--- Test {i+1} ---")
        logger.info(f"Instruction: {instruction}")
        logger.info(f"Response: {response[:200]}...")

    # 保存评估结果
    Evaluator.save_results(
        gen_results,
        os.path.join(output_dir, "eval_results.json"),
    )

    logger.info("=" * 70)
    logger.info("  LoRA Fine-tuning Complete!")
    logger.info(f"  Model saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
