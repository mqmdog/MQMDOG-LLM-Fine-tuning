"""
示例3: DPO对齐微调
==================
演示如何使用DPO (Direct Preference Optimization) 对模型进行人类偏好对齐

应用场景: AI对齐 (Alignment)
- 使模型输出更符合人类偏好 (有用、无害、诚实)
- 相比RLHF: 无需训练奖励模型, 无需PPO, 训练更简单稳定

运行方式:
    python examples/03_dpo_alignment.py --config configs/dpo_config.yaml

训练流程:
    SFT模型 → DPO训练(偏好数据) → 对齐后的模型

DPO vs RLHF:
    RLHF: SFT → 训练RM → PPO(4个模型: actor, critic, ref, reward)
    DPO:  SFT → DPO(2个模型: policy, reference) → 完成
"""

from src.training.dpo_trainer import DPOFineTuneTrainer
from src.data.data_loader import DataManager
from src.models.peft_config import PeftConfigFactory
from src.models.model_loader import ModelLoader
from src.utils.common import set_seed, get_device_info
from src.utils.logger import setup_logger
from src.utils.config_parser import ConfigParser
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="DPO Alignment Fine-tuning")
    parser.add_argument("--config", type=str,
                        default="configs/dpo_config.yaml")
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
    logger.info("  DPO Alignment Training")
    logger.info("=" * 70)

    set_seed(config.get("training", {}).get("seed", 42))
    get_device_info()

    # 加载策略模型 (将被训练)
    logger.info("Loading policy model...")
    model_loader = ModelLoader(config)
    policy_model, tokenizer = model_loader.load()

    # 应用PEFT (可选, DPO也可以全参数微调)
    peft_factory = PeftConfigFactory(config)
    policy_model = peft_factory.apply_peft(policy_model)

    # 加载参考模型 (冻结, 用于计算KL散度)
    logger.info("Loading reference model...")
    ref_model_path = config.get("model", {}).get(
        "ref_model_name_or_path",
        config.get("model", {}).get("model_name_or_path"),
    )
    ref_config = config.copy()
    ref_config["model"]["model_name_or_path"] = ref_model_path
    ref_config["finetuning"] = {"method": "full"}  # 参考模型不需要PEFT
    ref_loader = ModelLoader(ref_config)
    ref_model, _ = ref_loader.load()

    # 准备DPO数据
    logger.info("Preparing DPO datasets...")
    template_name = config.get("data", {}).get("template", "dpo")
    data_manager = DataManager(tokenizer, config, template_name=template_name)
    train_dataset, eval_dataset = data_manager.prepare_datasets(dpo_mode=True)

    # DPO训练
    logger.info("Starting DPO training...")
    dpo_trainer = DPOFineTuneTrainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
    )
    metrics = dpo_trainer.train(train_dataset, eval_dataset)
    logger.info(f"DPO training metrics: {metrics}")

    # 保存
    output_dir = config.get("training", {}).get(
        "output_dir", "outputs/models/dpo-alignment"
    )
    dpo_trainer.save_model(output_dir)

    logger.info("=" * 70)
    logger.info("  DPO Alignment Training Complete!")
    logger.info(f"  Model saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
