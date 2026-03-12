"""
SFT训练器
=========
核心功能: 封装HuggingFace Trainer, 实现SFT (Supervised Fine-Tuning) 训练流程

技术架构:
1. 基于 transformers.Trainer 构建, 继承其分布式训练、混合精度、梯度累积等能力
2. 自定义训练回调 (Callbacks) 实现日志监控和模型保存策略
3. 集成多种优化技术: 梯度检查点、学习率调度、权重衰减等

训练流程:
Input Data → DataCollator → Forward Pass → Loss计算 → Backward Pass → 梯度裁剪 → 参数更新


- Trainer内部使用 Accelerate 库实现分布式训练
- 梯度检查点 (Gradient Checkpointing) 以2x计算换约30%显存
- 有效batch_size = per_device_batch × gradient_accumulation × num_gpus
"""

import os
import logging
from typing import Dict, Any, Optional

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import TrainerCallback
from datasets import Dataset
from peft import PeftModel

from src.data.data_collator import DataCollatorForSFT

logger = logging.getLogger(__name__)


class TrainingMetricsCallback(TrainerCallback):
    """
    自定义训练回调 - 记录详细的训练指标

    在训练过程中捕获:
    - 每步的loss变化趋势 (用于检测训练是否收敛)
    - 学习率变化曲线 (验证lr scheduler是否正常工作)
    - GPU显存使用情况 (用于优化batch_size和梯度累积)
    """

    def __init__(self):
        self.training_loss_history = []
        self.eval_loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.training_loss_history.append({
                    "step": state.global_step,
                    "loss": logs["loss"],
                    "learning_rate": logs.get("learning_rate", 0),
                })
            if "eval_loss" in logs:
                self.eval_loss_history.append({
                    "step": state.global_step,
                    "eval_loss": logs["eval_loss"],
                })

    def on_train_end(self, args, state, control, **kwargs):
        logger.info(
            f"Training completed. Total steps: {state.global_step}, "
            f"Best metric: {state.best_metric}"
        )


class FineTuneTrainer:
    """
    微调训练器

    封装完整的SFT训练流程:
    1. 构建TrainingArguments (控制所有训练超参数)
    2. 创建DataCollator (处理batch动态填充)
    3. 配置回调函数 (日志、早停、模型保存)
    4. 启动训练并返回结果

    使用示例:
        trainer = FineTuneTrainer(model, tokenizer, config)
        result = trainer.train(train_dataset, eval_dataset)
        trainer.save_model("path/to/save")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.training_config = config.get("training", {})
        self.logging_config = config.get("logging", {})
        self.trainer = None
        self.metrics_callback = TrainingMetricsCallback()

    def _build_training_arguments(self) -> TrainingArguments:
        """
        构建TrainingArguments

        关键参数说明:
        - gradient_accumulation_steps: 模拟更大batch_size
          实际效果: effective_batch = per_device_batch × accumulation × num_gpus
          例: batch=4, accumulation=4, 2 GPUs → effective_batch=32

        - gradient_checkpointing: 训练时不保存中间激活值, 反向传播时重新计算
          优势: 显存降低约30%
          代价: 训练时间增加约20-30%

        - lr_scheduler_type: 学习率调度策略
          cosine: 余弦退火, 平滑降低, 训练后期学习率趋近0
          linear: 线性衰减, 简单直观
          cosine_with_restarts: 余弦重启, 适合长训练

        - warmup_ratio: 预热比例
          训练初期逐步增大学习率, 避免初始梯度不稳定导致训练崩溃
          经验值: 0.05-0.1 (即前5%-10%步数用于预热)
        """
        tc = self.training_config
        lc = self.logging_config

        # 确定输出目录
        output_dir = tc.get("output_dir", "outputs/models")
        log_dir = lc.get("log_dir", "outputs/logs")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        args_dict = {
            "output_dir": output_dir,
            "num_train_epochs": tc.get("num_epochs", 3),
            "per_device_train_batch_size": tc.get("per_device_train_batch_size", 4),
            "per_device_eval_batch_size": tc.get("per_device_eval_batch_size", 8),
            "gradient_accumulation_steps": tc.get("gradient_accumulation_steps", 4),
            "learning_rate": tc.get("learning_rate", 2e-5),
            "weight_decay": tc.get("weight_decay", 0.01),
            "max_grad_norm": tc.get("max_grad_norm", 1.0),
            "warmup_steps": tc.get("warmup_steps", 100),
            "lr_scheduler_type": tc.get("lr_scheduler_type", "cosine"),
            "logging_steps": tc.get("logging_steps", 10),
            "save_steps": tc.get("save_steps", 500),
            "save_total_limit": tc.get("save_total_limit", 3),
            "eval_strategy": tc.get("eval_strategy", tc.get("evaluation_strategy", "steps")),
            "eval_steps": tc.get("eval_steps", 500),
            "load_best_model_at_end": tc.get("load_best_model_at_end", True),
            "metric_for_best_model": tc.get("metric_for_best_model", "eval_loss"),
            "greater_is_better": tc.get("greater_is_better", False),
            "fp16": tc.get("fp16", False),
            "bf16": tc.get("bf16", True),
            "gradient_checkpointing": tc.get("gradient_checkpointing", True),
            "dataloader_num_workers": tc.get("dataloader_num_workers", 4),
            "seed": tc.get("seed", 42),
            "report_to": [],
            "remove_unused_columns": False,
        }

        # 配置日志报告工具
        if lc.get("use_tensorboard", False):
            args_dict["report_to"].append("tensorboard")
        if lc.get("use_wandb", False):
            args_dict["report_to"].append("wandb")
            os.environ["WANDB_PROJECT"] = lc.get(
                "wandb_project", "llm-finetuning")
            if lc.get("wandb_run_name"):
                os.environ["WANDB_NAME"] = lc["wandb_run_name"]

        if not args_dict["report_to"]:
            args_dict["report_to"] = ["none"]

        # DeepSpeed配置
        if tc.get("deepspeed"):
            args_dict["deepspeed"] = tc["deepspeed"]

        # 优化器 (QLoRA推荐paged_adamw)
        if tc.get("optim"):
            args_dict["optim"] = tc["optim"]

        return TrainingArguments(**args_dict)

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> Dict[str, float]:
        """
        执行SFT训练

        完整流程:
        1. 构建TrainingArguments
        2. 创建DataCollator
        3. 配置Trainer (模型、参数、数据、回调)
        4. 启动训练循环
        5. 返回训练指标

        Args:
            train_dataset: 预处理后的训练数据集
            eval_dataset: 预处理后的评估数据集 (可选)

        Returns:
            训练结果指标 (包含loss、训练速度等)
        """
        # 1. 构建训练参数
        training_args = self._build_training_arguments()

        # 2. 创建Data Collator
        data_collator = DataCollatorForSFT(
            tokenizer=self.tokenizer,
            padding="longest",
            max_length=self.config.get("tokenizer", {}).get("max_length", 512),
        )

        # 3. 构建回调列表
        callbacks = [self.metrics_callback]

        # 可选: 早停回调 (当验证loss连续N次不下降时停止)
        early_stopping_patience = self.training_config.get(
            "early_stopping_patience")
        if early_stopping_patience and eval_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience)
            )

        # 4. 如果没有eval_dataset，强制关闭evaluation
        if eval_dataset is None:
            training_args.eval_strategy = "no"

        # 5. 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # 5. 启动训练
        logger.info("=" * 60)
        logger.info("Starting SFT Training")
        logger.info(
            f"  Method: {self.config.get('finetuning', {}).get('method', 'unknown')}")
        logger.info(f"  Train samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"  Eval samples: {len(eval_dataset)}")
        logger.info(f"  Epochs: {training_args.num_train_epochs}")
        logger.info(
            f"  Batch size: {training_args.per_device_train_batch_size} × "
            f"{training_args.gradient_accumulation_steps} accumulation"
        )
        logger.info(f"  Learning rate: {training_args.learning_rate}")
        logger.info("=" * 60)

        result = self.trainer.train()

        # 6. 记录结果
        metrics = result.metrics
        logger.info(f"Training completed. Metrics: {metrics}")

        return metrics

    def save_model(self, output_path: Optional[str] = None):
        """
        保存微调后的模型

        保存策略:
        - PEFT模型: 只保存适配器权重 (体积小, 加载时与基座模型合并)
        - 全参数微调: 保存完整模型权重
        - 同时保存tokenizer和训练配置
        """
        save_path = output_path or self.training_config.get(
            "output_dir", "outputs/models"
        )
        os.makedirs(save_path, exist_ok=True)

        if self.trainer:
            self.trainer.save_model(save_path)
        else:
            # 直接保存
            if isinstance(self.model, PeftModel):
                self.model.save_pretrained(save_path)
            else:
                self.model.save_pretrained(save_path)

        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model and tokenizer saved to: {save_path}")

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """在评估数据集上评估模型"""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call train() first.")

        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
