"""
DPO训练器
=========
核心功能: 实现DPO (Direct Preference Optimization) 对齐训练

DPO理论背景:
- RLHF的标准流程: SFT → 训练奖励模型(RM) → PPO优化策略
- DPO的创新: 跳过RM训练, 直接从偏好数据优化策略模型
- 数学原理: 证明了最优策略可以用参考策略和偏好数据的闭式解表示

DPO损失函数:
L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

其中:
- π_θ: 当前策略模型 (待优化)
- π_ref: 参考模型 (通常是SFT后的模型, 冻结)
- y_w: chosen response (人类偏好的回答)
- y_l: rejected response (人类不偏好的回答)
- β: 温度参数, 控制偏好强度
- σ: sigmoid函数


- DPO vs RLHF: DPO更简单(无需RM+PPO), 更稳定, 计算成本更低
- β的选择: 太小→偏好不明显, 太大→过度偏离参考模型
- DPO的局限: 对数据质量敏感, 需要高质量的偏好标注
"""

import os
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from datasets import Dataset

logger = logging.getLogger(__name__)


class DPOFineTuneTrainer:
    """
    DPO训练器

    实现了完整的DPO训练流程:
    1. 加载策略模型和参考模型
    2. 计算chosen/rejected的log概率
    3. 应用DPO损失函数
    4. 优化策略模型 (参考模型冻结)

    使用示例:
        trainer = DPOFineTuneTrainer(
            model=policy_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            config=config
        )
        trainer.train(train_dataset, eval_dataset)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
    ):
        """
        Args:
            model: 策略模型 (将被优化)
            ref_model: 参考模型 (冻结, 用于计算KL散度)
            tokenizer: 分词器
            config: 完整配置字典
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.dpo_config = config.get("dpo", {})
        self.training_config = config.get("training", {})

        # DPO超参数
        self.beta = self.dpo_config.get("beta", 0.1)
        self.loss_type = self.dpo_config.get("loss_type", "sigmoid")
        self.label_smoothing = self.dpo_config.get("label_smoothing", 0.0)

        # 冻结参考模型
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _compute_log_probs(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算序列的对数概率

        原理: 对于自回归模型, 序列的概率是每个token条件概率的乘积
        log P(y|x) = Σ log P(y_t | y_{<t}, x)

        实现细节:
        1. 前向传播得到每个位置的logits
        2. 取log_softmax得到对数概率
        3. 使用gather选取实际token的概率
        4. 按attention_mask加权求和
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # logits: [batch, seq_len, vocab_size]
        # 对齐: logits[t]预测的是位置t+1的token
        logits = logits[:, :-1, :]  # 去掉最后一个位置
        labels = input_ids[:, 1:]   # 去掉第一个位置
        mask = attention_mask[:, 1:]  # 对应调整mask

        # 计算每个token的对数概率
        log_probs = F.log_softmax(logits, dim=-1)

        # gather: 选取实际token对应的概率
        per_token_log_probs = log_probs.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        # 按mask求和得到序列级对数概率
        sequence_log_probs = (per_token_log_probs * mask).sum(dim=-1)

        return sequence_log_probs

    def _compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算DPO损失

        数学推导:
        reward_diff = β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x))
                    = β * ((log π_θ(y_w|x) - log π_ref(y_w|x)) - (log π_θ(y_l|x) - log π_ref(y_l|x)))

        标准DPO: L = -log σ(reward_diff)
        IPO变体: L = (reward_diff - 1/(2β))²

        label_smoothing: 平滑处理, 缓解过度自信
        L_smooth = (1-ε) * L_DPO + ε * L_reverse
        """
        # 计算log ratio: log(π_θ/π_ref)
        chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratios = policy_rejected_logps - ref_rejected_logps

        # reward difference (即隐式奖励的差值)
        logits = self.beta * (chosen_log_ratios - rejected_log_ratios)

        if self.loss_type == "sigmoid":
            # 标准DPO损失
            if self.label_smoothing > 0:
                # 带标签平滑的DPO
                losses = (
                    -F.logsigmoid(logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-logits) * self.label_smoothing
                )
            else:
                losses = -F.logsigmoid(logits)
        elif self.loss_type == "ipo":
            # IPO (Identity Preference Optimization) 变体
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(f"Unknown DPO loss type: {self.loss_type}")

        return losses.mean()

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> Dict[str, float]:
        """
        执行DPO训练

        使用TRL库的DPOTrainer或自定义训练循环

        注: 这里提供了自定义实现以展示DPO的核心逻辑
        生产环境建议使用 trl.DPOTrainer

        Args:
            train_dataset: DPO训练数据集
            eval_dataset: DPO评估数据集

        Returns:
            训练指标
        """
        try:
            # 优先使用TRL的DPOTrainer (更成熟稳定)
            from trl import DPOTrainer as TRLDPOTrainer, DPOConfig

            logger.info("Using TRL DPOTrainer for training")
            return self._train_with_trl(train_dataset, eval_dataset)
        except ImportError:
            logger.info("TRL not available, using custom DPO training loop")
            return self._train_custom(train_dataset, eval_dataset)

    def _train_with_trl(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
    ) -> Dict[str, float]:
        """使用TRL库的DPOTrainer进行训练"""
        from trl import DPOTrainer, DPOConfig

        tc = self.training_config
        lc = self.config.get("logging", {})

        output_dir = tc.get("output_dir", "outputs/models/dpo")
        log_dir = lc.get("log_dir", "outputs/logs/dpo")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        dpo_config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=tc.get("num_epochs", 1),
            per_device_train_batch_size=tc.get(
                "per_device_train_batch_size", 2),
            per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=tc.get(
                "gradient_accumulation_steps", 8),
            learning_rate=tc.get("learning_rate", 5e-7),
            lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
            warmup_ratio=tc.get("warmup_ratio", 0.1),
            bf16=tc.get("bf16", True),
            gradient_checkpointing=tc.get("gradient_checkpointing", True),
            logging_dir=log_dir,
            logging_steps=tc.get("logging_steps", 10),
            save_steps=tc.get("save_steps", 200),
            eval_steps=tc.get("eval_steps", 200),
            beta=self.beta,
            loss_type=self.loss_type,
            seed=tc.get("seed", 42),
            report_to=["none"],
            remove_unused_columns=False,
        )

        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("=" * 60)
        logger.info("Starting DPO Training (TRL)")
        logger.info(f"  Beta: {self.beta}")
        logger.info(f"  Loss type: {self.loss_type}")
        logger.info("=" * 60)

        result = trainer.train()
        return result.metrics

    def _train_custom(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
    ) -> Dict[str, float]:
        """
        自定义DPO训练循环 (教学用, 展示DPO核心计算逻辑)

        面试时可以展示对DPO底层实现的理解
        """
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from transformers import get_scheduler

        tc = self.training_config
        device = next(self.model.parameters()).device

        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=tc.get("per_device_train_batch_size", 2),
            shuffle=True,
        )

        # 优化器
        optimizer = AdamW(
            self.model.parameters(),
            lr=tc.get("learning_rate", 5e-7),
            weight_decay=tc.get("weight_decay", 0.01),
        )

        num_training_steps = len(train_loader) * tc.get("num_epochs", 1)
        scheduler = get_scheduler(
            tc.get("lr_scheduler_type", "cosine"),
            optimizer=optimizer,
            num_warmup_steps=int(num_training_steps *
                                 tc.get("warmup_ratio", 0.1)),
            num_training_steps=num_training_steps,
        )

        # 训练循环
        self.model.train()
        global_step = 0
        total_loss = 0.0

        for epoch in range(tc.get("num_epochs", 1)):
            for batch in train_loader:
                # 将数据移到设备
                chosen_ids = batch["chosen_input_ids"].to(device)
                chosen_mask = batch["chosen_attention_mask"].to(device)
                rejected_ids = batch["rejected_input_ids"].to(device)
                rejected_mask = batch["rejected_attention_mask"].to(device)

                # 计算策略模型的log概率
                policy_chosen_logps = self._compute_log_probs(
                    self.model, chosen_ids, chosen_mask
                )
                policy_rejected_logps = self._compute_log_probs(
                    self.model, rejected_ids, rejected_mask
                )

                # 计算参考模型的log概率 (no grad)
                with torch.no_grad():
                    ref_chosen_logps = self._compute_log_probs(
                        self.ref_model, chosen_ids, chosen_mask
                    )
                    ref_rejected_logps = self._compute_log_probs(
                        self.ref_model, rejected_ids, rejected_mask
                    )

                # 计算DPO损失
                loss = self._compute_dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                )

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    tc.get("max_grad_norm", 1.0),
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                global_step += 1

                if global_step % tc.get("logging_steps", 10) == 0:
                    avg_loss = total_loss / global_step
                    logger.info(
                        f"Step {global_step}: loss={loss.item():.4f}, "
                        f"avg_loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}"
                    )

        return {"train_loss": total_loss / max(global_step, 1)}

    def save_model(self, output_path: Optional[str] = None):
        """保存DPO训练后的模型"""
        save_path = output_path or self.training_config.get(
            "output_dir", "outputs/models/dpo"
        )
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"DPO model saved to: {save_path}")
