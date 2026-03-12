"""
示例4: 知识蒸馏 - 用大模型教小模型
===================================
演示如何将大模型(Teacher)的知识蒸馏到小模型(Student)

应用场景: 模型压缩与部署优化
- 大模型推理成本高, 小模型性能不足
- 知识蒸馏: 用大模型的输出分布指导小模型学习
- 部署时只用小模型, 兼顾性能和效率

运行方式:
    python examples/04_knowledge_distillation.py

知识蒸馏原理:
    L_total = α * L_CE(student, hard_labels) + (1-α) * L_KD(student, teacher)
    L_KD = KL(softmax(z_s/T), softmax(z_t/T)) * T²

其中:
- z_s, z_t: Student和Teacher的logits
- T: 温度参数 (T>1使分布更平滑, 暴露更多"暗知识")
- α: 硬标签和软标签损失的权重

面试要点:
- 为什么要用温度T? → 软化概率分布, 让小概率类别的信息也能传递
- 为什么乘T²? → 梯度补偿, 温度越高logits缩放越大, 需要补偿
- "暗知识"(Dark Knowledge): Teacher模型中非最大概率类别的概率分布
"""

from src.utils.common import set_seed
from src.utils.logger import setup_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数

    组合两部分损失:
    1. 硬标签损失 (Hard Label Loss): Student预测 vs 真实标签
       - 标准交叉熵, 确保Student能正确预测
    2. 软标签损失 (Soft Label Loss): Student分布 vs Teacher分布
       - KL散度, 让Student学习Teacher的概率分布

    温度T的作用:
    - T=1: 标准softmax, 概率集中在top类别
    - T=5: 分布变平滑, 各类别概率差异缩小
    - T=20: 非常平滑, 近似均匀分布
    - 实践中T=2~5效果较好
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        """
        Args:
            temperature: 蒸馏温度 (T>1使分布更平滑)
            alpha: 硬标签损失权重 (1-alpha为软标签权重)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算蒸馏损失

        Args:
            student_logits: Student模型输出 [batch, seq_len, vocab]
            teacher_logits: Teacher模型输出 [batch, seq_len, vocab]
            labels: 真实标签 [batch, seq_len]

        Returns:
            总损失值
        """
        # 1. 硬标签损失 (标准交叉熵)
        # 将logits展平为2D以计算交叉熵
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # 2. 软标签损失 (KL散度)
        # 使用温度T软化分布
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        # KL散度: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
        # F.kl_div的输入是log_probs, target是probs
        soft_loss = F.kl_div(
            student_soft.view(-1, student_soft.size(-1)),
            teacher_soft.view(-1, teacher_soft.size(-1)),
            reduction="batchmean",
        )

        # 乘以T²补偿温度缩放对梯度的影响
        soft_loss = soft_loss * (T ** 2)

        # 3. 组合损失
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return total_loss


def create_sample_data(tokenizer, num_samples=100, max_length=128):
    """创建示例数据 (实际使用时替换为真实数据集)"""
    texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
        "Natural language processing helps computers understand and generate human language.",
        "Transfer learning allows models trained on one task to be adapted for related tasks.",
        "Reinforcement learning trains agents through interaction with an environment using rewards.",
    ] * (num_samples // 5)

    encodings = tokenizer(
        texts[:num_samples],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # 创建简单的数据集
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.encodings["input_ids"][idx].clone(),
            }

    return SimpleDataset(encodings)


def main():
    logger = setup_logger(
        name="distillation",
        log_dir="outputs/logs",
    )

    logger.info("=" * 70)
    logger.info("  Knowledge Distillation Example")
    logger.info("=" * 70)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 配置
    teacher_model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 实际应用中用更大的模型
    student_model_name = "Qwen/Qwen2.5-0.5B"  # 小模型
    temperature = 2.0
    alpha = 0.5
    num_epochs = 3
    batch_size = 4
    learning_rate = 2e-5

    # 加载Teacher模型 (冻结)
    logger.info(f"Loading teacher model: {teacher_model_name}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 加载Student模型
    logger.info(f"Loading student model: {student_model_name}")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备数据
    dataset = create_sample_data(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 蒸馏损失和优化器
    distill_loss_fn = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    num_training_steps = len(dataloader) * num_epochs
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )

    # 训练循环
    logger.info("Starting knowledge distillation training...")
    student_model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Teacher前向传播 (no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_outputs.logits

            # Student前向传播
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_logits = student_outputs.logits

            # 计算蒸馏损失
            loss = distill_loss_fn(student_logits, teacher_logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if (step + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Step {step+1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # 保存Student模型
    output_dir = "outputs/models/distilled-model"
    os.makedirs(output_dir, exist_ok=True)
    student_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("=" * 70)
    logger.info("  Knowledge Distillation Complete!")
    logger.info(f"  Student model saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
