"""
示例5: 文本分类 - 情感分析微调
==============================
演示如何将预训练LLM微调为文本分类模型 (以情感分析为例)

应用场景: 文本分类与情感分析
- 将生成式LLM适配到分类任务
- 两种方案: (1) 用生成方式做分类 (2) 添加分类头

运行方式:
    python examples/05_text_classification.py

技术要点:
1. 生成式分类: 让模型生成"positive"/"negative"等标签文本
2. 分类头方式: 在LLM最后一层隐藏状态上添加线性分类器
3. Prompt设计对分类效果影响很大
"""

from src.evaluation.metrics import MetricsCalculator
from src.utils.common import set_seed
from src.utils.logger import setup_logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LLMForClassification(nn.Module):
    """
    基于LLM的文本分类模型

    方案: 在预训练LLM的最后隐藏层上添加分类头

    架构:
    Input → LLM Encoder → [CLS] hidden state → Dropout → Linear → Softmax → Label

    这里使用序列最后一个非padding token的隐藏状态作为句子表示
    (因为causal LM没有[CLS] token, 最后一个token汇聚了全序列信息)

    面试要点:
    - 为什么用最后一个token? → Causal LM单向注意力, 最后token看到了所有前面的token
    - 为什么要冻结大部分参数? → 防止灾难性遗忘, 且分类任务数据通常较少
    """

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.num_labels = num_labels

        # 分类头: Dropout + Linear
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_labels),
        )

        # 冻结基座模型, 只训练分类头
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        # 获取LLM的最后隐藏状态
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # 取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]

        # 取每个样本最后一个非padding token的表示
        # 通过attention_mask找到最后一个1的位置
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1  # 最后一个非pad位置

        # 使用gather提取对应位置的hidden state
        pooled_output = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths,
        ]

        # 分类
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


def main():
    logger = setup_logger(name="classification", log_dir="outputs/logs")

    logger.info("=" * 70)
    logger.info("  Text Classification (Sentiment Analysis) Example")
    logger.info("=" * 70)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_name = "Qwen/Qwen2.5-0.5B"
    logger.info(f"Loading base model: {model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建分类模型
    model = LLMForClassification(base_model, num_labels=2).to(device)

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)"
    )

    # 示例数据 (实际应用中使用SST-2, IMDB等数据集)
    texts = [
        "This movie is absolutely wonderful and I loved every minute of it!",
        "The food was terrible and the service was even worse.",
        "I had a great experience at this restaurant, highly recommend!",
        "This product broke after just one day, very disappointed.",
        "The book was engaging and well-written from start to finish.",
        "I regret buying this, it was a complete waste of money.",
        "The concert was amazing, the band played beautifully!",
        "The hotel room was dirty and the staff was rude.",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

    # Tokenize
    encodings = tokenizer(
        texts,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    label_tensor = torch.tensor(labels, device=device)

    # 训练
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)

    model.train()
    for epoch in range(20):
        outputs = model(input_ids, attention_mask, label_tensor)
        loss = outputs["loss"]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 5 == 0:
            # 评估
            model.eval()
            with torch.no_grad():
                eval_outputs = model(input_ids, attention_mask)
                predictions = eval_outputs["logits"].argmax(
                    dim=-1).cpu().tolist()

            metrics = MetricsCalculator.compute_classification_metrics(
                predictions, labels
            )
            logger.info(
                f"Epoch {epoch+1}: loss={loss.item():.4f}, "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"f1={metrics['f1']:.4f}"
            )
            model.train()

    # 最终评估
    model.eval()
    with torch.no_grad():
        eval_outputs = model(input_ids, attention_mask)
        predictions = eval_outputs["logits"].argmax(dim=-1).cpu().tolist()

    logger.info("\n--- Final Results ---")
    for text, pred, true in zip(texts, predictions, labels):
        sentiment = "Positive" if pred == 1 else "Negative"
        correct = "OK" if pred == true else "WRONG"
        logger.info(f"  [{correct}] {sentiment}: {text[:60]}...")

    logger.info("=" * 70)
    logger.info("  Classification Example Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
