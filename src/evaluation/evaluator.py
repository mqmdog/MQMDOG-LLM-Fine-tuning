"""
模型评估器
=========
核心功能: 对微调后的模型进行全面评估

评估流程:
1. 在测试集上计算困惑度 (Perplexity)
2. 生成文本并计算BLEU/ROUGE分数
3. (可选) 在下游任务上评估分类性能
4. 汇总所有指标并输出评估报告


- 评估应覆盖多个维度: 语言建模能力、生成质量、任务性能
- 自动评估指标与人工评估往往不完全一致
- 评估时的采样策略 (greedy vs sampling vs beam search) 会影响结果
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

from src.evaluation.metrics import MetricsCalculator
from src.inference.generator import TextGenerator

logger = logging.getLogger(__name__)


class Evaluator:
    """
    模型评估器

    提供标准化的评估流程:
    1. Perplexity评估: 衡量语言模型整体质量
    2. 生成评估: 在给定prompt下生成文本, 计算与参考的相似度
    3. 任务评估: 在特定下游任务上评估性能


        evaluator = Evaluator(model, tokenizer, config)
        results = evaluator.evaluate_all(test_dataset)
        evaluator.save_results(results, "outputs/results/eval.json")
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
        self.metrics = MetricsCalculator()
        self.generator = TextGenerator(model, tokenizer, config)
        self.device = next(model.parameters()).device

    def evaluate_perplexity(self, dataset: Dataset) -> Dict[str, float]:
        """
        在数据集上评估模型困惑度

        实现:
        1. 遍历数据集中的每个样本
        2. 计算每个样本的交叉熵loss
        3. 对所有样本取平均得到整体loss
        4. PPL = exp(avg_loss)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]

                input_ids = torch.tensor(
                    [sample["input_ids"]], device=self.device
                )
                attention_mask = torch.tensor(
                    [sample["attention_mask"]], device=self.device
                )
                labels = torch.tensor(
                    [sample["labels"]], device=self.device
                )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # 累积loss (按token数加权)
                num_tokens = (labels != -100).sum().item()
                if num_tokens > 0:
                    total_loss += outputs.loss.item() * num_tokens
                    total_tokens += num_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = self.metrics.compute_perplexity(avg_loss)

        results = {
            "eval_loss": avg_loss,
            "perplexity": ppl,
            "total_tokens": total_tokens,
        }

        logger.info(
            f"Perplexity evaluation: loss={avg_loss:.4f}, PPL={ppl:.2f}")
        return results

    def evaluate_generation(
        self,
        prompts: List[str],
        references: Optional[List[str]] = None,
        **generation_kwargs,
    ) -> Dict[str, Any]:
        """
        评估文本生成质量

        流程:
        1. 对每个prompt生成回复
        2. 如果有参考答案, 计算BLEU和ROUGE-L
        3. 返回生成结果和评估指标

        Args:
            prompts: 输入提示列表
            references: 参考答案列表 (可选)
            **generation_kwargs: 生成参数 (max_new_tokens, temperature等)
        """
        # 生成回复
        generated_texts = []
        for prompt in prompts:
            text = self.generator.generate(prompt, **generation_kwargs)
            generated_texts.append(text)

        results = {"generated_texts": generated_texts}

        # 如果有参考答案, 计算自动评估指标
        if references:
            bleu_scores = self.metrics.compute_bleu(
                generated_texts, references)
            rouge_scores = self.metrics.compute_rouge_l(
                generated_texts, references)

            results.update(bleu_scores)
            results.update(rouge_scores)

            logger.info(
                f"Generation evaluation: "
                f"BLEU-4={bleu_scores.get('bleu-4', 0):.4f}, "
                f"ROUGE-L={rouge_scores.get('rouge-l-f1', 0):.4f}"
            )

        return results

    def evaluate_all(
        self,
        test_dataset: Optional[Dataset] = None,
        prompts: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        执行全面评估

        综合perplexity和生成质量评估
        """
        all_results = {}

        if test_dataset is not None:
            logger.info("Evaluating perplexity...")
            ppl_results = self.evaluate_perplexity(test_dataset)
            all_results["perplexity_eval"] = ppl_results

        if prompts is not None:
            logger.info("Evaluating generation quality...")
            gen_results = self.evaluate_generation(
                prompts, references,
                max_new_tokens=256,
                temperature=0.7,
            )
            all_results["generation_eval"] = gen_results

        return all_results

    @staticmethod
    def save_results(results: Dict[str, Any], output_path: str):
        """保存评估结果到JSON文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 将numpy类型转换为Python原生类型
        def convert(obj):
            if hasattr(obj, "item"):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        serializable = convert(results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to: {output_path}")
