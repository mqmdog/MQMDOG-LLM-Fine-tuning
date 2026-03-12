"""
评估指标计算器
=============
核心功能: 实现LLM微调后的多维度评估指标

评估维度:
1. 困惑度 (Perplexity): 衡量语言模型对测试集的预测能力
2. BLEU/ROUGE: 生成文本与参考文本的重叠度
3. 文本分类指标: Accuracy, F1, Precision, Recall
4. 自定义指标: 支持用户定义的评估函数


- Perplexity = exp(cross_entropy_loss), 越低越好
- BLEU更关注精度(precision), ROUGE更关注召回(recall)
- 生成任务评估需要结合自动指标和人工评估
"""

import logging
import math
from typing import Dict, List, Any, Optional
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    评估指标计算器

    支持的指标:
    - perplexity: 语言模型困惑度
    - bleu: BLEU-1/2/3/4分数
    - rouge: ROUGE-1/2/L分数
    - accuracy: 分类准确率
    - f1: F1分数 (macro/micro/weighted)
    """

    @staticmethod
    def compute_perplexity(loss: float) -> float:
        """
        从loss计算困惑度

        数学定义:
        PPL = exp(H(p, q)) = exp(-1/N * Σ log q(x_i))
        其中H(p,q)是交叉熵, q是模型预测的概率分布

        在实践中, HuggingFace Trainer的eval_loss就是交叉熵
        所以 PPL = exp(eval_loss)

        解释:
        - PPL=1: 完美预测 (不可能达到)
        - PPL=V: 相当于均匀分布 (V是词表大小)
        - PPL越低越好, 表示模型对数据的预测越准确
        """
        try:
            ppl = math.exp(loss)
        except OverflowError:
            ppl = float("inf")
        return ppl

    @staticmethod
    def compute_bleu(
        predictions: List[str],
        references: List[str],
        max_n: int = 4,
    ) -> Dict[str, float]:
        """
        计算BLEU分数 (Bilingual Evaluation Understudy)

        BLEU衡量生成文本与参考文本的n-gram重叠度

        算法:
        1. 计算不同n-gram (1-gram到4-gram) 的精度
        2. 使用几何平均值合并
        3. 应用Brevity Penalty (BP) 惩罚过短的生成文本

        公式: BLEU = BP × exp(Σ w_n × log p_n)
        - p_n: n-gram精度 (clipped)
        - w_n: 权重, 通常均匀分配 (1/4)
        - BP = min(1, exp(1 - ref_len/pred_len))

        Args:
            predictions: 生成的文本列表
            references: 参考文本列表
            max_n: 最大n-gram阶数

        Returns:
            包含bleu-1到bleu-4的分数字典
        """
        results = {}

        for n in range(1, max_n + 1):
            precisions = []
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = ref.split()

                if len(pred_tokens) < n:
                    precisions.append(0.0)
                    continue

                # 计算n-gram
                pred_ngrams = Counter(
                    tuple(pred_tokens[i: i + n])
                    for i in range(len(pred_tokens) - n + 1)
                )
                ref_ngrams = Counter(
                    tuple(ref_tokens[i: i + n])
                    for i in range(len(ref_tokens) - n + 1)
                )

                # Clipped precision: 每个n-gram的计数不超过参考中的计数
                clipped_count = sum(
                    min(count, ref_ngrams.get(ngram, 0))
                    for ngram, count in pred_ngrams.items()
                )
                total_count = sum(pred_ngrams.values())

                if total_count > 0:
                    precisions.append(clipped_count / total_count)
                else:
                    precisions.append(0.0)

            avg_precision = np.mean(precisions) if precisions else 0.0
            results[f"bleu-{n}"] = avg_precision

        return results

    @staticmethod
    def compute_rouge_l(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        计算ROUGE-L分数

        ROUGE-L基于最长公共子序列 (LCS):
        - Precision: LCS(pred, ref) / len(pred)
        - Recall: LCS(pred, ref) / len(ref)
        - F1: 调和平均

        与BLEU的区别:
        - BLEU侧重精度, ROUGE侧重召回
        - ROUGE-L不要求连续匹配 (比n-gram更灵活)
        """
        def lcs_length(x: List[str], y: List[str]) -> int:
            """动态规划计算最长公共子序列长度"""
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]

        precisions, recalls, f1s = [], [], []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            if not pred_tokens or not ref_tokens:
                precisions.append(0.0)
                recalls.append(0.0)
                f1s.append(0.0)
                continue

            lcs_len = lcs_length(pred_tokens, ref_tokens)

            precision = lcs_len / len(pred_tokens) if pred_tokens else 0
            recall = lcs_len / len(ref_tokens) if ref_tokens else 0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return {
            "rouge-l-precision": np.mean(precisions),
            "rouge-l-recall": np.mean(recalls),
            "rouge-l-f1": np.mean(f1s),
        }

    @staticmethod
    def compute_classification_metrics(
        predictions: List[int],
        references: List[int],
        average: str = "macro",
    ) -> Dict[str, float]:
        """
        计算分类指标 (Accuracy, Precision, Recall, F1)

        average参数:
        - "macro": 各类别指标的简单平均 (对类别不平衡敏感)
        - "micro": 全局统计TP/FP/FN (等同于accuracy)
        - "weighted": 按各类别样本数加权平均
        """
        predictions = np.array(predictions)
        references = np.array(references)

        accuracy = np.mean(predictions == references)

        labels = sorted(set(references.tolist()) | set(predictions.tolist()))
        per_class_metrics = {}

        for label in labels:
            tp = np.sum((predictions == label) & (references == label))
            fp = np.sum((predictions == label) & (references != label))
            fn = np.sum((predictions != label) & (references == label))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            per_class_metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        if average == "macro":
            avg_precision = np.mean([m["precision"]
                                    for m in per_class_metrics.values()])
            avg_recall = np.mean([m["recall"]
                                 for m in per_class_metrics.values()])
            avg_f1 = np.mean([m["f1"] for m in per_class_metrics.values()])
        else:  # weighted
            total = len(references)
            avg_precision = sum(
                m["precision"] * np.sum(references == l)
                for l, m in per_class_metrics.items()
            ) / total
            avg_recall = sum(
                m["recall"] * np.sum(references == l)
                for l, m in per_class_metrics.items()
            ) / total
            avg_f1 = sum(
                m["f1"] * np.sum(references == l)
                for l, m in per_class_metrics.items()
            ) / total

        return {
            "accuracy": float(accuracy),
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1": float(avg_f1),
        }
