"""
数据加载与预处理管理器
====================
核心功能: 统一管理数据集的加载、预处理、分词和划分

技术架构:
1. 支持从HuggingFace Hub或本地路径加载数据
2. 使用数据模板将原始数据转换为统一格式
3. 实现Tokenization并构建模型输入 (input_ids, attention_mask, labels)
4. 关键: SFT训练中只对response部分计算loss (通过labels=-100实现mask)


- 数据预处理是微调的关键环节, 直接影响训练效果
- Label Masking: 将prompt部分的labels设为-100, CrossEntropyLoss会自动忽略
- 动态Padding vs 静态Padding的trade-off
"""

import os
import logging
from typing import Dict, Optional, Any, Union, List

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from src.data.data_template import TemplateManager, Template

logger = logging.getLogger(__name__)


class DataManager:
    """
    数据管理器

    职责:
    1. 加载原始数据集 (HuggingFace Hub / 本地文件)
    2. 应用数据模板进行格式转换
    3. Tokenization并构建训练所需的input_ids, attention_mask, labels
    4. 数据集划分 (train/eval/test)

    使用示例:
        dm = DataManager(tokenizer=tokenizer, config=data_config)
        train_dataset, eval_dataset = dm.prepare_datasets()
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        template_name: str = "alpaca",
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer实例
            config: 数据相关配置字典
            template_name: 使用的数据模板名称
        """
        self.tokenizer = tokenizer
        self.config = config
        self.template_manager = TemplateManager()
        self.template = self.template_manager.get_template(template_name)
        self.max_length = config.get("tokenizer", {}).get("max_length", 512)

        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(
                f"Set pad_token to eos_token: {self.tokenizer.eos_token}"
            )

    def load_dataset(self) -> DatasetDict:
        """
        加载数据集

        支持三种数据源:
        1. HuggingFace Hub: 通过dataset_name指定 (如 "tatsu-lab/alpaca")
        2. 本地文件: 通过dataset_path指定 (支持json, csv, parquet等)
        3. 预处理后的数据: 直接加载Arrow格式

        Returns:
            DatasetDict: 包含train和可能的eval/test集
        """
        data_config = self.config.get("data", {})
        dataset_name = data_config.get("dataset_name")
        dataset_path = data_config.get("dataset_path")

        if dataset_name:
            logger.info(
                f"Loading dataset from HuggingFace Hub: {dataset_name}")
            dataset = load_dataset(dataset_name)
        elif dataset_path:
            logger.info(f"Loading dataset from local path: {dataset_path}")
            if os.path.isdir(dataset_path):
                dataset = load_dataset(dataset_path)
            else:
                # 根据文件扩展名判断格式
                ext = os.path.splitext(dataset_path)[1].lower()
                format_map = {
                    ".json": "json",
                    ".jsonl": "json",
                    ".csv": "csv",
                    ".parquet": "parquet",
                    ".tsv": "csv",
                }
                file_format = format_map.get(ext, "json")
                dataset = load_dataset(
                    file_format,
                    data_files=dataset_path,
                    split="train",
                )
                # 单文件加载时需要手动划分
                dataset = self._split_dataset(dataset, data_config)
        else:
            raise ValueError(
                "Must specify either 'dataset_name' or 'dataset_path' in config"
            )

        # 限制样本数量 (用于快速实验和调试)
        max_samples = data_config.get("max_samples")
        if max_samples:
            dataset = self._limit_samples(dataset, max_samples)
            logger.info(f"Limited dataset to {max_samples} samples per split")

        return dataset

    def _split_dataset(
        self, dataset: Dataset, config: Dict[str, Any]
    ) -> DatasetDict:
        """
        将单个数据集划分为训练集和验证集

        使用分层抽样确保数据分布一致性 (当有标签列时)
        """
        split_ratio = config.get("train_val_split_ratio", 0.9)
        seed = config.get("preprocessing", {}).get("seed", 42)

        split = dataset.train_test_split(
            test_size=1 - split_ratio,
            seed=seed,
        )
        return DatasetDict({
            "train": split["train"],
            "validation": split["test"],
        })

    def _limit_samples(
        self, dataset: Union[Dataset, DatasetDict], max_samples: int
    ) -> Union[Dataset, DatasetDict]:
        """限制每个split的最大样本数"""
        if isinstance(dataset, DatasetDict):
            for split_name in dataset:
                if len(dataset[split_name]) > max_samples:
                    dataset[split_name] = dataset[split_name].select(
                        range(max_samples)
                    )
        elif isinstance(dataset, Dataset):
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
        return dataset

    def preprocess_sft(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        SFT (Supervised Fine-Tuning) 数据预处理

        核心逻辑:
        1. 使用模板将原始数据转换为 prompt + response
        2. 分别对prompt和response进行tokenize
        3. 拼接得到完整的input_ids
        4. 构建labels: prompt部分设为-100 (不计算loss), response部分保留原始token_id

        这样模型只学习如何生成response, 而不是记忆prompt

        Args:
            examples: 批量原始样本 (HuggingFace datasets map格式)

        Returns:
            包含 input_ids, attention_mask, labels 的字典
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        # 获取第一个key来确定batch大小
        first_key = next(iter(examples.keys()))
        batch_size = len(examples[first_key])

        for i in range(batch_size):
            # 提取单个样本
            example = {k: v[i] for k, v in examples.items()}

            # 使用模板格式化
            formatted = self.template.format_example(example)
            prompt = formatted["prompt"]
            response = formatted["response"]

            # 分别tokenize prompt和response
            # 注意: 不加特殊token, 手动控制拼接
            prompt_ids = self.tokenizer.encode(
                prompt, add_special_tokens=False
            )
            response_ids = self.tokenizer.encode(
                response, add_special_tokens=False
            )

            # 拼接: [BOS] + prompt + response + [EOS]
            # BOS token在某些模型中是可选的
            input_ids = prompt_ids + response_ids + \
                [self.tokenizer.eos_token_id]

            # 构建labels: prompt部分为-100 (masked), response部分保留
            # -100是PyTorch CrossEntropyLoss的ignore_index默认值
            labels = [-100] * len(prompt_ids) + \
                response_ids + [self.tokenizer.eos_token_id]

            # 截断到最大长度
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]

            attention_mask = [1] * len(input_ids)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }

    def preprocess_dpo(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        DPO (Direct Preference Optimization) 数据预处理

        核心逻辑:
        1. 将每个样本拆分为两组: (prompt + chosen) 和 (prompt + rejected)
        2. 分别tokenize这两组
        3. DPOTrainer需要: prompt_input_ids, chosen_input_ids, rejected_input_ids

        - DPO训练需要成对的数据 (chosen vs rejected)
        - 参考模型(ref_model)用于计算KL散度, 防止策略模型偏离太远
        """
        batch_prompt_ids = []
        batch_chosen_ids = []
        batch_rejected_ids = []

        first_key = next(iter(examples.keys()))
        batch_size = len(examples[first_key])

        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            formatted = self.template.format_example(example)

            prompt = formatted.get("prompt", "")
            chosen = formatted.get("chosen", "")
            rejected = formatted.get("rejected", "")

            prompt_ids = self.tokenizer.encode(
                prompt, add_special_tokens=False)
            chosen_ids = self.tokenizer.encode(
                chosen, add_special_tokens=False)
            rejected_ids = self.tokenizer.encode(
                rejected, add_special_tokens=False)

            batch_prompt_ids.append(prompt_ids)
            batch_chosen_ids.append(
                prompt_ids + chosen_ids + [self.tokenizer.eos_token_id]
            )
            batch_rejected_ids.append(
                prompt_ids + rejected_ids + [self.tokenizer.eos_token_id]
            )

        return {
            "prompt_input_ids": batch_prompt_ids,
            "chosen_input_ids": batch_chosen_ids,
            "rejected_input_ids": batch_rejected_ids,
        }

    def prepare_datasets(
        self, dpo_mode: bool = False
    ) -> tuple:
        """
        准备训练和评估数据集的完整流程

        Args:
            dpo_mode: 是否为DPO训练模式

        Returns:
            (train_dataset, eval_dataset) 元组
        """
        # 1. 加载原始数据
        raw_datasets = self.load_dataset()

        # 2. 选择预处理函数
        preprocess_fn = self.preprocess_dpo if dpo_mode else self.preprocess_sft

        # 3. 应用预处理
        data_config = self.config.get("data", {}).get("preprocessing", {})
        num_workers = data_config.get("num_workers", 4)

        processed_datasets = {}
        for split_name, split_dataset in raw_datasets.items():
            logger.info(
                f"Processing {split_name} split ({len(split_dataset)} samples)..."
            )
            processed = split_dataset.map(
                preprocess_fn,
                batched=True,
                num_proc=num_workers,
                remove_columns=split_dataset.column_names,
                desc=f"Tokenizing {split_name}",
            )
            processed_datasets[split_name] = processed

        train_dataset = processed_datasets.get("train")
        eval_dataset = processed_datasets.get(
            "validation", processed_datasets.get("test")
        )

        if train_dataset:
            logger.info(f"Train dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Eval dataset size: {len(eval_dataset)}")

        return train_dataset, eval_dataset
