"""
文本生成器
=========
核心功能: 封装LLM的推理过程, 支持多种解码策略

解码策略详解:
1. Greedy Decoding: 每步选概率最高的token, 确定性但可能陷入重复
2. Beam Search: 维护top-k候选序列, 质量更高但多样性低
3. Top-k Sampling: 从概率最高的k个token中采样, 增加多样性
4. Top-p (Nucleus) Sampling: 从累积概率达到p的最小token集合中采样
5. Temperature: 控制概率分布的平滑度, T>1更随机, T<1更确定


- Temperature原理: softmax(logits/T), T↑→分布更均匀→更随机
- Top-p vs Top-k: Top-p自适应(不同分布切不同数量), Top-k固定切top个
- 生成时的KV Cache: 缓存之前token的Key/Value, 避免重复计算
- 停止条件: EOS token, max_length, 自定义停止词
"""

import logging
from typing import Dict, Any, Optional, List

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
    TextStreamer,
)
from peft import PeftModel

logger = logging.getLogger(__name__)


class TextGenerator:
    """
    文本生成器

    封装模型推理过程, 支持:
    - 单条/批量生成
    - 流式输出 (Streaming)
    - 多种解码策略配置
    - LoRA模型的自动处理

    使用示例:
        generator = TextGenerator(model, tokenizer, config)
        # 单条生成
        result = generator.generate("请介绍一下深度学习")
        # 流式生成
        generator.generate_stream("请介绍一下深度学习")
        # 批量生成
        results = generator.batch_generate(["问题1", "问题2"])
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
        self.device = next(model.parameters()).device

        # 确保tokenizer配置正确
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_generation_config(self, **kwargs) -> GenerationConfig:
        """
        构建生成配置

        参数优先级: 函数参数 > 配置文件默认值

        关键参数:
        - max_new_tokens: 最大生成token数 (不包含prompt)
        - temperature: 采样温度 (1.0=标准, <1.0更确定, >1.0更随机)
        - top_p: nucleus采样阈值 (0.9表示从累积概率90%的token中采样)
        - top_k: top-k采样数量 (50=只考虑概率最高的50个token)
        - do_sample: 是否使用采样 (False=greedy/beam search)
        - repetition_penalty: 重复惩罚 (>1.0抑制重复)
        """
        defaults = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "num_beams": 1,
        }

        # 合并配置
        gen_params = {**defaults, **kwargs}

        # 如果不使用采样, 关闭temperature和top_p/top_k
        if not gen_params.get("do_sample", True):
            gen_params.pop("temperature", None)
            gen_params.pop("top_p", None)
            gen_params.pop("top_k", None)

        return GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_params,
        )

    @torch.inference_mode()
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成单条文本

        流程:
        1. 将prompt编码为input_ids
        2. 调用model.generate()执行自回归生成
        3. 解码生成的token_ids为文本
        4. 去除prompt部分, 返回纯生成内容

        @torch.inference_mode(): 比no_grad更高效, 禁用梯度计算和autograd追踪

        Args:
            prompt: 输入提示文本
            **kwargs: 生成参数覆盖

        Returns:
            生成的文本 (不含prompt)
        """
        self.model.eval()

        # Tokenize输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("tokenizer", {}).get("max_length", 512),
        ).to(self.device)

        # 构建生成配置
        gen_config = self._build_generation_config(**kwargs)

        # 生成
        output_ids = self.model.generate(
            **inputs,
            generation_config=gen_config,
        )

        # 只取新生成的部分 (去除prompt的token)
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_length:]

        # 解码
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return generated_text

    @torch.inference_mode()
    def generate_stream(self, prompt: str, **kwargs):
        """
        流式生成文本 (逐token输出)

        使用HuggingFace的TextStreamer实现
        适用于交互式场景 (如聊天界面)

        流式生成原理:
        - 每生成一个token就立即输出, 而不是等全部生成完
        - 用户体验更好 (感知延迟低)
        - TextStreamer通过回调机制实现
        """
        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        gen_config = self._build_generation_config(**kwargs)

        # TextStreamer会在每个token生成时打印到控制台
        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        self.model.generate(
            **inputs,
            generation_config=gen_config,
            streamer=streamer,
        )

    @torch.inference_mode()
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **kwargs,
    ) -> List[str]:
        """
        批量生成文本

        实现:
        1. 将prompts分成batch
        2. 使用左填充 (left padding) 使所有序列右对齐
        3. 批量推理, 提高GPU利用率
        4. 分别解码每个样本的生成结果

        为什么生成时要用左填充?
        - 自回归模型从左到右生成, 右边是生成区域
        - 左填充确保所有序列的prompt都右对齐
        - 这样生成时所有序列从同一位置开始, 避免padding干扰
        """
        self.model.eval()

        # 生成时切换为左填充
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        gen_config = self._build_generation_config(**kwargs)
        all_results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get(
                    "tokenizer", {}).get("max_length", 512),
            ).to(self.device)

            output_ids = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )

            # 逐个解码
            prompt_lengths = inputs["attention_mask"].sum(dim=1)
            for j, (output, prompt_len) in enumerate(
                zip(output_ids, prompt_lengths)
            ):
                generated_ids = output[prompt_len:]
                text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                all_results.append(text)

        # 恢复填充方向
        self.tokenizer.padding_side = original_padding_side

        return all_results

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        base_model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "TextGenerator":
        """
        从保存的模型路径加载生成器

        支持两种模型:
        1. 完整模型: 直接从model_path加载
        2. PEFT模型: 先加载base_model, 再加载adapter

        Args:
            model_path: 模型保存路径
            base_model_path: 基座模型路径 (PEFT模型需要)
            config: 可选配置
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = config or {}

        if base_model_path:
            # 加载PEFT模型
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            # 可选: 合并LoRA权重以加速推理
            model = model.merge_and_unload()
            logger.info(f"PEFT model loaded and merged from: {model_path}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info(f"Full model loaded from: {model_path}")

        return cls(model, tokenizer, config)
