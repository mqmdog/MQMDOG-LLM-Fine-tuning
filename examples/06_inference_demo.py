"""
示例6: 模型推理与部署
====================
演示微调后模型的推理用法, 包括:
1. 加载LoRA/QLoRA适配器并合并到基座模型
2. 单条和批量推理
3. 流式输出
4. 不同解码策略的对比

运行方式:
    python examples/06_inference_demo.py --model_path outputs/models/lora-chat

面试要点:
- LoRA推理两种方式: (1) 保留adapter动态计算 (2) merge_and_unload合并后推理
- 合并后推理无额外计算开销, 但不能再切换不同adapter
- KV Cache: 缓存历史token的Key/Value, 避免重复计算, 推理速度提升N倍
"""

from src.inference.generator import TextGenerator
from src.utils.logger import setup_logger
import os
import sys
import argparse
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Model Inference Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/models/lora-chat",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to base model (required for PEFT models)",
    )
    parser.add_argument(
        "--use_base_model",
        action="store_true",
        help="Use base model directly for demo (no fine-tuned model needed)",
    )
    args = parser.parse_args()

    logger = setup_logger(name="inference_demo")

    logger.info("=" * 70)
    logger.info("  Model Inference Demo")
    logger.info("=" * 70)

    # 加载模型
    if args.use_base_model:
        # 直接用基座模型演示 (无需微调)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Loading base model for demo: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        generator = TextGenerator(model, tokenizer, {})
    else:
        logger.info(f"Loading fine-tuned model from: {args.model_path}")
        generator = TextGenerator.from_pretrained(
            model_path=args.model_path,
            base_model_path=args.base_model_path,
        )

    # ==================== 测试用例 ====================

    test_prompts = [
        # 1. 通用对话
        "What is the difference between machine learning and deep learning?",
        # 2. 代码生成
        "Write a Python function that implements binary search.",
        # 3. 知识问答
        "Explain the attention mechanism in transformers.",
        # 4. 文本摘要
        "Summarize the following: Deep learning is a subset of machine learning that uses neural networks with multiple layers. These networks can automatically learn representations of data at multiple levels of abstraction.",
    ]

    # ==================== 1. Greedy解码 ====================
    logger.info("\n" + "=" * 50)
    logger.info("  Greedy Decoding (deterministic)")
    logger.info("=" * 50)

    for i, prompt in enumerate(test_prompts[:2]):
        start_time = time.time()
        result = generator.generate(
            prompt,
            max_new_tokens=128,
            do_sample=False,  # Greedy
        )
        elapsed = time.time() - start_time
        logger.info(f"\n--- Prompt {i+1} (Greedy, {elapsed:.2f}s) ---")
        logger.info(f"Q: {prompt[:80]}...")
        logger.info(f"A: {result[:200]}...")

    # ==================== 2. Sampling解码 ====================
    logger.info("\n" + "=" * 50)
    logger.info("  Sampling with Temperature")
    logger.info("=" * 50)

    prompt = test_prompts[0]
    for temp in [0.3, 0.7, 1.2]:
        result = generator.generate(
            prompt,
            max_new_tokens=128,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
        )
        logger.info(f"\n--- Temperature={temp} ---")
        logger.info(f"A: {result[:200]}...")

    # ==================== 3. 批量推理 ====================
    logger.info("\n" + "=" * 50)
    logger.info("  Batch Inference")
    logger.info("=" * 50)

    start_time = time.time()
    batch_results = generator.batch_generate(
        test_prompts,
        batch_size=2,
        max_new_tokens=128,
        temperature=0.7,
    )
    elapsed = time.time() - start_time

    logger.info(
        f"Batch inference ({len(test_prompts)} prompts): {elapsed:.2f}s")
    for i, (prompt, result) in enumerate(zip(test_prompts, batch_results)):
        logger.info(f"\n  [{i+1}] Q: {prompt[:60]}...")
        logger.info(f"      A: {result[:150]}...")

    # ==================== 4. 流式输出 ====================
    logger.info("\n" + "=" * 50)
    logger.info("  Streaming Generation")
    logger.info("=" * 50)

    logger.info(f"Q: {test_prompts[2][:80]}")
    logger.info("A: ", )
    generator.generate_stream(
        test_prompts[2],
        max_new_tokens=200,
        temperature=0.7,
    )

    logger.info("\n" + "=" * 70)
    logger.info("  Inference Demo Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
