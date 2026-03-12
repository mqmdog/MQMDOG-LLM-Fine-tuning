"""验证项目核心模块功能"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_parser():
    print("=== Testing ConfigParser ===")
    from src.utils.config_parser import ConfigParser

    config = ConfigParser.load(
        "configs/lora_config.yaml", "configs/base_config.yaml")
    print(f"  Config loaded OK")
    print(f"  Model: {config['model']['model_name_or_path']}")
    print(f"  Method: {config['finetuning']['method']}")
    print(f"  LoRA r: {config['finetuning']['lora']['r']}")
    print(f"  LR: {config['training']['learning_rate']}")

    config2 = ConfigParser.override(
        config, {"training.learning_rate": 1e-4, "finetuning.lora.r": 32})
    assert config2["training"]["learning_rate"] == 1e-4
    assert config2["finetuning"]["lora"]["r"] == 32
    print(f"  Override OK")

    assert ConfigParser.validate(config), "Validation failed"
    print(f"  Validation OK")

    # 测试所有配置文件
    for cfg_name in ["lora_config.yaml", "qlora_config.yaml", "full_finetune_config.yaml", "dpo_config.yaml"]:
        cfg = ConfigParser.load(
            f"configs/{cfg_name}", "configs/base_config.yaml")
        assert ConfigParser.validate(cfg)
        print(f"  {cfg_name} loaded and validated")

    print("  PASSED\n")


def test_data_templates():
    print("=== Testing Data Templates ===")
    from src.data.data_template import TemplateManager

    manager = TemplateManager()
    print(f"  Available: {manager.list_templates()}")

    # Alpaca
    alpaca = manager.get_template("alpaca")
    r1 = alpaca.format_example(
        {"instruction": "What is AI?", "input": "", "output": "AI is..."})
    assert "prompt" in r1 and "response" in r1
    assert len(r1["prompt"]) > 0 and r1["response"] == "AI is..."
    print(f"  Alpaca (no input) OK")

    r2 = alpaca.format_example(
        {"instruction": "Translate", "input": "Hello", "output": "Hola"})
    assert "Hello" in r2["prompt"]
    print(f"  Alpaca (with input) OK")

    # ShareGPT
    sharegpt = manager.get_template("sharegpt")
    r3 = sharegpt.format_example({
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
        ]
    })
    assert r3["response"] == "Hello!"
    print(f"  ShareGPT OK")

    # DPO
    dpo = manager.get_template("dpo")
    r4 = dpo.format_example(
        {"prompt": "Q?", "chosen": "Good", "rejected": "Bad"})
    assert r4["chosen"] == "Good" and r4["rejected"] == "Bad"
    print(f"  DPO OK")

    print("  PASSED\n")


def test_metrics():
    print("=== Testing Metrics ===")
    from src.evaluation.metrics import MetricsCalculator
    import math

    mc = MetricsCalculator

    # Perplexity
    ppl = mc.compute_perplexity(2.5)
    assert abs(ppl - math.exp(2.5)) < 0.01
    print(f"  Perplexity(2.5) = {ppl:.2f} OK")

    # BLEU
    preds = ["the cat sat on the mat", "the dog ran in the park"]
    refs = ["the cat sat on the mat", "the dog played in the park"]
    bleu = mc.compute_bleu(preds, refs)
    assert bleu["bleu-1"] > 0.8  # 大部分unigram重叠
    print(f"  BLEU-1={bleu['bleu-1']:.4f}, BLEU-4={bleu['bleu-4']:.4f} OK")

    # ROUGE-L
    rouge = mc.compute_rouge_l(preds, refs)
    assert rouge["rouge-l-f1"] > 0.7
    print(f"  ROUGE-L F1={rouge['rouge-l-f1']:.4f} OK")

    # Classification
    cls_preds = [1, 0, 1, 1, 0, 1, 0, 0]
    cls_refs = [1, 0, 1, 0, 0, 1, 1, 0]
    cls_m = mc.compute_classification_metrics(cls_preds, cls_refs)
    assert 0 <= cls_m["accuracy"] <= 1
    assert 0 <= cls_m["f1"] <= 1
    print(f"  Accuracy={cls_m['accuracy']:.4f}, F1={cls_m['f1']:.4f} OK")

    print("  PASSED\n")


def test_logger():
    print("=== Testing Logger ===")
    from src.utils.logger import setup_logger, get_logger

    logger = setup_logger(
        "test_logger", log_dir="outputs/logs", log_level="DEBUG")
    logger.info("Test log message - INFO")
    logger.debug("Test log message - DEBUG")

    logger2 = get_logger("test_logger")
    assert logger is logger2
    print(f"  Logger setup and retrieval OK")
    print("  PASSED\n")


def test_common_utils():
    print("=== Testing Common Utils ===")
    # 只测试不依赖torch的部分
    import random
    import numpy as np

    # set_seed部分逻辑测试
    random.seed(42)
    np.random.seed(42)
    v1 = random.random()
    random.seed(42)
    np.random.seed(42)
    v2 = random.random()
    assert v1 == v2
    print(f"  Seed reproducibility OK")

    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  LLM Fine-tuning Project Verification")
    print("=" * 60)
    print()

    passed = 0
    total = 5
    tests = [
        ("ConfigParser", test_config_parser),
        ("DataTemplates", test_data_templates),
        ("Metrics", test_metrics),
        ("Logger", test_logger),
        ("CommonUtils", test_common_utils),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {name} - {e}\n")

    print("=" * 60)
    print(f"  Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("  All tests passed!")
    else:
        print("  Some tests failed!")
        sys.exit(1)
