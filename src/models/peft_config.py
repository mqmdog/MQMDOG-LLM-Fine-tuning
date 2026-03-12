"""
PEFT (Parameter-Efficient Fine-Tuning) 配置工厂
================================================
核心功能: 根据配置创建不同的参数高效微调策略

支持的微调方法:
1. LoRA: 低秩适配 - 在注意力层插入低秩矩阵 A 和 B
2. QLoRA: 量化LoRA - 在4bit量化基础上应用LoRA
3. AdaLoRA: 自适应LoRA - 动态调整不同层的秩
4. Prefix Tuning: 前缀调优 - 在每层添加可学习的前缀向量
5. Prompt Tuning: 提示调优 - 在输入层添加可学习的软提示
6. IA3: 通过抑制和放大内部激活实现微调



【LoRA原理】
- 核心思想: 预训练权重的更新矩阵是低秩的, 即 ΔW ≈ BA (B∈R^{d×r}, A∈R^{r×k}, r << min(d,k))
- 训练时: h = Wx + BAx, 其中W冻结, 只训练B和A
- 推理时: 可将BA合并到W中 (W' = W + BA), 无额外推理开销
- 初始化: A用高斯初始化, B初始化为零 → 训练开始时ΔW=0

【AdaLoRA原理】
- 问题: 不同层/模块的重要性不同, 固定秩不够灵活
- 方案: 使用SVD分解 ΔW = PΛQ, 通过重要性评分动态剪枝奇异值
- 优势: 自动将更多参数分配给重要的层

【Prefix Tuning原理】
- 在每个Transformer层的Key和Value前面添加可训练的前缀向量
- 通过MLP重参数化 (prefix_projection) 提高训练稳定性
- 与Prompt Tuning区别: Prefix Tuning作用于每层, Prompt Tuning只作用于输入层

【Prompt Tuning原理】
- 在输入embedding前添加可学习的软提示 (soft prompt)
- 参数量极少 (仅 num_virtual_tokens × hidden_size)
- 随模型规模增大, 效果接近全参数微调 (论文中10B以上模型)
"""

import logging
from typing import Dict, Any, Optional

from peft import (
    LoraConfig,
    AdaLoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# 任务类型映射
TASK_TYPE_MAP = {
    "CAUSAL_LM": TaskType.CAUSAL_LM,
    "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    "TOKEN_CLS": TaskType.TOKEN_CLS,
    "SEQ_CLS": TaskType.SEQ_CLS,
}


class PeftConfigFactory:
    """
    PEFT配置工厂 - 根据配置文件创建对应的PEFT配置并应用到模型

    设计模式: 工厂模式 (Factory Pattern)
    - 封装不同PEFT方法的创建逻辑
    - 统一接口, 通过method字符串选择具体实现

    使用示例:
        factory = PeftConfigFactory(config)
        model = factory.apply_peft(model)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.finetuning_config = config.get("finetuning", {})
        self.method = self.finetuning_config.get("method", "lora")

    def _get_task_type(self, method_config: Dict) -> TaskType:
        """获取PEFT任务类型"""
        task_str = method_config.get("task_type", "CAUSAL_LM")
        return TASK_TYPE_MAP.get(task_str, TaskType.CAUSAL_LM)

    def create_lora_config(self) -> LoraConfig:
        """
        创建LoRA配置

        关键参数解释:
        - r (秩): LoRA矩阵的秩, 越大表达能力越强但参数越多. 经验值: 8-64
        - lora_alpha: 缩放因子, 实际缩放比例为 alpha/r. 通常设为2*r
        - target_modules: 应用LoRA的模块, 通常选择注意力层的QKV投影
          - 更多模块 → 更强的适应能力, 但参数更多
          - 实践中对q_proj和v_proj应用LoRA效果最好 (原论文结论)
        - lora_dropout: 正则化, 防止过拟合
        """
        lora_cfg = self.finetuning_config.get("lora", {})

        config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
            bias=lora_cfg.get("bias", "none"),
            task_type=self._get_task_type(lora_cfg),
        )

        logger.info(
            f"LoRA config: r={config.r}, alpha={config.lora_alpha}, "
            f"dropout={config.lora_dropout}, targets={config.target_modules}"
        )
        return config

    def create_qlora_config(self) -> LoraConfig:
        """
        创建QLoRA配置

        QLoRA = 4bit量化模型 + LoRA适配器
        量化部分在模型加载时处理 (ModelLoader), 这里只创建LoRA配置


        - QLoRA的量化是在模型加载时完成的, 不是在PEFT配置中
        - QLoRA训练时, 前向传播用4bit, 反向传播梯度用BF16
        - LoRA参数始终以高精度 (BF16/FP16) 存储和更新
        """
        qlora_cfg = self.finetuning_config.get("qlora", {})

        config = LoraConfig(
            r=qlora_cfg.get("r", 16),
            lora_alpha=qlora_cfg.get("lora_alpha", 32),
            lora_dropout=qlora_cfg.get("lora_dropout", 0.05),
            target_modules=qlora_cfg.get("target_modules"),
            bias=qlora_cfg.get("bias", "none"),
            task_type=self._get_task_type(qlora_cfg),
        )

        logger.info(
            f"QLoRA config: r={config.r}, alpha={config.lora_alpha}, "
            f"bits={qlora_cfg.get('bits', 4)}, quant={qlora_cfg.get('quant_type', 'nf4')}"
        )
        return config

    def create_adalora_config(self) -> AdaLoraConfig:
        """
        创建AdaLoRA配置

        AdaLoRA特有参数:
        - init_r: 初始秩 (比目标秩大, 训练过程中会自动裁剪)
        - target_r: 目标秩 (最终每个模块的平均秩)
        - tinit/tfinal: 秩调整的起止步数
        - deltaT: 秩调整间隔
        """
        adalora_cfg = self.finetuning_config.get("adalora", {})

        config = AdaLoraConfig(
            init_r=adalora_cfg.get("init_r", 12),
            target_r=adalora_cfg.get("target_r", 8),
            beta1=adalora_cfg.get("beta1", 0.85),
            beta2=adalora_cfg.get("beta2", 0.85),
            tinit=adalora_cfg.get("tinit", 200),
            tfinal=adalora_cfg.get("tfinal", 1000),
            deltaT=adalora_cfg.get("deltaT", 10),
            lora_alpha=adalora_cfg.get("lora_alpha", 32),
            lora_dropout=adalora_cfg.get("lora_dropout", 0.05),
            target_modules=adalora_cfg.get("target_modules"),
            task_type=self._get_task_type(adalora_cfg),
        )

        logger.info(
            f"AdaLoRA config: init_r={config.init_r}, target_r={config.target_r}"
        )
        return config

    def create_prefix_tuning_config(self) -> PrefixTuningConfig:
        """
        创建Prefix Tuning配置

        原理:
        - 在每个Transformer层的Key和Value张量前拼接可学习的前缀向量
        - P_k, P_v ∈ R^{l×d}, l为前缀长度, d为隐藏维度
        - 通过MLP重参数化: P = MLP(P'), P'是低维参数, MLP将其映射到高维
          这样做提高训练稳定性, 训练完成后丢弃MLP
        """
        prefix_cfg = self.finetuning_config.get("prefix_tuning", {})

        config = PrefixTuningConfig(
            num_virtual_tokens=prefix_cfg.get("num_virtual_tokens", 20),
            prefix_projection=prefix_cfg.get("prefix_projection", True),
            task_type=self._get_task_type(prefix_cfg),
        )

        logger.info(
            f"Prefix Tuning config: num_virtual_tokens={config.num_virtual_tokens}, "
            f"prefix_projection={config.prefix_projection}"
        )
        return config

    def create_prompt_tuning_config(self) -> PromptTuningConfig:
        """
        创建Prompt Tuning配置

        与Prefix Tuning的区别:
        - Prompt Tuning: 只在输入embedding层添加软提示
        - Prefix Tuning: 在每一层都添加前缀
        - 参数量: Prompt Tuning << Prefix Tuning
        - 效果: 小模型上Prompt Tuning较弱, 大模型(>10B)上接近全参数微调
        """
        prompt_cfg = self.finetuning_config.get("prompt_tuning", {})

        init_type = prompt_cfg.get("prompt_tuning_init", "TEXT")
        init_enum = (
            PromptTuningInit.TEXT
            if init_type == "TEXT"
            else PromptTuningInit.RANDOM
        )

        config_kwargs = {
            "num_virtual_tokens": prompt_cfg.get("num_virtual_tokens", 20),
            "prompt_tuning_init": init_enum,
            "task_type": self._get_task_type(prompt_cfg),
        }

        if init_enum == PromptTuningInit.TEXT:
            config_kwargs["prompt_tuning_init_text"] = prompt_cfg.get(
                "prompt_tuning_init_text",
                "Classify if the text is positive or negative:",
            )
            # tokenizer_name_or_path 在应用时设置

        config = PromptTuningConfig(**config_kwargs)

        logger.info(
            f"Prompt Tuning config: num_virtual_tokens={config.num_virtual_tokens}, "
            f"init={init_type}"
        )
        return config

    def create_peft_config(self):
        """
        根据配置创建对应的PEFT配置对象

        工厂方法: 通过method字符串分发到具体的创建方法
        """
        method_map = {
            "lora": self.create_lora_config,
            "qlora": self.create_qlora_config,
            "adalora": self.create_adalora_config,
            "prefix_tuning": self.create_prefix_tuning_config,
            "prompt_tuning": self.create_prompt_tuning_config,
        }

        if self.method == "full":
            logger.info("Full fine-tuning mode: no PEFT config needed")
            return None

        creator = method_map.get(self.method)
        if creator is None:
            raise ValueError(
                f"Unknown PEFT method '{self.method}'. "
                f"Supported: {list(method_map.keys()) + ['full']}"
            )

        return creator()

    def apply_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        将PEFT配置应用到模型

        流程:
        1. 如果是全参数微调, 直接返回原模型
        2. 如果是QLoRA, 先调用prepare_model_for_kbit_training
        3. 创建PEFT配置并包装模型
        4. 打印可训练参数统计

        Args:
            model: 原始预训练模型

        Returns:
            包装后的PEFT模型 (或全参数微调时返回原模型)
        """
        if self.method == "full":
            logger.info("Full fine-tuning: all parameters are trainable")
            return model

        # QLoRA: 需要先准备量化模型以进行训练
        # 这一步会: (1)将LayerNorm转为FP32 (2)设置output layer为FP32 (3)启用梯度检查点
        if self.method == "qlora":
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
            logger.info("Model prepared for k-bit training (QLoRA)")

        # 创建PEFT配置
        peft_config = self.create_peft_config()
        if peft_config is None:
            return model

        # 应用PEFT
        model = get_peft_model(model, peft_config)

        # 打印参数统计
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        ratio = trainable_params / total_params * 100

        logger.info(
            f"PEFT applied [{self.method}]: "
            f"trainable={trainable_params:,} / total={total_params:,} "
            f"({ratio:.4f}%)"
        )

        return model
