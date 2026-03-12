# LLM Fine-tuning: 大语言模型微调框架

一个完整的大语言模型微调项目，集成多种主流微调方法，覆盖从数据处理、模型训练到评估部署的全流程。

## 项目特性

- **多种微调方法**: Full Fine-tuning、LoRA、QLoRA、AdaLoRA、Prefix Tuning、Prompt Tuning
- **对齐训练**: DPO (Direct Preference Optimization) 实现
- **知识蒸馏**: Teacher-Student蒸馏框架
- **多场景覆盖**: 对话系统、代码生成、文本分类、情感分析、知识蒸馏
- **工程化设计**: 配置管理、日志监控、评估框架、推理引擎

## 项目结构

```
LLM_Fine-tuning/
├── configs/                          # 配置文件
│   ├── base_config.yaml              # 基础默认配置
│   ├── lora_config.yaml              # LoRA微调配置
│   ├── qlora_config.yaml             # QLoRA微调配置
│   ├── full_finetune_config.yaml     # 全参数微调配置
│   └── dpo_config.yaml               # DPO对齐训练配置
├── src/                              # 核心源码
│   ├── data/                         # 数据处理模块
│   │   ├── data_loader.py            # 数据加载与预处理
│   │   ├── data_template.py          # 数据格式模板 (Alpaca/ShareGPT/DPO)
│   │   └── data_collator.py          # 动态批处理与填充
│   ├── models/                       # 模型模块
│   │   ├── model_loader.py           # 模型加载器 (支持量化加载)
│   │   └── peft_config.py            # PEFT配置工厂
│   ├── training/                     # 训练模块
│   │   ├── trainer.py                # SFT训练器
│   │   └── dpo_trainer.py            # DPO对齐训练器
│   ├── evaluation/                   # 评估模块
│   │   ├── evaluator.py              # 模型评估器
│   │   └── metrics.py                # 评估指标 (PPL/BLEU/ROUGE/F1)
│   ├── inference/                    # 推理模块
│   │   └── generator.py              # 文本生成器 (多种解码策略)
│   └── utils/                        # 工具模块
│       ├── config_parser.py          # 配置解析与合并
│       ├── logger.py                 # 日志管理
│       └── common.py                 # 通用工具函数
├── examples/                         # 应用示例
│   ├── 01_lora_chat.py               # LoRA微调对话模型
│   ├── 02_qlora_efficient.py         # QLoRA资源高效微调
│   ├── 03_dpo_alignment.py           # DPO对齐训练
│   ├── 04_knowledge_distillation.py  # 知识蒸馏
│   ├── 05_text_classification.py     # 文本分类/情感分析
│   └── 06_inference_demo.py          # 推理与部署示例
├── scripts/                          # 脚本
├── tests/                            # 测试
├── data/                             # 数据目录
├── outputs/                          # 输出目录
├── run_train.py                      # 统一训练入口
├── requirements.txt                  # 项目依赖
└── README.md
```

## 环境要求

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8 (GPU训练)
- 显存: LoRA >= 8GB, QLoRA >= 6GB, Full >= 16GB (以0.5B模型为例)

## 安装

```bash
# 克隆项目
git clone https://github.com/mqmdog/MQMDOG-LLM-Fine-tuning.git
cd MQMDOG-LLM-Fine-tuning

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. LoRA微调

```bash
python run_train.py --config configs/lora_config.yaml
```

### 2. QLoRA微调 (低显存)

```bash
python run_train.py --config configs/qlora_config.yaml
```

### 3. 全参数微调

```bash
python run_train.py --config configs/full_finetune_config.yaml
```

### 4. DPO对齐训练

```bash
python run_train.py --config configs/dpo_config.yaml --mode dpo
```

### 5. 命令行覆盖参数

```bash
python run_train.py --config configs/lora_config.yaml \
    --override training.learning_rate=1e-4 \
    --override training.num_epochs=5 \
    --override finetuning.lora.r=32
```

### 6. 运行示例

```bash
# LoRA对话微调
python examples/01_lora_chat.py

# 知识蒸馏
python examples/04_knowledge_distillation.py

# 文本分类
python examples/05_text_classification.py

# 推理演示 (使用基座模型)
python examples/06_inference_demo.py --use_base_model
```

## 核心技术说明

### 微调方法对比

| 方法             | 可训练参数比例 | 显存需求 | 训练速度 | 效果         |
| ---------------- | -------------- | -------- | -------- | ------------ |
| Full Fine-tuning | 100%           | 最高     | 慢       | 最好         |
| LoRA             | ~0.1-1%        | 低       | 快       | 接近Full     |
| QLoRA            | ~0.1-1%        | 最低     | 中等     | 接近LoRA     |
| AdaLoRA          | ~0.1-1%        | 低       | 中等     | 优于LoRA     |
| Prefix Tuning    | <0.1%          | 低       | 快       | 中等         |
| Prompt Tuning    | <0.01%         | 最低     | 最快     | 依赖模型规模 |

### LoRA核心原理

```
原始: h = Wx
LoRA: h = Wx + BAx  (B∈R^{d×r}, A∈R^{r×k}, r << min(d,k))

- W: 冻结的预训练权重
- B, A: 可训练的低秩矩阵
- 推理时: W' = W + BA (合并后无额外开销)
```

### QLoRA核心技术

1. **NF4量化**: 基于权重正态分布假设的信息论最优4bit量化
2. **双重量化**: 对量化常数本身再量化, 额外节省~0.37bit/param
3. **分页优化器**: GPU OOM时自动将优化器状态卸载到CPU

### DPO对齐原理

```
L_DPO = -log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

相比RLHF: 无需训练奖励模型, 无需PPO, 更简单稳定
```

## 配置说明

配置采用分层设计，支持继承和覆盖：

```
base_config.yaml (默认值)
    ↓ 继承 + 覆盖
task_config.yaml (任务特定)
    ↓ 覆盖
CLI参数 (最高优先级)
```

关键配置项:

```yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-0.5B-Instruct"  # 模型
  torch_dtype: "bf16"                                   # 精度

finetuning:
  method: "lora"         # 微调方法
  lora:
    r: 16                # LoRA秩
    lora_alpha: 32       # 缩放因子
    target_modules: ["q_proj", "v_proj"]  # 目标模块

training:
  learning_rate: 2.0e-4  # 学习率
  num_epochs: 3          # 训练轮数
  gradient_checkpointing: true  # 梯度检查点
```

## 重点


1. **Transformer架构**: 自注意力机制、位置编码、前馈网络
2. **微调策略**: LoRA数学原理、QLoRA量化技术、参数高效方法对比
3. **训练优化**: 学习率调度、梯度裁剪、混合精度、梯度累积
4. **数据工程**: 数据模板设计、Label Masking、动态Padding
5. **对齐技术**: DPO损失函数推导、与RLHF的对比
6. **推理优化**: KV Cache、解码策略、模型量化
7. **知识蒸馏**: 软标签/硬标签、温度参数作用
8. **工程实践**: 配置管理、日志系统、评估指标
