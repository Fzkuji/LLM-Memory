# Ebbinghaus记忆增强LLM

## 🎯 核心思路

基于艾宾浩斯记忆遗忘曲线，为每个token在每一层建立独立的记忆模型。当token没有被其他token关注时，可以随时从KV cache中删除，实现真正的动态内存管理。

**关键设计原则：**
- 每个token在每一层都有独立的记忆状态（时间步、记忆强度）
- 基于attention权重动态更新记忆强度
- 记忆权重自然衰减，无人为干预
- 支持每层不同长度的KV cache（层间独立删除）

## 🚀 功能特点

- **Baseline模式**: 标准Transformer生成（作为对照基准）
- **Memory Enhanced模式**: 基于艾宾浩斯曲线的动态KV cache管理，支持每层独立的硬删除

## 📊 核心原理

### 艾宾浩斯记忆公式
```
R = e^(-t/S)
```
- **R**: 记忆保持率 [0,1]
- **t**: 时间步数（每token +0.05）  
- **S**: 记忆强度（初始5.0，随attention增强）

### 工作流程
1. **独立记忆建模**: 每个token在每层都有独立的记忆状态（时间步t、记忆强度S）
2. **动态更新**: 每次生成新token时，根据attention权重更新所有token的记忆强度
3. **自然衰减**: 记忆权重按艾宾浩斯公式自然衰减（0到1），无人为约束
4. **层间独立删除**: 每层可以独立判断并删除低权重token，实现不同层的不同cache长度
5. **记忆同步**: 删除token后，相应层的记忆管理器也同步更新位置映射

## 🔧 安装

```bash
pip install torch transformers
```

## 📖 使用方法

### Python API

```python
from memollm import EbbinghausLLM

# 初始化模型
llm = EbbinghausLLM("Qwen/Qwen2.5-0.5B-Instruct")

# Baseline模式（最快）
result = llm.generate(
    "请解释什么是人工智能",
    max_new_tokens=100,
    generation_mode="baseline"
)

# Memory Enhanced模式（内存优化）
result = llm.generate(
    "请解释什么是人工智能",
    max_new_tokens=100,
    generation_mode="memory_enhanced",
    return_attention_weights=True  # 可选：返回注意力分析
)

# 查看结果
print(f"生成文本: {result['generated_text']}")
print(f"速度: {result['num_tokens']/result['generation_time']:.2f} tokens/秒")

# Memory Enhanced模式额外信息
if result.get('cache_deletion_percentage'):
    print(f"Cache删除率: {result['cache_deletion_percentage']:.2f}%")
```

### 命令行使用

```bash
# 单模式生成
python demo.py --mode baseline --prompt "什么是深度学习？"
python demo.py --mode memory_enhanced --prompt "什么是深度学习？" --verbose

# 对比两种模式
python demo.py --compare --max_tokens 100

# 长文本测试（查看cache管理效果）
python demo.py --mode memory_enhanced --max_tokens 200 --verbose

# 固定长度生成（用于性能测试）
python demo.py --mode memory_enhanced --max_tokens 200 --force_exact_length
```

### 参数说明

- `--mode`: 生成模式 (`baseline`, `memory_enhanced`)
- `--prompt`: 输入提示词
- `--max_tokens`: 最大生成token数（默认50）
- `--model`: 模型名称（默认"Qwen/Qwen2.5-0.5B-Instruct"）
- `--compare`: 对比两种模式
- `--verbose`: 显示详细信息
- `--force_exact_length`: 强制生成固定长度（忽略EOS）

## 📊 性能特点

| 模式 | 速度 | 内存效率 | 适用场景 |
|------|------|----------|----------|
| Baseline | ~18 tokens/秒 | 标准 | 短文本、高速生成 |
| Memory Enhanced | ~5-6 tokens/秒 | 动态优化 | 长文本、内存受限 |

Memory Enhanced模式在长文本生成时的优势：
- 自动删除低权重token，释放内存
- 支持更长的上下文窗口
- Cache删除率通常在20-40%（取决于文本）

## 🛠️ 高级配置

```python
# 自定义硬删除阈值
result = llm.generate(
    "长文本输入...",
    generation_mode="memory_enhanced",
    hard_delete_threshold=0.01  # 更严格的删除标准
)
```

## 📁 项目结构

```
LLM-Memory/
├── memollm/              # 核心模块
│   ├── llm.py           # 主要LLM实现
│   ├── memory.py        # 艾宾浩斯记忆管理器
│   └── __init__.py      
├── demo.py              # 演示脚本
├── CLAUDE.md            # 技术文档
└── README.md            
```

## 🔍 技术实现

### 核心组件

1. **VariableLengthCache**: 支持每层不同长度的KV缓存
   - 每层独立的key/value tensor存储
   - 支持按位置删除特定token
   - 支持应用记忆权重到特定层

2. **EbbinghausMemoryManager**: 艾宾浩斯记忆管理器
   - 为每个token-layer组合维护独立记忆状态
   - 动态更新记忆强度和时间步
   - 计算基于遗忘曲线的权重

3. **VariableLengthModel**: 模型包装器
   - 透明包装现有transformer模型
   - 提供层级cache操作接口
   - 自动处理cache长度差异

### 删除策略

每层独立根据记忆权重决定是否删除token：
- 权重 < 阈值（默认0.01）→ 硬删除
- 删除后自动调整记忆管理器的位置映射
- 不同层可以有完全不同的cache长度

详细技术文档请参考 [CLAUDE.md](CLAUDE.md)

## 📝 许可证

MIT License