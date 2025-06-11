# Ebbinghaus Memory-Enhanced LLM - 核心设计文档

## 🎯 核心思路

通过艾宾浩斯遗忘曲线为每个token在每一层建立独立的记忆模型，实现真正的动态KV cache管理。

**设计理念**: 模拟人类记忆的自然衰减过程，当token没有被关注时，可以从相应层的cache中删除，不同层可以有不同的cache长度。

## 🔬 艾宾浩斯遗忘曲线原理

系统基于艾宾浩斯遗忘曲线公式：
```
R = e^(-t/S)
```
其中：
- **R**: 记忆保持率（权重）[0, 1]
- **t**: 时间步数（每token增加0.05）
- **S**: 记忆强度（初始5.0，随attention增强）

### 关键实现原则
**每生成一个token都必须更新记忆参数**。这是核心要求 - 艾宾浩斯模型的科学准确性依赖于持续的时间推进和基于attention的记忆更新。

## 🏗️ 系统架构

### 两种生成模式

1. **Baseline模式** (`generate_baseline`)
   - 标准transformer生成，使用forward()
   - 不应用任何记忆机制
   - 作为性能基准

2. **Memory Enhanced模式** (`generate_memory_enhanced`)
   - 使用自然记忆权重（0-1范围），无人为限制
   - 当权重低于阈值时执行硬删除（默认0.01）
   - 支持每层不同长度的cache
   - 删除token后停止更新其记忆
   - 无人为最小权重约束

### 关键设计决策

1. **基于Forward的生成**: 直接使用`model.forward()`而不是`transformers.generate()`，以实现每token的记忆更新
2. **每token更新**: 每生成一个token后都必须更新记忆
3. **层间独立**: 每层独立决定是否删除token，支持不同层的不同cache长度
4. **真正的硬删除**: 使用VariableLengthCache实现真正的token删除，而不是软屏蔽

## 📊 性能特征

- **Baseline**: ~18 tokens/s（标准生成）
- **Memory Enhanced**: ~5-6 tokens/s（动态cache管理）

Memory Enhanced模式虽然速度较慢，但在长文本生成时能够：
- 自动删除低权重token，释放内存
- 支持更长的上下文窗口
- Cache删除率通常在20-40%

## 🔧 核心实现

### 记忆管理器 (`memollm/memory.py`)

```python
class LayerTokenMemory:
    S: float = 5.0  # 记忆强度
    t: int = 0      # 时间步数
    
    def get_retention_weight(self) -> float:
        return math.exp(-self.t / self.S)
    
    def update_memory(self, attention_weight: float):
        self.S += attention_weight  # 增强记忆
        if attention_weight >= 0.01:
            self.t = 0  # 重要token重置时间
    
    def step_time(self):
        self.t += 0.05  # 每token时间推进
```

### 变长Cache实现 (`memollm/model.py`)

```python
class VariableLengthCache:
    def delete_tokens(self, layer_idx: int, positions_to_delete: List[int]):
        """删除指定层的特定token位置"""
        # 为指定层创建保留mask
        keep_mask = torch.ones(seq_len, dtype=torch.bool)
        for pos in positions_to_delete:
            keep_mask[pos] = False
        
        # 应用mask到该层cache
        self.key_cache[layer_idx] = key_cache[:, :, keep_mask, :]
        self.value_cache[layer_idx] = value_cache[:, :, keep_mask, :]
```

### Memory Enhanced生成流程:
1. 获取当前序列长度
2. 检索记忆权重（带缓存）
3. 对每层应用记忆权重
4. 识别需要删除的token（低于阈值）
5. **每层独立删除token**从KV cache
6. Forward传播（使用修改后的cache）
7. 更新所有层记忆（跳过已删除token）
8. 采样下一个token
9. 所有记忆时间步进
10. **清除缓存** ← 关键！
11. 重复

## 🎯 基本使用

```python
from memollm import EbbinghausLLM

llm = EbbinghausLLM("Qwen/Qwen2.5-0.5B-Instruct")

# 标准生成（基准）
result = llm.generate(
    "解释人工智能",
    max_new_tokens=100,
    generation_mode="baseline"
)

# 记忆增强生成（支持每层不同长度cache）
result = llm.generate(
    "写一个详细分析",
    max_new_tokens=500,
    generation_mode="memory_enhanced",
    hard_delete_threshold=0.01  # 硬删除阈值
)

print(f"Cache删除率: {result['cache_deletion_percentage']:.2f}%")
print(f"每层cache长度: {result['layer_cache_lengths'][:5]}")
```

## 📝 核心配置

### 记忆系统参数:
- 初始记忆强度(S): 5.0
- 每token时间增量: 0.05
- Attention重置阈值: 0.01
- 硬删除阈值: 0.01（可配置）

### 关键特性:
- 记忆权重自然范围: [0, 1]
- 无人为限制或最小阈值
- 每层独立删除决策
- 支持不同层的不同cache长度

---

**核心思想**: 通过艾宾浩斯遗忘曲线为每个token在每一层建立独立记忆模型，当token没有被关注时可以随时删除，实现真正的动态KV cache管理。

**最后更新**: 2025年1月
**版本**: 3.0 (Variable-Length Cache)