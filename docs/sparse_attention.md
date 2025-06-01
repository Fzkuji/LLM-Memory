# 稀疏注意力机制实现

## 概述

本项目实现了三种不同的生成模式，用于比较记忆增强机制的效果：

1. **Baseline模式**：标准Transformer生成，不使用任何记忆机制
2. **Soft Delete模式**：使用权重调整K值，软性降低某些位置的影响
3. **Sparse Attention模式**：使用attention mask直接屏蔽低保持率的位置

## 实现细节

### 1. 生成模式参数

在 `EbbinghausLLM.generate()` 方法中添加了 `generation_mode` 参数：

```python
result = llm.generate(
    prompt,
    max_new_tokens=100,
    generation_mode="sparse_attention",  # 可选: "baseline", "soft_delete", "sparse_attention"
    enable_memory=True,  # baseline模式会自动设为False
    return_attention_weights=True
)
```

### 2. Sparse Attention实现

#### 2.1 Attention Mask生成

```python
def create_layer_specific_attention_mask(self, layer_idx, weights, seq_len_q, seq_len_kv, num_heads, threshold=0.01):
    """为特定层创建attention mask"""
    # 创建基础mask
    mask = torch.ones(1, num_heads, seq_len_q, seq_len_kv, device=weights.device, dtype=weights.dtype)
    
    # 应用阈值：保持率低于threshold的位置设为0
    if weights.shape[0] == seq_len_kv:
        mask_values = (weights >= threshold).float()
        mask = mask_values.view(1, 1, 1, -1).expand(1, num_heads, seq_len_q, -1)
        
        # 转换为注意力分数mask (0 -> -inf, 1 -> 0)
        attention_mask = mask.masked_fill(mask == 0, -float('inf'))
        attention_mask = attention_mask.masked_fill(mask == 1, 0.0)
    
    return attention_mask
```

#### 2.2 每层独立的Mask

- 每一层都有自己独立的attention mask
- 基于该层的记忆保持率计算mask
- 保持率 < 0.01 的位置被完全屏蔽

### 3. 三种模式的区别

| 模式 | K值处理 | Attention Mask | 记忆机制 | 性能特点 |
|------|---------|----------------|----------|----------|
| Baseline | 不变 | 无 | 无 | 最快，无记忆 |
| Soft Delete | 权重调整 | 无 | 有 | 平滑衰减 |
| Sparse Attention | 不变 | 有 | 有 | 真正稀疏 |

## 使用示例

### 1. 基础使用

```python
from memollm.llm import EbbinghausLLM

# 初始化模型
llm = EbbinghausLLM(model_name="Qwen/Qwen2.5-0.5B-Instruct")

# 使用不同模式生成
prompt = "请解释什么是深度学习"

# Baseline模式
result_baseline = llm.generate(prompt, generation_mode="baseline")

# Soft Delete模式（默认）
result_soft = llm.generate(prompt, generation_mode="soft_delete")

# Sparse Attention模式
result_sparse = llm.generate(prompt, generation_mode="sparse_attention")
```

### 2. 性能测试

运行性能测试脚本：

```bash
python scripts/test_sparse_attention.py
```

输出示例：
```
测试模式: baseline
生成时间: 2.34秒
生成速度: 42.7 tokens/秒

测试模式: soft_delete
生成时间: 2.67秒
生成速度: 37.5 tokens/秒
记忆统计：
- 记忆token数: 85
- 平均保持率: 0.6234

测试模式: sparse_attention
生成时间: 2.51秒
生成速度: 39.8 tokens/秒
稀疏率: 42.3%
```

### 3. 稀疏性分析

运行详细分析脚本：

```bash
python scripts/sparse_attention_analysis.py
```

该脚本会：
- 可视化各层的注意力稀疏性
- 测量实际的mask比例
- 比较不同模式的生成质量
- 生成稀疏性变化图表

### 4. 简单演示

运行演示脚本查看三种模式的效果：

```bash
python scripts/demo_sparse_attention.py
```

## 性能优化建议

1. **阈值调整**：默认阈值为0.01，可以根据需要调整以控制稀疏程度
2. **层级差异**：不同层可以使用不同的阈值
3. **动态稀疏**：可以根据生成长度动态调整稀疏率

## 注意事项

1. Sparse Attention模式需要额外的mask计算，会有轻微的性能开销
2. 过高的稀疏率可能影响生成质量
3. 当前实现使用简化的全局mask，未来可以实现真正的per-layer mask

## 未来改进

1. 实现真正的per-layer attention mask传递
2. 优化mask计算以减少内存使用
3. 添加动态阈值调整机制
4. 支持自定义稀疏模式（如块稀疏、局部稀疏等）