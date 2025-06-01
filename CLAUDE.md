# Ebbinghaus Memory-Enhanced LLM - Technical Design Document

## 🎯 System Overview

This document captures the optimized design of the Ebbinghaus memory-enhanced LLM system, which implements dynamic memory management based on the Ebbinghaus forgetting curve to achieve efficient text generation.

## 🔬 Core Principle: Ebbinghaus Forgetting Curve

The system is built on the Ebbinghaus forgetting curve formula:
```
R = e^(-t/S)
```
Where:
- **R**: Retention rate (memory weight) [0, 1]
- **t**: Time steps since last access (increments by 0.01 per token)
- **S**: Memory strength (starts at 5.0, increases with attention)

### Critical Implementation Rule
**Every generated token MUST update memory parameters**. This is non-negotiable - the scientific accuracy of the Ebbinghaus model depends on continuous time progression and attention-based memory updates.

## 🏗️ Architecture

### Three Generation Modes

1. **Baseline Mode** (`generate_baseline`)
   - Standard transformer generation using forward() 
   - No memory mechanisms applied
   - Serves as performance benchmark

2. **Soft Delete Mode** (`generate_soft_delete`)
   - Dynamically modifies KV cache values
   - Applies memory weights ∈ [0.8, 1.0] to key cache
   - Only affects first 50% of layers for efficiency

3. **Sparse Attention Mode** (`generate_sparse_attention`)
   - Dynamically generates attention masks
   - Masks positions when ≥50% of layers have weight < 0.001
   - True sparsity through attention masking

### Key Design Decisions

1. **Forward-based Generation**: All modes use `model.forward()` directly instead of `transformers.generate()` to enable per-token memory updates
2. **Per-token Updates**: Memory must be updated after EVERY token generation
3. **Cross-layer Aggregation**: 50% of layers must agree before masking (stability)
4. **Optimized but Correct**: Performance improvements through code optimization, NOT logic simplification

## 📊 Performance Characteristics

Achieved performance (after optimization):
- **Baseline**: 15.20 tokens/s
- **Soft Delete**: 16.11 tokens/s  
- **Sparse Attention**: 14.14 tokens/s

Performance improvements of 50-77% through:
- Vectorized operations
- NumPy array pre-allocation
- Batch processing
- Efficient caching with proper invalidation

## 🔧 Implementation Details

### Memory Manager (`memollm/memory.py`)

#### Key Components:
```python
class LayerTokenMemory:
    S: float = 5.0  # Memory strength
    t: int = 0      # Time steps
    
    def get_retention_weight(self) -> float:
        return math.exp(-self.t / self.S)
    
    def update_memory(self, attention_weight: float):
        self.S += attention_weight
        if attention_weight >= 0.01:
            self.t = 0  # Reset time for important tokens
    
    def step_time(self):
        self.t += 0.01  # Small increment per token
```

#### Optimizations:
- Pre-allocated NumPy arrays for weight calculations
- Batch processing of memory updates
- Lazy cleanup (every 100 steps) with 0.5% threshold
- Direct dictionary operations avoiding intermediate lists

### LLM Implementation (`memollm/llm.py`)

#### Critical Cache Management:
```python
# 清除缓存以获取最新的记忆权重
self._cached_memory_weights = None
```
**This line is ESSENTIAL** - it ensures fresh memory weights are computed for each token, maintaining the scientific integrity of the Ebbinghaus model.

#### Key Optimized Methods:

1. **Fast Sampling** (`_sample_next_token_fast`):
   - In-place logits modification
   - Efficient top-k/top-p filtering
   - Minimal tensor allocations

2. **Cached Memory Weights** (`_get_memory_weights_fast`):
   - Caches weights until sequence length changes
   - Cache MUST be cleared after time stepping

3. **Fast KV Modification** (`_apply_memory_to_past_kv_fast`):
   - In-place tensor operations
   - Only processes first 50% of layers
   - Clamps weights to [0.8, 1.0] range

4. **Vectorized Mask Generation** (`_generate_dynamic_attention_mask_fast`):
   - Stack all layer weights for batch processing
   - Single threshold comparison operation
   - Pre-allocated result tensors

### Generation Flow (Per Mode)

#### Soft Delete Generation:
1. Get current sequence length
2. Retrieve memory weights (with caching)
3. Apply weights to past_key_values (in-place)
4. Forward pass with attention output
5. Update all layer memories with attention
6. Sample next token
7. Step time for all memories
8. **Clear cache** ← Critical!
9. Repeat

#### Sparse Attention Generation:
1. Get current sequence length
2. Retrieve memory weights (with caching)
3. Generate dynamic attention mask
4. Forward pass with mask and attention output
5. Update all layer memories with attention
6. Sample next token
7. Step time for all memories
8. **Clear cache** ← Critical!
9. Repeat

## 🎯 Usage Patterns

### Basic Usage:
```python
from memollm import EbbinghausLLM

llm = EbbinghausLLM("Qwen/Qwen2.5-0.5B-Instruct")

# High-speed generation
result = llm.generate(
    "Explain AI",
    max_new_tokens=100,
    generation_mode="soft_delete",
    temperature=0.7,
    top_p=0.9
)

# Long-context with sparsity
result = llm.generate(
    "Write a detailed analysis",
    max_new_tokens=500,
    generation_mode="sparse_attention"
)
```

### Performance Testing:
```python
# Compare all three modes
for mode in ["baseline", "soft_delete", "sparse_attention"]:
    result = llm.generate(prompt, generation_mode=mode)
    speed = result['num_tokens'] / result['generation_time']
    print(f"{mode}: {speed:.2f} tokens/s")
```

### Fixed-Length Generation:
```python
# Generate exactly N tokens (ignore EOS)
result = llm.generate(
    "Tell a story",
    max_new_tokens=1000,
    generation_mode="sparse_attention",
    force_exact_length=True  # Continue generating even after EOS
)
```

### Command-line Usage:
```bash
# Standard usage
python demo.py --compare --max_tokens 1000

# Force exact length generation
python demo.py --compare --max_tokens 1000 --force_exact_length

# Single mode with fixed length
python demo.py --mode sparse_attention --max_tokens 1000 --force_exact_length
```

## 🚀 Optimization Principles

1. **Never Compromise Core Logic**: Every token updates memory
2. **Vectorize Everything**: NumPy/PyTorch batch operations
3. **Cache Wisely**: Cache immutable data, invalidate on changes
4. **Minimize Allocations**: In-place operations where possible
5. **Profile First**: Measure before optimizing

## 📝 Configuration Parameters

### Memory System:
- Initial memory strength (S): 5.0
- Time increment per token: 0.01
- Attention threshold for reset: 0.01
- Cleanup threshold: 0.005 (0.5%)
- Cleanup frequency: Every 100 steps

### Soft Delete:
- Weight range: [0.8, 1.0]
- Applied layers: First 50%

### Sparse Attention:
- Masking threshold: 0.001
- Cross-layer agreement: 50%

## 🔍 Debugging and Monitoring

### Key Metrics to Track:
1. Tokens per second
2. Average retention rates per layer
3. Number of masked positions (sparse attention)
4. Memory cleanup frequency

### Common Issues:
- Cache not cleared → Stale memory weights
- Wrong time increment → Too fast/slow forgetting
- Threshold too high → Excessive masking
- Missing attention outputs → No memory updates

## 📚 Previous Implementation Notes (Historical Context)

### Original Hook-based Approach (Deprecated)
The initial implementation used transformers.generate() with forward hooks:
- **Problem**: Dimension mismatches and complex KV cache management
- **Solution**: Moved to custom forward-based generation

### Evolution of Time Steps:
- V1: 0.1 per token (too fast)
- V2: 0.01 per token (optimal for token-level granularity)

### Performance Journey:
1. Initial: ~10 tokens/s (baseline slowest)
2. After fixing update logic: ~12 tokens/s
3. Final optimized: 15-16 tokens/s (50-77% improvement)

## 🆕 Recent Updates (January 2025)

### Force Exact Length Generation
Added `force_exact_length` parameter to all generation methods:
- **Purpose**: Generate exactly `max_new_tokens` tokens, ignoring EOS markers
- **Use case**: Fair performance comparisons, fixed-length evaluations
- **Implementation**: Modified EOS checking logic in `generate_baseline`, `generate_soft_delete`, and `generate_sparse_attention`
- **CLI support**: Added `--force_exact_length` flag to `demo.py`

### Memory System Update
- Time step adjusted from 0.01 to 0.05 per token to slow down forgetting rate
- This provides better long-context retention while maintaining the Ebbinghaus curve dynamics

## 📚 References

- Ebbinghaus, H. (1885). "Memory: A Contribution to Experimental Psychology"
- Implementation based on exponential decay model
- Optimizations inspired by FlashAttention principles

---

**Last Updated**: January 2025
**Version**: 2.1 (Added force_exact_length)
**Status**: Production-ready with verified performance improvements