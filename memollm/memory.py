import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class LayerTokenMemory:
    """每层每个token的记忆信息"""
    S: float = 5.0  # 记忆强度 (memory strength) - 增强默认值
    t: int = 0      # 时间步 (time steps since last access)
    
    def get_retention_weight(self) -> float:
        """计算记忆保持率 R = e^(-t/S)"""
        if self.S <= 0:
            return 0.0
        # 使用 Python 内置的 exp 避免 numpy 开销
        import math
        return math.exp(-self.t / self.S)
    
    def update_memory(self, attention_weight: float):
        """根据注意力权重更新记忆参数"""
        # 记忆强度增加注意力权重
        self.S += attention_weight
        
        # 如果attention weight足够大(≥0.05)，说明token仍然重要，重置时间
        if attention_weight >= 0.01:
            self.t = 0  # 重置时间，给重要token"续命"
        # 否则时间继续累积，让不重要的token自然衰减
    
    def step_time(self):
        """时间步进"""
        self.t += 0.05  # 进一步降低时间步进，避免记忆衰减过快


class EbbinghausMemoryManager:
    """艾宾浩斯记忆管理器"""
    
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # memories[layer_idx][position] = LayerTokenMemory
        self.memories: List[Dict[int, LayerTokenMemory]] = [
            {} for _ in range(num_layers)
        ]
        self.current_step = 0
    
    def get_layer_retention_weights(self, layer_idx: int, seq_len: int, 
                                   device, dtype) -> torch.Tensor:
        """获取指定层的所有token的保持权重（超高性能版本）"""
        layer_memories = self.memories[layer_idx]
        
        # 快速路径：如果没有记忆，直接返回全1权重
        if not layer_memories:
            return torch.ones(seq_len, device=device, dtype=dtype)
        
        # 预分配numpy数组，避免Python list
        weights_np = np.ones(seq_len, dtype=np.float32)
        
        # 直接遍历字典，避免中间列表
        for pos, memory in layer_memories.items():
            if pos < seq_len and memory.S > 0:
                weights_np[pos] = memory.get_retention_weight()
        
        # 一次性转换为tensor
        return torch.from_numpy(weights_np).to(device=device, dtype=dtype)
    
    def get_all_layer_weights_batch(self, seq_len: int, device, dtype) -> List[torch.Tensor]:
        """批量获取所有层的权重（性能优化版本）"""
        all_weights = []
        
        for layer_idx in range(self.num_layers):
            weights = self.get_layer_retention_weights(layer_idx, seq_len, device, dtype)
            all_weights.append(weights)
        
        return all_weights
    
    def update_layer_memories(self, layer_idx: int, attention_weights: torch.Tensor):
        """
        更新指定层的记忆参数（高性能优化版本）
        attention_weights: [seq_len] 或 [heads, seq_len] 或 [seq_len, seq_len]
        """
        # 处理不同形状的attention weights
        if attention_weights.dim() == 3:  # [seq_len, seq_len]
            attn_scores = attention_weights[-1, :]
        elif attention_weights.dim() == 2:  # [heads, seq_len]
            attn_scores = attention_weights.mean(dim=0)
        else:  # [seq_len]
            attn_scores = attention_weights
        
        # 一次性转换为numpy，避免重复的GPU-CPU传输
        attn_scores_np = attn_scores.detach().cpu().numpy()
        layer_memories = self.memories[layer_idx]
        
        # 向量化过滤：只处理显著的注意力分数
        significant_mask = attn_scores_np > 1e-6
        significant_positions = np.where(significant_mask)[0]
        significant_scores = attn_scores_np[significant_mask]
        
        # 批量处理显著位置
        for i, pos in enumerate(significant_positions):
            pos = int(pos)  # 确保是Python int
            score = float(significant_scores[i])  # 确保是Python float
            
            if pos not in layer_memories:
                layer_memories[pos] = LayerTokenMemory(S=5.0, t=0)
            
            layer_memories[pos].update_memory(score)
    
    def update_all_layer_memories_batch(self, attention_outputs: List[torch.Tensor]):
        """批量更新所有层的记忆（性能优化版本）"""
        for layer_idx, attn_weights in enumerate(attention_outputs):
            if layer_idx < self.num_layers:
                self.update_layer_memories(layer_idx, attn_weights)
    
    def step_all_memories(self):
        """所有记忆的时间步进（高性能版本）"""
        # 高效的延迟清理：减少频率但保持清理
        should_cleanup = (self.current_step % 100 == 0)  # 每100步清理一次
        
        for layer_memories in self.memories:
            if should_cleanup and layer_memories:
                # 批量收集过期位置，减少字典操作
                expired_positions = []
                for pos, memory in layer_memories.items():
                    if memory.get_retention_weight() < 0.005:  # 调整阈值到0.5%
                        expired_positions.append(pos)
                
                # 批量删除
                for pos in expired_positions:
                    del layer_memories[pos]
            
            # 批量时间步进：直接操作values()避免重复查找
            for memory in layer_memories.values():
                memory.step_time()
        
        self.current_step += 1
    
    def get_memory_stats(self, layer_idx: int = 0) -> Dict:
        """获取指定层的记忆统计信息"""
        if layer_idx >= len(self.memories) or not self.memories[layer_idx]:
            return {}
        
        layer_memories = self.memories[layer_idx]
        S_values = [m.S for m in layer_memories.values()]
        t_values = [m.t for m in layer_memories.values()]
        R_values = [m.get_retention_weight() for m in layer_memories.values()]
        
        return {
            'num_tokens': len(layer_memories),
            'avg_strength': np.mean(S_values),
            'avg_time': np.mean(t_values),
            'avg_retention': np.mean(R_values),
            'max_strength': max(S_values) if S_values else 0,
            'max_time': max(t_values) if t_values else 0
        }
    
    def get_all_stats(self) -> Dict:
        """获取所有层的统计信息"""
        stats = {}
        for layer_idx in range(self.num_layers):
            stats[f'layer_{layer_idx}'] = self.get_memory_stats(layer_idx)
        return stats