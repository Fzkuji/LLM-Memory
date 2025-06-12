import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class EfficientEbbinghausMemoryManager:
    """高效的张量化艾宾浩斯记忆管理器"""
    
    def __init__(self, num_layers: int, max_seq_len: int = 8192) -> None:
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # 使用张量存储所有记忆参数，避免Python对象开销
        # Shape: [num_layers, max_seq_len]
        self.strength = torch.ones(num_layers, max_seq_len) * 5.0
        self.time_steps = torch.zeros(num_layers, max_seq_len)
        
        # 有效token掩码，标记哪些位置有活跃的记忆
        # Shape: [num_layers, max_seq_len]
        self.active_mask = torch.zeros(num_layers, max_seq_len, dtype=torch.bool)
        
        # 记忆权重缓存
        self._cached_weights = None
        self._cache_valid = False
        
        self.current_step = 0
        
    def _compute_retention_weights(self) -> torch.Tensor:
        """批量计算所有层的记忆保持率"""
        # R = e^(-t/S)，使用向量化操作
        # 避免除零
        safe_strength = torch.where(self.strength > 0, self.strength, torch.ones_like(self.strength))
        weights = torch.exp(-self.time_steps / safe_strength)
        
        # 应用active_mask，非活跃位置设为1.0
        weights = torch.where(self.active_mask, weights, torch.ones_like(weights))
        
        return weights
    
    def get_layer_retention_weights(self, layer_idx: int, seq_len: int, 
                                   device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """获取指定层的保持率权重"""
        if not self._cache_valid:
            self._cached_weights = self._compute_retention_weights()
            self._cache_valid = True
            
        # 直接从缓存切片，避免重复计算
        layer_weights = self._cached_weights[layer_idx, :seq_len]
        
        # 移到目标设备并转换类型
        return layer_weights.to(device=device, dtype=dtype)
    
    def get_all_layer_weights(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        """批量获取所有层的权重"""
        if not self._cache_valid:
            self._cached_weights = self._compute_retention_weights()
            self._cache_valid = True
            
        # 一次性转换所有需要的权重
        all_weights = self._cached_weights[:, :seq_len].to(device=device, dtype=dtype)
        
        # 转换为列表格式以保持接口兼容
        return [all_weights[i] for i in range(self.num_layers)]
    
    def update_layer_memories(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        """更新指定层的记忆参数（向量化版本）"""
        # 处理不同形状的attention权重
        if attention_weights.dim() == 3:
            attn_scores = attention_weights[-1, :]
        elif attention_weights.dim() == 2:
            attn_scores = attention_weights.mean(dim=0)
        else:
            attn_scores = attention_weights
        
        seq_len = attn_scores.shape[0]
        
        # 将attention scores移到CPU进行记忆更新
        attn_scores_cpu = attn_scores.detach().cpu()
        
        # 标记有显著attention的位置为活跃
        significant_mask = attn_scores_cpu > 1e-6
        self.active_mask[layer_idx, :seq_len] |= significant_mask
        
        # 更新记忆强度（向量化）
        self.strength[layer_idx, :seq_len] += torch.where(
            attn_scores_cpu >= 0.01,
            attn_scores_cpu,
            torch.zeros_like(attn_scores_cpu)
        )
        
        # 重置重要token的时间步
        reset_mask = attn_scores_cpu >= 0.01
        self.time_steps[layer_idx, :seq_len] = torch.where(
            reset_mask,
            torch.zeros_like(self.time_steps[layer_idx, :seq_len]),
            self.time_steps[layer_idx, :seq_len]
        )
        
        # 使缓存失效
        self._cache_valid = False
    
    def update_all_layer_memories(self, attention_outputs: List[torch.Tensor]) -> None:
        """批量更新所有层的记忆"""
        for layer_idx, attn_weights in enumerate(attention_outputs):
            if layer_idx < self.num_layers:
                self.update_layer_memories(layer_idx, attn_weights)
    
    def step_all_memories(self) -> None:
        """时间步进（向量化版本）"""
        # 批量更新所有活跃位置的时间步
        self.time_steps[self.active_mask] += 0.01
        
        # 周期性清理（向量化）
        if self.current_step % 100 == 0:
            # 计算当前权重
            weights = self._compute_retention_weights()
            
            # 标记需要清理的位置
            cleanup_mask = (weights < 0.005) & self.active_mask
            
            # 清理：重置参数并标记为非活跃
            self.strength[cleanup_mask] = 5.0
            self.time_steps[cleanup_mask] = 0.0
            self.active_mask[cleanup_mask] = False
        
        self.current_step += 1
        self._cache_valid = False
    
    def get_memory_stats(self, layer_idx: int = 0) -> Dict[str, float]:
        """获取指定层的记忆统计"""
        layer_mask = self.active_mask[layer_idx]
        if not layer_mask.any():
            return {}
        
        active_strength = self.strength[layer_idx][layer_mask]
        active_time = self.time_steps[layer_idx][layer_mask]
        
        # 计算保持率
        safe_strength = torch.where(active_strength > 0, active_strength, torch.ones_like(active_strength))
        active_retention = torch.exp(-active_time / safe_strength)
        
        return {
            'num_tokens': int(layer_mask.sum()),
            'avg_strength': float(active_strength.mean()),
            'avg_time': float(active_time.mean()),
            'avg_retention': float(active_retention.mean()),
            'max_strength': float(active_strength.max()),
            'max_time': float(active_time.max())
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有层的统计"""
        stats = {}
        for layer_idx in range(self.num_layers):
            stats[f'layer_{layer_idx}'] = self.get_memory_stats(layer_idx)
        return stats
    
    def get_tokens_to_delete(self, layer_idx: int, seq_len: int, threshold: float) -> List[int]:
        """获取需要删除的token位置（与原接口兼容）"""
        if not self._cache_valid:
            self._cached_weights = self._compute_retention_weights()
            self._cache_valid = True
        
        layer_weights = self._cached_weights[layer_idx, :seq_len]
        
        # 找出低于阈值的位置
        delete_mask = (layer_weights < threshold) & self.active_mask[layer_idx, :seq_len]
        positions = torch.where(delete_mask)[0].tolist()
        
        return positions
    
    def get_token_memory_info(self, layer_idx: int, pos: int) -> Optional[Dict[str, float]]:
        """获取特定位置的记忆信息（兼容接口）"""
        if not (0 <= layer_idx < self.num_layers and 0 <= pos < self.max_seq_len):
            return None
            
        if not self.active_mask[layer_idx, pos]:
            return None
            
        strength = float(self.strength[layer_idx, pos])
        time_steps = float(self.time_steps[layer_idx, pos])
        retention = math.exp(-time_steps / strength) if strength > 0 else 0.0
        
        return {
            'strength': strength,
            'time_steps': time_steps,
            'retention': retention
        }
    
    def adjust_positions_after_deletion(self, layer_idx: int, deleted_positions: List[int]) -> None:
        """调整删除token后的位置映射"""
        if not deleted_positions:
            return
            
        # 将deleted_positions转为tensor进行批量处理
        deleted_tensor = torch.tensor(sorted(deleted_positions), dtype=torch.long)
        
        # 创建新的张量来存储调整后的数据
        new_strength = torch.ones(self.max_seq_len) * 5.0
        new_time_steps = torch.zeros(self.max_seq_len)
        new_active_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        
        # 计算新位置映射
        old_pos = 0
        new_pos = 0
        
        while old_pos < self.max_seq_len and self.active_mask[layer_idx, old_pos].any():
            if old_pos not in deleted_positions:
                # 复制到新位置
                new_strength[new_pos] = self.strength[layer_idx, old_pos]
                new_time_steps[new_pos] = self.time_steps[layer_idx, old_pos]
                new_active_mask[new_pos] = self.active_mask[layer_idx, old_pos]
                new_pos += 1
            old_pos += 1
        
        # 更新层数据
        self.strength[layer_idx] = new_strength
        self.time_steps[layer_idx] = new_time_steps
        self.active_mask[layer_idx] = new_active_mask
        
        # 使缓存失效
        self._cache_valid = False