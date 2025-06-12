import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LayerTokenMemory:
    """Memory information for each token in each layer"""
    strength: float = 1.0  # Memory strength (S) - enhanced default value
    time_steps: int = 0    # Time steps since last access (t)
    
    def get_retention_weight(self) -> float:
        """Calculate memory retention rate R = e^(-t/S)"""
        if self.strength <= 0:
            return 0.0
        # Use Python built-in exp to avoid numpy overhead
        import math
        return math.exp(-self.time_steps / self.strength)
    
    def update_memory(self, attention_weight: float) -> None:
        """Update memory parameters based on attention weight"""
        # Memory strength increases with attention weight
        if attention_weight >= 0.01:
            self.time_steps = 0  # Reset time, extend important token's life
            self.strength += attention_weight

    def step_time(self) -> None:
        """Time step progression"""
        self.time_steps += 0.01  # Increase time step to make forgetting faster for demonstration


class EbbinghausMemoryManager:
    """Ebbinghaus memory manager"""
    
    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        # layer_memories[layer_idx][position] = LayerTokenMemory
        self.layer_memories: List[Dict[int, LayerTokenMemory]] = [
            {} for _ in range(num_layers)
        ]
        self.current_step = 0
        self.deleted_tokens = set()  # Track deleted token positions
    
    def get_layer_retention_weights(self, layer_idx: int, seq_len: int, 
                                   device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get retention weights for all tokens in specified layer (ultra high performance version)"""
        layer_memories = self.layer_memories[layer_idx]
        
        # Fast path: if no memory, directly return all-ones weights
        if not layer_memories:
            return torch.ones(seq_len, device=device, dtype=dtype)
        
        # Pre-allocate numpy array, avoid Python list
        weights_np = np.ones(seq_len, dtype=np.float32)
        
        # Directly iterate dictionary, avoid intermediate lists
        for pos, memory in layer_memories.items():
            if pos < seq_len and memory.strength > 0:
                weights_np[pos] = memory.get_retention_weight()
        
        # Convert to tensor at once
        return torch.from_numpy(weights_np).to(device=device, dtype=dtype)
    
    def get_all_layer_weights(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        """Batch get weights for all layers (performance optimized version)"""
        all_weights = []
        
        for layer_idx in range(self.num_layers):
            weights = self.get_layer_retention_weights(layer_idx, seq_len, device, dtype)
            all_weights.append(weights)
        
        return all_weights
    
    def update_layer_memories(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        """
        Update memory parameters for specified layer (high performance optimized version)
        attention_weights: [seq_len] or [heads, seq_len] or [seq_len, seq_len]
        """
        # Handle different shapes of attention weights
        if attention_weights.dim() == 3:  # [seq_len, seq_len]
            attn_scores = attention_weights[-1, :]
        elif attention_weights.dim() == 2:  # [heads, seq_len]
            attn_scores = attention_weights.mean(dim=0)
        else:  # [seq_len]
            attn_scores = attention_weights
        
        # Convert to numpy at once, avoid repeated GPU-CPU transfers
        attn_scores_np = attn_scores.detach().cpu().numpy()
        layer_memories = self.layer_memories[layer_idx]
        
        # Vectorized filtering: only process significant attention scores
        significant_mask = attn_scores_np > 1e-6
        significant_positions = np.where(significant_mask)[0]
        significant_scores = attn_scores_np[significant_mask]
        
        # Batch process significant positions
        for i, pos in enumerate(significant_positions):
            pos = int(pos)  # Ensure it's Python int
            score = float(significant_scores[i])  # Ensure it's Python float
            
            if pos not in layer_memories:
                layer_memories[pos] = LayerTokenMemory(strength=5.0, time_steps=0)
            
            layer_memories[pos].update_memory(score)
    
    def update_all_layer_memories(self, attention_outputs: List[torch.Tensor]) -> None:
        """Batch update memory for all layers (performance optimized version)"""
        for layer_idx, attn_weights in enumerate(attention_outputs):
            if layer_idx < self.num_layers:
                self.update_layer_memories(layer_idx, attn_weights)
    
    def step_all_memories(self) -> None:
        """Time step progression for all memories (high performance version)"""
        # Efficient lazy cleanup: reduce frequency but maintain cleanup
        should_cleanup = (self.current_step % 100 == 0)  # Clean every 100 steps
        
        for layer_memories in self.layer_memories:
            if should_cleanup and layer_memories:
                # Batch collect expired positions, reduce dictionary operations
                expired_positions = []
                for pos, memory in layer_memories.items():
                    if memory.get_retention_weight() < 0.005:  # Adjust threshold to 0.5%
                        expired_positions.append(pos)
                
                # Batch deletion
                for pos in expired_positions:
                    del layer_memories[pos]
            
            # Batch time stepping: directly operate on values() to avoid repeated lookups
            for memory in layer_memories.values():
                memory.step_time()
        
        self.current_step += 1
    
    def get_memory_stats(self, layer_idx: int = 0) -> Dict[str, float]:
        """Get memory statistics for specified layer"""
        if layer_idx >= len(self.layer_memories) or not self.layer_memories[layer_idx]:
            return {}
        
        layer_memories = self.layer_memories[layer_idx]
        strength_values = [m.strength for m in layer_memories.values()]
        time_values = [m.time_steps for m in layer_memories.values()]
        R_values = [m.get_retention_weight() for m in layer_memories.values()]
        
        return {
            'num_tokens': len(layer_memories),
            'avg_strength': np.mean(strength_values),
            'avg_time': np.mean(time_values),
            'avg_retention': np.mean(R_values),
            'max_strength': max(strength_values) if strength_values else 0,
            'max_time': max(time_values) if time_values else 0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all layers"""
        stats = {}
        for layer_idx in range(self.num_layers):
            stats[f'layer_{layer_idx}'] = self.get_memory_stats(layer_idx)
        return stats