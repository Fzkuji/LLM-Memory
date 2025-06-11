import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
from transformers import Cache
from transformers.cache_utils import DynamicCache


class VariableLengthCache(Cache):
    """
    Variable-length KV cache that supports different cache lengths per layer.
    Each layer can independently delete tokens from its cache.
    """
    
    def __init__(self):
        super().__init__()
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []
        self._seen_tokens = 0
        self._cos_sin_rerotation_cache = {}
        
    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Get sequence length. If layer_idx is specified, return length for that layer."""
        if layer_idx is not None:
            if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
                return self.key_cache[layer_idx].shape[-2]
            return 0
        
        # Return minimum length across all layers
        if not self.key_cache:
            return 0
        
        lengths = []
        for key_cache in self.key_cache:
            if key_cache is not None:
                lengths.append(key_cache.shape[-2])
        
        return min(lengths) if lengths else 0
    
    def get_max_length(self) -> Optional[int]:
        """Get maximum cache size - return None for unlimited"""
        return None
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key and value states"""
        # Ensure we have enough cache slots
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
        
        # Initialize cache for this layer if needed
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Concatenate new states to existing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def delete_tokens(self, layer_idx: int, positions_to_delete: List[int]) -> None:
        """Delete specific token positions from cache for a given layer"""
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return
        
        if not positions_to_delete:
            return
        
        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]
        seq_len = key_cache.shape[-2]
        
        # Create mask for positions to keep
        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=key_cache.device)
        for pos in positions_to_delete:
            if 0 <= pos < seq_len:
                keep_mask[pos] = False
        
        # Apply mask to cache
        self.key_cache[layer_idx] = key_cache[:, :, keep_mask, :]
        self.value_cache[layer_idx] = value_cache[:, :, keep_mask, :]
    
    def apply_memory_weights(self, layer_idx: int, memory_weights: torch.Tensor) -> None:
        """Apply memory weights to specific layer cache"""
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return
        
        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]
        cache_len = key_cache.shape[-2]
        
        # Ensure weights match cache length
        if memory_weights.shape[0] != cache_len:
            # Truncate or pad weights to match cache length
            if memory_weights.shape[0] > cache_len:
                memory_weights = memory_weights[:cache_len]
            else:
                # This shouldn't happen in normal operation
                return
        
        # Apply weights (shape: [seq_len] -> [1, 1, seq_len, 1])
        weights_expanded = memory_weights.view(1, 1, -1, 1)
        
        # Apply to both key and value cache
        self.key_cache[layer_idx] = key_cache * weights_expanded
        self.value_cache[layer_idx] = value_cache * weights_expanded
    
    def get_layer_cache_lengths(self) -> List[int]:
        """Get cache length for each layer"""
        lengths = []
        for key_cache in self.key_cache:
            if key_cache is not None:
                lengths.append(key_cache.shape[-2])
            else:
                lengths.append(0)
        return lengths
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search (not implemented for this use case)"""
        raise NotImplementedError("Beam search reordering not implemented for VariableLengthCache")


class VariableLengthModel(nn.Module):
    """
    A wrapper around transformers models that supports variable-length KV cache per layer.
    This enables true hard deletion of tokens from different layers independently.
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
        # Track whether we should use variable-length cache
        self.use_variable_cache = True
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """Forward pass with support for variable-length cache"""
        
        # Initialize variable-length cache if needed
        if use_cache and past_key_values is None and self.use_variable_cache:
            past_key_values = VariableLengthCache()
        
        # Call the base model
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs
    ):
        """Prepare inputs for generation with variable-length cache support"""
        
        # Initialize variable-length cache if needed
        if use_cache and past_key_values is None and self.use_variable_cache:
            past_key_values = VariableLengthCache()
        
        return self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs
        )
    
    def delete_tokens_from_layer(self, layer_idx: int, positions_to_delete: List[int], past_key_values: VariableLengthCache) -> None:
        """Delete specific tokens from a specific layer's cache"""
        if isinstance(past_key_values, VariableLengthCache):
            past_key_values.delete_tokens(layer_idx, positions_to_delete)
    
    def apply_memory_weights_to_layer(self, layer_idx: int, memory_weights: torch.Tensor, past_key_values: VariableLengthCache) -> None:
        """Apply memory weights to a specific layer's cache"""
        if isinstance(past_key_values, VariableLengthCache):
            past_key_values.apply_memory_weights(layer_idx, memory_weights)
    
    def get_layer_cache_lengths(self, past_key_values: VariableLengthCache) -> List[int]:
        """Get cache lengths for all layers"""
        if isinstance(past_key_values, VariableLengthCache):
            return past_key_values.get_layer_cache_lengths()
        else:
            # Fallback for standard cache
            if past_key_values is None:
                return []
            
            lengths = []
            if isinstance(past_key_values, DynamicCache):
                for key_cache in past_key_values.key_cache:
                    if key_cache is not None:
                        lengths.append(key_cache.shape[-2])
                    else:
                        lengths.append(0)
            elif isinstance(past_key_values, (list, tuple)):
                for layer_cache in past_key_values:
                    if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
                        key_cache = layer_cache[0]
                        if key_cache is not None:
                            lengths.append(key_cache.shape[-2])
                        else:
                            lengths.append(0)
                    else:
                        lengths.append(0)
            
            return lengths
    
    def __getattr__(self, name):
        """Delegate attribute access to base model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)