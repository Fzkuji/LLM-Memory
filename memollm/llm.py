import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import time
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

from .memory import EbbinghausMemoryManager
from .model import VariableLengthCache, VariableLengthModel

warnings.filterwarnings('ignore')


class TokenSampler:
    """Token sampling utility class"""
    
    @staticmethod
    def sample_next_token(
        logits: torch.Tensor, 
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> int:
        """Sample next token with temperature, top-k and top-p sampling"""
        if not do_sample:
            return torch.argmax(logits).item()
        
        # Modify logits in-place to avoid copying
        if temperature != 1.0:
            logits.div_(temperature)
        
        # If there is top_k, filter first
        if top_k > 0 and top_k < logits.size(-1):
            top_k_values, top_k_indices = torch.topk(logits, top_k)
            # Create new smaller logits tensor
            logits = top_k_values
            probs = F.softmax(logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1).item()
            return top_k_indices[sampled_idx].item()
        
        probs = F.softmax(logits, dim=-1)
        
        # Top-p sampling (optimized version)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            
            # Find the cutoff point
            cutoff_idx = (cumulative_probs <= top_p).sum().item()
            if cutoff_idx < sorted_probs.size(0):
                # Truncate and renormalize
                selected_probs = sorted_probs[:cutoff_idx + 1]
                selected_indices = sorted_indices[:cutoff_idx + 1]
                selected_probs = selected_probs / selected_probs.sum()
                
                sampled_idx = torch.multinomial(selected_probs, num_samples=1).item()
                return selected_indices[sampled_idx].item()
        
        return torch.multinomial(probs, num_samples=1).item()


class EbbinghausLLM:
    """Ebbinghaus memory-enhanced LLM"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: Optional[str] = None) -> None:
        print(f"Initializing model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if device is None else device,
            trust_remote_code=True,
            attn_implementation="eager"
        )

        # Get device and model info
        self.device = next(self.model.parameters()).device
        self.num_layers = len(self.model.model.layers) if hasattr(self.model, 'model') else 12
        print(f"Model loaded on device: {self.device}, layers: {self.num_layers}")

        # Initialize memory manager
        self.memory_manager = EbbinghausMemoryManager(self.num_layers)
        
        # Performance optimization cache
        self._cached_memory_weights = None
        self._cached_seq_len = 0
        self._last_update_step = -1
        self._cached_attention_mask = None
        self._cached_mask_seq_len = 0

    def prepare_input(self, text: str) -> Dict[str, Any]:
        """Prepare model input"""
        if "Qwen" in self.model.__class__.__name__:
            messages = [{"role": "user", "content": text}]
            formatted_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_text = text

        return self.tokenizer(formatted_text, return_tensors="pt").to(self.device)

    

    def get_token_weight_details(self, token_ids: torch.Tensor) -> Dict[str, Any]:
        """Get detailed weight information for each token"""
        # Avoid multiple device conversions, convert to CPU at once for tokenizer
        token_ids_list = token_ids.tolist() if not token_ids.is_cuda else token_ids.cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids_list)
        token_texts = [self.tokenizer.decode([tid]) for tid in token_ids_list]
        seq_len = token_ids.shape[0]
        
        detailed_info = {
            'tokens': tokens,
            'token_texts': token_texts,
            'token_ids': token_ids_list,
            'layers_info': {}
        }
        
        # Get weight information for each layer
        for layer_idx in range(self.num_layers):
            layer_info = {
                'retention_weights': [],
                'memory_strength': [],
                'time_steps': [],
                'memory_formula': []
            }
            
            for pos in range(seq_len):
                if pos in self.memory_manager.layer_memories[layer_idx]:
                    memory = self.memory_manager.layer_memories[layer_idx][pos]
                    retention = memory.get_retention_weight()
                    
                    layer_info['retention_weights'].append(round(retention, 4))
                    layer_info['memory_strength'].append(round(memory.strength, 4))
                    layer_info['time_steps'].append(memory.time_steps)
                    layer_info['memory_formula'].append(f"e^(-{memory.time_steps}/{memory.strength:.2f}) = {retention:.4f}")
                else:
                    layer_info['retention_weights'].append(1.0)
                    layer_info['memory_strength'].append(1.0)
                    layer_info['time_steps'].append(0)
                    layer_info['memory_formula'].append("e^(-0/1.00) = 1.0000")
            
            detailed_info['layers_info'][f'layer_{layer_idx}'] = layer_info
        
        return detailed_info

    def print_token_weight_summary(self, token_weights_info: Dict[str, Any], show_layers: Optional[List[int]] = None) -> None:
        """Print token weight summary"""
        if show_layers is None:
            show_layers = [0, -1]
            
        print(f"\n{'='*80}")
        print("TOKEN WEIGHT DETAILS")
        print(f"{'='*80}")
        
        tokens = token_weights_info['tokens']
        token_texts = token_weights_info['token_texts']
        
        # Display basic token information
        print(f"Sequence length: {len(tokens)}")
        print(f"Token list: {' | '.join([f'{i}:{t}' for i, t in enumerate(tokens[:20])])}")
        if len(tokens) > 20:
            print("... (showing first 20)")
        
        # Display weight information for specified layers
        layers_to_show = []
        for layer_idx in show_layers:
            if layer_idx == -1:
                layer_idx = self.num_layers - 1
            if f'layer_{layer_idx}' in token_weights_info['layers_info']:
                layers_to_show.append(layer_idx)
        
        for layer_idx in layers_to_show:
            layer_key = f'layer_{layer_idx}'
            layer_info = token_weights_info['layers_info'][layer_key]
            
            print(f"\n--- {layer_key.upper()} ---")
            print(f"{'Pos':<4} {'Token':<15} {'Strength(S)':<8} {'Time(t)':<7} {'Weight(R)':<8} {'Formula'}")
            print("-" * 80)
            
            for i in range(min(20, len(tokens))):  # Only show first 20
                token_display = token_texts[i][:12] if len(token_texts[i]) > 12 else token_texts[i]
                print(f"{i:<4} {token_display:<15} {layer_info['memory_strength'][i]:<8} "
                      f"{layer_info['time_steps'][i]:<7} {layer_info['retention_weights'][i]:<8} "
                      f"{layer_info['memory_formula'][i]}")
            
            if len(tokens) > 20:
                print("... (showing first 20)")
        
        print(f"{'='*80}")


    
    def _sample_next_token(
        self, 
        logits: torch.Tensor, 
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> int:
        """Sample next token using TokenSampler"""
        return TokenSampler.sample_next_token(logits, temperature, do_sample, top_k, top_p)
    
    def _get_memory_weights(self, seq_len: int) -> List[torch.Tensor]:
        """Get memory weights with caching"""
        # Only recalculate when sequence length changes
        if seq_len != self._cached_seq_len or self._cached_memory_weights is None:
            self._cached_memory_weights = self.memory_manager.get_all_layer_weights(
                seq_len, self.device, self.model.dtype
            )
            self._cached_seq_len = seq_len
        
        return self._cached_memory_weights
    
    
    def _apply_memory_to_past_kv_enhanced(
        self, past_key_values: Optional[Union[Tuple, DynamicCache]], memory_weights: List[torch.Tensor]
    ) -> Optional[Union[Tuple, DynamicCache]]:
        """Apply memory weights without threshold clamping (enhanced version)"""
        if past_key_values is None or not memory_weights:
            return past_key_values
        
        # Handle DynamicCache objects
        if isinstance(past_key_values, DynamicCache):
            # Apply weights to DynamicCache
            max_layers = min(len(past_key_values.key_cache), len(memory_weights))
            
            for layer_idx in range(max_layers):
                weights = memory_weights[layer_idx]
                key_cache = past_key_values.key_cache[layer_idx]
                value_cache = past_key_values.value_cache[layer_idx]
                
                if key_cache is not None and weights.shape[0] == key_cache.shape[2]:
                    # Apply weights directly without clamping
                    weights_expanded = weights.view(1, 1, -1, 1)
                    
                    # Apply to both key and value cache
                    key_cache.mul_(weights_expanded)
                    value_cache.mul_(weights_expanded)
        else:
            # Handle tuple format
            max_layers = min(len(past_key_values), len(memory_weights))
            
            for layer_idx in range(max_layers):
                weights = memory_weights[layer_idx]
                layer_cache = past_key_values[layer_idx]
                
                # 检查layer_cache是否是tuple
                if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
                    key_cache, value_cache = layer_cache
                    
                    if key_cache is not None and hasattr(key_cache, 'shape') and weights.shape[0] == key_cache.shape[2]:
                        # Apply weights directly without clamping
                        weights_expanded = weights.view(1, 1, -1, 1)
                        
                        # Apply to both key and value cache
                        key_cache.mul_(weights_expanded)
                        if value_cache is not None:
                            value_cache.mul_(weights_expanded)
                else:
                    # 可能是其他格式，跳过
                    continue
        
        return past_key_values
    
    def _generate_per_layer_attention_masks(self, current_seq_len: int, tokens_to_delete_per_layer: List[List[int]], device: torch.device) -> List[torch.Tensor]:
        """Generate per-layer attention masks to soft-mask deleted tokens instead of hard deletion"""
        if not tokens_to_delete_per_layer:
            return []
        
        per_layer_masks = []
        
        for layer_idx, tokens_to_delete in enumerate(tokens_to_delete_per_layer):
            if layer_idx < self.num_layers:
                # Create attention mask for this layer: 1 for attend, 0 for mask
                layer_mask = torch.ones(current_seq_len, dtype=torch.bool, device=device)
                
                # Mask the tokens this layer wants to delete
                for pos in tokens_to_delete:
                    if 0 <= pos < current_seq_len:
                        layer_mask[pos] = False
                
                # Convert to additive attention mask format (0 for attend, -inf for mask)
                # Shape: [1, 1, 1, seq_len] for broadcasting to [batch, heads, seq_len, seq_len]
                additive_mask = torch.where(
                    layer_mask, 
                    0.0, 
                    float('-inf')
                ).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                
                per_layer_masks.append(additive_mask)
                
                if layer_idx < 3:  # Debug info for first few layers
                    deleted_count = len(tokens_to_delete)
                    print(f"  DEBUG: Layer {layer_idx} mask: {deleted_count} tokens masked")
            else:
                # For layers beyond what we have deletion info, create empty mask (attend to all)
                empty_mask = torch.zeros(1, 1, 1, current_seq_len, device=device)
                per_layer_masks.append(empty_mask)
        
        return per_layer_masks
    
    def _adjust_memory_per_layer(self, tokens_to_delete_per_layer: List[List[int]]) -> None:
        """Adjust memory manager after layer-specific token deletion"""
        if not tokens_to_delete_per_layer:
            return
        
        # Update each layer independently
        for layer_idx in range(self.num_layers):
            if layer_idx >= len(tokens_to_delete_per_layer) or not tokens_to_delete_per_layer[layer_idx]:
                continue
                
            layer_memories = self.memory_manager.layer_memories[layer_idx]
            deleted_positions = sorted(set(tokens_to_delete_per_layer[layer_idx]))
            
            # Create new memory dictionary with adjusted positions
            new_layer_memories = {}
            
            for old_pos, memory in list(layer_memories.items()):
                # Count how many positions before this one were deleted
                positions_deleted_before = sum(1 for dp in deleted_positions if dp < old_pos)
                
                # Calculate new position
                new_pos = old_pos - positions_deleted_before
                
                # Only keep if the position itself wasn't deleted
                if old_pos not in deleted_positions and new_pos >= 0:
                    new_layer_memories[new_pos] = memory
            
            self.memory_manager.layer_memories[layer_idx] = new_layer_memories
    
    def _delete_tokens_from_sequence(self, token_ids: torch.Tensor, tokens_to_delete: List[int]) -> torch.Tensor:
        """Delete tokens from input sequence"""
        if not tokens_to_delete or token_ids.shape[1] == 0:
            return token_ids
        
        # Sort positions to delete
        positions_to_delete = sorted(set(tokens_to_delete), reverse=True)
        
        # Create a mask for positions to keep
        seq_len = token_ids.shape[1]
        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=token_ids.device)
        
        for pos in positions_to_delete:
            if 0 <= pos < seq_len:
                keep_mask[pos] = False
        
        # Return only the positions we want to keep
        return token_ids[:, keep_mask]
    
    def _forward_with_per_layer_masks(self, input_ids, past_key_values=None, per_layer_masks=None, output_attentions=False):
        """Custom forward method with per-layer attention masks for soft deletion"""
        
        # 使用标准的模型forward，但修改attention mask
        # 这比重新实现整个forward简单得多
        
        # 我们将通过monkey patching每层的attention来实现per-layer mask
        original_forwards = []
        
        try:
            # 保存原始的layer forward方法并替换为自定义版本
            for layer_idx, layer in enumerate(self.model.model.layers):
                if layer_idx < len(per_layer_masks):
                    layer_mask = per_layer_masks[layer_idx]
                    
                    # 保存原始方法
                    original_forward = layer.forward
                    original_forwards.append(original_forward)
                    
                    # 创建带有特定mask的wrapper
                    def create_masked_forward(original_func, mask):
                        def masked_forward(hidden_states, attention_mask=None, **kwargs):
                            # 如果有layer-specific mask，将其与现有mask结合
                            if mask is not None and mask.shape[-1] > 0:
                                if attention_mask is not None:
                                    # 结合现有mask和layer-specific mask
                                    combined_mask = attention_mask + mask.expand_as(attention_mask)
                                else:
                                    # 使用layer-specific mask
                                    batch_size = hidden_states.shape[0]
                                    seq_len = hidden_states.shape[1]
                                    combined_mask = mask.expand(batch_size, 1, seq_len, -1)
                                
                                kwargs['attention_mask'] = combined_mask
                            
                            return original_func(hidden_states, attention_mask=attention_mask, **kwargs)
                        return masked_forward
                    
                    # 替换layer的forward方法
                    layer.forward = create_masked_forward(original_forward, layer_mask)
                else:
                    original_forwards.append(layer.forward)
            
            # 现在调用标准的model forward
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=True,
                return_dict=True
            )
            
            return outputs
            
        finally:
            # 恢复原始的forward方法
            for layer_idx, layer in enumerate(self.model.model.layers):
                if layer_idx < len(original_forwards):
                    layer.forward = original_forwards[layer_idx]
    
    
    def _identify_tokens_to_delete_per_layer(
        self, current_seq_len: int, memory_weights: List[torch.Tensor], threshold: float
    ) -> List[List[int]]:
        """Identify tokens to delete independently for each layer"""
        if not memory_weights:
            return []
        
        tokens_to_delete_per_layer = []
        
        # Process each layer independently
        for layer_idx, weights in enumerate(memory_weights):
            if weights.shape[0] >= current_seq_len:
                layer_weights = weights[:current_seq_len]
                # Find positions where weight is below threshold for this layer
                low_weight_positions = (layer_weights < threshold).nonzero(as_tuple=True)[0].tolist()
                tokens_to_delete_per_layer.append(low_weight_positions)
            else:
                tokens_to_delete_per_layer.append([])
        
        return tokens_to_delete_per_layer
    
    def _update_memory_positions_after_deletion(self, tokens_to_delete: List[int], original_seq_len: int) -> None:
        """Update memory manager positions after token deletion"""
        if not tokens_to_delete:
            return
        
        # Create position mapping
        deleted_positions = set(tokens_to_delete)
        position_mapping = {}
        new_idx = 0
        for old_idx in range(original_seq_len):
            if old_idx not in deleted_positions:
                position_mapping[old_idx] = new_idx
                new_idx += 1
        
        # Update memory positions for all layers
        for layer_idx in range(self.num_layers):
            layer_memories = self.memory_manager.layer_memories[layer_idx]
            new_layer_memories = {}
            
            for old_pos, memory in layer_memories.items():
                if old_pos in position_mapping:
                    new_pos = position_mapping[old_pos]
                    new_layer_memories[new_pos] = memory
            
            self.memory_manager.layer_memories[layer_idx] = new_layer_memories
    
    def _hard_delete_generation_loop(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        top_k: int,
        top_p: float,
        force_exact_length: bool,
        verbose: bool,
        return_attention_weights: bool = False,
        hard_delete_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """Generation loop with true hard deletion using DynamicCache manipulation"""
        generated_ids = []
        current_ids = input_ids
        past_key_values = None
        attention_weights_history = [] if return_attention_weights else None
        
        # Track deletion statistics
        total_cache_deletions = 0
        deletion_events = []
        
        for step in range(max_new_tokens):
            # Forward pass
            outputs = self.model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True
            )
            
            # Update cache
            past_key_values = outputs.past_key_values
            
            # Update memory with attention weights
            if outputs.attentions is not None:
                layer_attentions = [
                    layer_attn[0, :, -1, :].mean(dim=0, keepdim=True) 
                    for layer_attn in outputs.attentions
                ]
                
                # Update memory for all layers
                self.memory_manager.update_all_layer_memories(layer_attentions)
                
                # Record attention history if needed
                if return_attention_weights:
                    attention_data = [attn.squeeze().detach() for attn in layer_attentions]
                    attention_weights_history.append({
                        'step': step,
                        'layer_weights': attention_data
                    })
            
            # Get current sequence length for memory weights
            if past_key_values is not None:
                current_seq_len = past_key_values.get_seq_length()
                if current_seq_len > 0:
                    # Get memory weights for each layer
                    memory_weights = self.memory_manager.get_all_layer_weights(
                        current_seq_len, self.device, self.model.dtype
                    )
                    
                    # Apply memory weights and perform hard deletion
                    self._apply_memory_to_past_kv_enhanced(past_key_values, memory_weights)
                    
                    # Identify tokens to delete per layer
                    tokens_to_delete_per_layer = self._identify_tokens_to_delete_per_layer(
                        current_seq_len, memory_weights, hard_delete_threshold
                    )
                    
                    # Perform hard deletion for each layer independently
                    layer_deletions = 0
                    for layer_idx, positions_to_delete in enumerate(tokens_to_delete_per_layer):
                        if positions_to_delete and isinstance(past_key_values, DynamicCache):
                            # Delete tokens from DynamicCache
                            if layer_idx < len(past_key_values.key_cache):
                                key_cache = past_key_values.key_cache[layer_idx]
                                value_cache = past_key_values.value_cache[layer_idx]
                                
                                if key_cache is not None:
                                    # Create mask for positions to keep
                                    seq_len = key_cache.shape[-2]
                                    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=key_cache.device)
                                    for pos in positions_to_delete:
                                        if 0 <= pos < seq_len:
                                            keep_mask[pos] = False
                                    
                                    # Apply mask to cache
                                    past_key_values.key_cache[layer_idx] = key_cache[:, :, keep_mask, :]
                                    past_key_values.value_cache[layer_idx] = value_cache[:, :, keep_mask, :]
                                    
                                    layer_deletions += len(positions_to_delete)
                                    
                                    # Update memory manager positions for this layer
                                    self._adjust_memory_for_layer(layer_idx, positions_to_delete)
                    
                    if layer_deletions > 0:
                        total_cache_deletions += layer_deletions
                        deletion_events.append({
                            'step': step,
                            'per_layer_deletions': [len(layer) for layer in tokens_to_delete_per_layer],
                            'total_deletions': layer_deletions
                        })
                        
                        if verbose and step < 5:
                            layer_stats = [len(layer) for layer in tokens_to_delete_per_layer]
                            print(f"Step {step}: Hard deleted tokens per layer: {layer_stats[:5]}{'...' if len(layer_stats) > 5 else ''}")
            
            # Get logits and sample next token
            logits = outputs.logits[0, -1, :]
            next_token_id = self._sample_next_token(
                logits, temperature, do_sample, top_k, top_p
            )
            
            generated_ids.append(next_token_id)
            
            # Check if finished
            if next_token_id == self.tokenizer.eos_token_id and not force_exact_length:
                if verbose:
                    print(f"Encountered EOS token, ending early")
                break
            
            # Update input for next iteration
            current_ids = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
            
            # Time step progression
            self.memory_manager.step_all_memories()
            
            # Clear memory weight cache
            self._cached_memory_weights = None
            
            if verbose and (step + 1) % 10 == 0:
                print(f"Generated {step + 1} tokens")
        
        # Calculate final statistics
        final_seq_len = input_ids.shape[1] + len(generated_ids)
        layer_cache_lengths = []
        
        if past_key_values is not None and isinstance(past_key_values, DynamicCache):
            for key_cache in past_key_values.key_cache:
                if key_cache is not None:
                    layer_cache_lengths.append(key_cache.shape[-2])
                else:
                    layer_cache_lengths.append(0)
        
        total_expected_tokens = final_seq_len
        total_actual_tokens = sum(layer_cache_lengths) / len(layer_cache_lengths) if layer_cache_lengths else 0
        
        total_cache_entries = final_seq_len * self.num_layers
        cache_deletion_percentage = (total_cache_deletions / total_cache_entries) * 100 if total_cache_entries > 0 else 0
        
        return {
            'generated_ids': generated_ids,
            'attention_weights_history': attention_weights_history,
            'deletion_events': deletion_events,
            'cache_deletion_percentage': cache_deletion_percentage,
            'total_cache_entries': total_cache_entries,
            'total_cache_deletions': total_cache_deletions,
            'layer_cache_lengths': layer_cache_lengths,
            'min_cache_length': min(layer_cache_lengths) if layer_cache_lengths else 0,
            'max_cache_length': max(layer_cache_lengths) if layer_cache_lengths else 0,
            'total_expected_tokens': total_expected_tokens,
            'total_actual_tokens': total_actual_tokens
        }
    
    def _adjust_memory_for_layer(self, layer_idx: int, positions_to_delete: List[int]) -> None:
        """Adjust memory manager for a specific layer after token deletion"""
        if not positions_to_delete:
            return
        
        layer_memories = self.memory_manager.layer_memories[layer_idx]
        deleted_positions = sorted(set(positions_to_delete))
        
        # Create new memory dictionary with adjusted positions
        new_layer_memories = {}
        
        for old_pos, memory in list(layer_memories.items()):
            # Count how many positions before this one were deleted
            positions_deleted_before = sum(1 for dp in deleted_positions if dp < old_pos)
            
            # Calculate new position
            new_pos = old_pos - positions_deleted_before
            
            # Only keep if the position itself wasn't deleted
            if old_pos not in deleted_positions and new_pos >= 0:
                new_layer_memories[new_pos] = memory
        
        self.memory_manager.layer_memories[layer_idx] = new_layer_memories

    def _common_generation_loop(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        top_k: int,
        top_p: float,
        force_exact_length: bool,
        verbose: bool,
        generation_mode: str,
        return_attention_weights: bool = False,
        hard_delete_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """Common generation loop shared by all generation modes"""
        generated_ids = []
        current_ids = input_ids
        past_key_values = None
        attention_weights_history = [] if return_attention_weights else None
        
        # Track cache deletion statistics for memory_enhanced mode
        total_cache_entries = 0  # Total cache entries (tokens * layers)
        total_cache_deletions = 0  # Total deleted cache entries across all layers
        deletion_events = []
        
        # For memory_enhanced, we need to track the full sequence to handle deletions
        full_sequence_ids = input_ids.clone() if generation_mode == "memory_enhanced" else None
        
        for step in range(max_new_tokens):
            # Get current sequence length
            if step == 0:
                seq_len = current_ids.shape[1]
            else:
                seq_len = len(generated_ids) + input_ids.shape[1]
            
            # Apply memory modifications based on generation mode
            if generation_mode == "memory_enhanced" and past_key_values is not None:
                # For memory modes, we need to track actual cache size
                if isinstance(past_key_values, DynamicCache):
                    actual_cache_len = past_key_values.get_seq_length() if past_key_values.get_seq_length() > 0 else seq_len
                else:
                    # past_key_values[0] 是 (key_cache, value_cache) tuple
                    if past_key_values and len(past_key_values) > 0 and past_key_values[0][0] is not None:
                        actual_cache_len = past_key_values[0][0].shape[2]
                    else:
                        actual_cache_len = seq_len
                
                # Get memory weights for current cache length
                cache_weights = self._get_memory_weights(actual_cache_len)
                
                # For memory_enhanced mode: apply memory weights directly
                past_key_values = self._apply_memory_to_past_kv_enhanced(
                    past_key_values, cache_weights
                )
                
                # Identify tokens to delete per layer based on memory weights
                tokens_to_delete_per_layer = self._identify_tokens_to_delete_per_layer(
                    actual_cache_len, cache_weights, hard_delete_threshold
                )
                
                # Calculate total deletions and statistics
                total_tokens_deleted = sum(len(layer_deletions) for layer_deletions in tokens_to_delete_per_layer)
                
                if total_tokens_deleted > 0:
                    # Calculate unique tokens that any layer wants to delete
                    all_deleted_tokens = set()
                    for layer_deletions in tokens_to_delete_per_layer:
                        all_deleted_tokens.update(layer_deletions)
                    unique_tokens_deleted = len(all_deleted_tokens)
                    
                    # Count cache deletions across all layers (for statistics)
                    cache_deletions_this_step = total_tokens_deleted
                    total_cache_deletions += cache_deletions_this_step
                    
                    deletion_events.append({
                        'step': step,
                        'tokens_deleted': unique_tokens_deleted,
                        'cache_deletions': cache_deletions_this_step,
                        'layers_affected': sum(1 for layer in tokens_to_delete_per_layer if layer),
                        'per_layer_deletions': [len(layer) for layer in tokens_to_delete_per_layer]
                    })
                    
                    if verbose and step < 5:
                        layer_stats = [len(layer) for layer in tokens_to_delete_per_layer]
                        print(f"Step {step}: Per-layer analysis: {layer_stats[:5]}{'...' if len(layer_stats) > 5 else ''} tokens")
                        print(f"  Consensus deletion: {unique_tokens_deleted} tokens (union of all layers)")
                        # Show detailed per-layer deletion info
                        different_layers = []
                        for i in range(min(5, len(tokens_to_delete_per_layer))):
                            if tokens_to_delete_per_layer[i]:
                                different_layers.append(f"L{i}:{len(tokens_to_delete_per_layer[i])}")
                        if different_layers:
                            print(f"  Layers with deletions: {', '.join(different_layers)}")
                    
                    # 使用per-layer attention masks实现软删除（每层独立）
                    if any(layer_deletions for layer_deletions in tokens_to_delete_per_layer):
                        # 生成每层独立的attention masks
                        per_layer_masks = self._generate_per_layer_attention_masks(
                            actual_cache_len, tokens_to_delete_per_layer, self.device
                        )
                        
                        # 存储masks以供forward使用
                        if not hasattr(self, '_current_per_layer_masks'):
                            self._current_per_layer_masks = per_layer_masks
                        else:
                            self._current_per_layer_masks = per_layer_masks
            
            # Forward propagation
            forward_kwargs = {
                "input_ids": current_ids,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": generation_mode != "baseline"  # Only for memory modes
            }
            
            # 检查各层cache长度以进行调试
            if generation_mode == "memory_enhanced" and past_key_values is not None:
                cache_lengths = []
                if isinstance(past_key_values, DynamicCache):
                    for layer_idx in range(len(past_key_values.key_cache)):
                        key_cache = past_key_values.key_cache[layer_idx]
                        if key_cache is not None and hasattr(key_cache, 'shape'):
                            cache_lengths.append(key_cache.shape[2])
                        else:
                            cache_lengths.append(0)
                
                unique_lengths = len(set(cache_lengths))
                if verbose and step < 5:
                    print(f"  DEBUG: Cache lengths before forward: {cache_lengths[:5]}{'...' if len(cache_lengths) > 5 else ''}")
                    print(f"  DEBUG: Unique lengths: {unique_lengths}")
            
            # 对于memory_enhanced模式，使用per-layer masks来实现软删除
            if generation_mode == "memory_enhanced" and hasattr(self, '_current_per_layer_masks'):
                outputs = self._forward_with_per_layer_masks(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    per_layer_masks=self._current_per_layer_masks,
                    output_attentions=generation_mode != "baseline"
                )
            else:
                # baseline模式或没有masks时使用标准forward
                forward_kwargs["past_key_values"] = past_key_values
                outputs = self.model(**forward_kwargs)
            
            # Update past_key_values
            past_key_values = outputs.past_key_values
            
            # Update memory (only for memory-enhanced modes)
            if generation_mode != "baseline" and outputs.attentions is not None:
                # Efficient batch processing of attention weights
                layer_attentions = [
                    layer_attn[0, :, -1, :].mean(dim=0, keepdim=True) 
                    for layer_attn in outputs.attentions
                ]
                
                # Update memory for all layers
                self.memory_manager.update_all_layer_memories(layer_attentions)
                
                # Record attention history (if needed)
                if return_attention_weights and attention_weights_history is not None:
                    # Keep GPU tensor, avoid CPU conversion
                    attention_data = [attn.squeeze().detach() for attn in layer_attentions]
                    attention_weights_history.append({
                        'step': step,
                        'seq_len': seq_len,
                        'layer_weights': attention_data
                    })
            
            # Get logits of the last token
            logits = outputs.logits[0, -1, :]
            
            # Sample next token
            next_token_id = self._sample_next_token(
                logits, temperature, do_sample, top_k, top_p
            )
            
            generated_ids.append(next_token_id)
            
            # Check if finished
            if next_token_id == self.tokenizer.eos_token_id and not force_exact_length:
                if verbose:
                    print(f"Encountered EOS token, ending early")
                break
            
            # Update input (only pass new token)
            current_ids = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
            
            # Update full sequence for memory_enhanced mode
            if generation_mode == "memory_enhanced" and full_sequence_ids is not None:
                full_sequence_ids = torch.cat([full_sequence_ids, current_ids], dim=1)
            
            # Time step progression for memory modes
            if generation_mode != "baseline":
                self.memory_manager.step_all_memories()
                
                # Clear cache to get latest memory weights
                self._cached_memory_weights = None
                self._cached_attention_mask = None
            
            if verbose and (step + 1) % 10 == 0:
                print(f"Generated {step + 1} tokens")
        
        # Calculate cache deletion percentage and layer statistics for memory_enhanced mode
        cache_deletion_percentage = 0.0
        layer_cache_lengths = []
        total_expected_tokens = 0
        total_actual_tokens = 0
        
        # Initialize cache length variables
        min_cache_length = 0
        max_cache_length = 0
        
        if generation_mode == "memory_enhanced":
            # Calculate total expected tokens and actual cache lengths per layer
            final_seq_len = input_ids.shape[1] + len(generated_ids)
            total_expected_tokens = final_seq_len
            
            # Get actual cache lengths from past_key_values if available
            if past_key_values is not None:
                if isinstance(past_key_values, DynamicCache):
                    # Handle DynamicCache
                    for layer_idx in range(len(past_key_values.key_cache)):
                        key_cache = past_key_values.key_cache[layer_idx]
                        if key_cache is not None and hasattr(key_cache, 'shape'):
                            actual_length = key_cache.shape[2]
                        else:
                            actual_length = 0
                        layer_cache_lengths.append(actual_length)
                        total_actual_tokens += actual_length
                else:
                    # Handle tuple format
                    for layer_idx, layer_cache in enumerate(past_key_values):
                        if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
                            key_cache, value_cache = layer_cache
                            if key_cache is not None and hasattr(key_cache, 'shape'):
                                actual_length = key_cache.shape[2]
                            else:
                                actual_length = 0
                        else:
                            # 不是标准tuple格式，假设长度为0
                            actual_length = 0
                        layer_cache_lengths.append(actual_length)
                        total_actual_tokens += actual_length
                
                # Debug: Show cache lengths for verification
                if verbose and len(layer_cache_lengths) >= 3:
                    unique_lengths = len(set(layer_cache_lengths))
                    print(f"🔍 Final cache lengths: Layer 0={layer_cache_lengths[0]}, Layer 1={layer_cache_lengths[1]}, Layer 2={layer_cache_lengths[2]} | {unique_lengths} unique lengths")
                    
                    # Show per-layer deletion analysis if we had deletion events
                    if deletion_events:
                        total_per_layer = [0] * len(layer_cache_lengths)
                        for event in deletion_events:
                            per_layer = event.get('per_layer_deletions', [])
                            for i, count in enumerate(per_layer):
                                if i < len(total_per_layer):
                                    total_per_layer[i] += count
                        
                        # Show which layers wanted to delete more tokens
                        layer_analysis = []
                        for i in range(min(5, len(total_per_layer))):
                            if total_per_layer[i] > 0:
                                layer_analysis.append(f"L{i}:{total_per_layer[i]}")
                        
                        if layer_analysis:
                            print(f"📊 Per-layer deletion desires: {', '.join(layer_analysis)}")
                
                # Calculate average actual tokens across all layers
                if layer_cache_lengths:
                    total_actual_tokens = total_actual_tokens / len(layer_cache_lengths)
                    min_cache_length = min(layer_cache_lengths)
                    max_cache_length = max(layer_cache_lengths)
                else:
                    min_cache_length = 0
                    max_cache_length = 0
            else:
                # If no past_key_values, assume all layers have full length
                layer_cache_lengths = [final_seq_len] * self.num_layers
                total_actual_tokens = final_seq_len
                min_cache_length = final_seq_len
                max_cache_length = final_seq_len
            
            # Calculate total cache entries and deletion percentage
            total_cache_entries = final_seq_len * self.num_layers
            if total_cache_entries > 0:
                cache_deletion_percentage = (total_cache_deletions / total_cache_entries) * 100
        
        return {
            'generated_ids': generated_ids,
            'attention_weights_history': attention_weights_history,
            'deletion_events': deletion_events if generation_mode == "memory_enhanced" else None,
            'cache_deletion_percentage': cache_deletion_percentage if generation_mode == "memory_enhanced" else 0.0,
            'total_cache_entries': total_cache_entries if generation_mode == "memory_enhanced" else 0,
            'total_cache_deletions': total_cache_deletions if generation_mode == "memory_enhanced" else 0,
            'layer_cache_lengths': layer_cache_lengths if generation_mode == "memory_enhanced" else [],
            'min_cache_length': min_cache_length if generation_mode == "memory_enhanced" else 0,
            'max_cache_length': max_cache_length if generation_mode == "memory_enhanced" else 0,
            'total_expected_tokens': total_expected_tokens if generation_mode == "memory_enhanced" else 0,
            'total_actual_tokens': total_actual_tokens if generation_mode == "memory_enhanced" else 0
        }

    @torch.no_grad()
    def generate_baseline(
        self,
        input_text: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 0.9,
        verbose: bool = True,
        force_exact_length: bool = False,
    ) -> Dict[str, Any]:
        """Standard generation method, implemented using forward"""
        # Prepare input
        inputs = self.prepare_input(input_text)
        input_ids = inputs["input_ids"]
        
        start_time = time.time()
        if verbose:
            print(f"Starting baseline generation, input length: {input_ids.shape[1]}")
        
        # Use common generation loop
        result = self._common_generation_loop(
            input_ids, max_new_tokens, temperature, do_sample, top_k, top_p,
            force_exact_length, verbose, "baseline", False
        )
        
        generated_ids = result['generated_ids']
        generation_time = time.time() - start_time
        
        # Build complete sequence
        full_ids = torch.cat([
            input_ids[0],
            torch.tensor(generated_ids, device=self.device, dtype=torch.long)
        ])
        
        # Decode
        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if verbose:
            print(f"Generation completed, time taken: {generation_time:.2f} seconds")
        
        return {
            'full_text': full_text,
            'generated_text': generated_text,
            'num_tokens': len(generated_ids),
            'generation_time': generation_time,
            'generation_mode': 'baseline'
        }


    @torch.no_grad()
    def generate_memory_enhanced(
        self,
        input_text: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 0.9,
        return_attention_weights: bool = False,
verbose: bool = True,
        force_exact_length: bool = False,
        hard_delete_threshold: float = 0.01,  # Threshold for hard deletion
    ) -> Dict[str, Any]:
        """Memory-enhanced generation combining soft delete and hard deletion
        
        This method:
        1. Uses Ebbinghaus memory manager to track memory for each token in each layer
        2. Applies soft delete by adjusting cache weights (clamp to [soft_delete_threshold, 1.0])
        3. Hard deletes tokens when weight < hard_delete_threshold from cache
        4. Stops updating memory for deleted tokens
        5. Supports variable-length cache per layer
        """
        # Prepare input
        inputs = self.prepare_input(input_text)
        input_ids = inputs["input_ids"]
        
        # Initialize memory manager
        self.memory_manager = EbbinghausMemoryManager(self.num_layers)
        
        start_time = time.time()
        if verbose:
            print(f"Starting memory-enhanced generation, input length: {input_ids.shape[1]}")
        
        # Use hard deletion generation loop
        result = self._hard_delete_generation_loop(
            input_ids, max_new_tokens, temperature, do_sample, top_k, top_p,
            force_exact_length, verbose, return_attention_weights,
            hard_delete_threshold=hard_delete_threshold
        )
        
        generated_ids = result['generated_ids']
        attention_weights_history = result['attention_weights_history']
        deletion_events = result['deletion_events']
        cache_deletion_percentage = result['cache_deletion_percentage']
        total_cache_entries = result['total_cache_entries']
        total_cache_deletions = result['total_cache_deletions']
        layer_cache_lengths = result['layer_cache_lengths']
        total_expected_tokens = result['total_expected_tokens']
        total_actual_tokens = result['total_actual_tokens']
        generation_time = time.time() - start_time
        
        # Build complete sequence
        full_ids = torch.cat([
            input_ids[0],
            torch.tensor(generated_ids, device=self.device, dtype=torch.long)
        ])
        
        # Decode
        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if verbose:
            print(f"Generation completed, time taken: {generation_time:.2f} seconds")
            if cache_deletion_percentage > 0:
                print(f"Cache deletion percentage: {cache_deletion_percentage:.2f}% ({total_cache_deletions}/{total_cache_entries})")
            print(f"Expected tokens: {total_expected_tokens}, Average actual tokens per layer: {total_actual_tokens:.1f}")
        
        # Get token weight information
        token_weights_info = None
        if return_attention_weights:
            token_weights_info = self.get_token_weight_details(full_ids)
        
        return {
            'full_text': full_text,
            'generated_text': generated_text,
            'num_tokens': len(generated_ids),
            'generation_time': generation_time,
            'memory_stats': self.memory_manager.get_all_stats(),
            'token_weights': token_weights_info,
            'attention_weights_history': attention_weights_history,
            'generation_mode': 'memory_enhanced',
            # Cache deletion statistics
            'deletion_events': deletion_events,
            'cache_deletion_percentage': cache_deletion_percentage,
            'total_cache_entries': total_cache_entries,
            'total_cache_deletions': total_cache_deletions,
            'layer_cache_lengths': layer_cache_lengths,
            'total_expected_tokens': total_expected_tokens,
            'total_actual_tokens': total_actual_tokens
        }


    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 0.9,
        return_attention_weights: bool = False,
        verbose: bool = True,
        generation_mode: str = "baseline",  # "baseline", "memory_enhanced"
        force_exact_length: bool = False,
        hard_delete_threshold: float = 0.01,  # Threshold for hard deletion
    ) -> Dict[str, Any]:
        """Unified generation interface"""
        
        if generation_mode == "baseline":
            return self.generate_baseline(
                input_text, max_new_tokens, temperature, do_sample, top_k, top_p, verbose,
                force_exact_length
            )
        elif generation_mode == "memory_enhanced":
            return self.generate_memory_enhanced(
                input_text, max_new_tokens, temperature, do_sample, top_k, top_p, 
                return_attention_weights, verbose, force_exact_length, hard_delete_threshold
            )
        else:
            raise ValueError(f"Unknown generation_mode: {generation_mode}. Available modes: baseline, memory_enhanced")

