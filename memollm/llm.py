import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from typing import Dict, List, Optional, Union
import warnings

from .memory import EbbinghausMemoryManager

warnings.filterwarnings('ignore')


class EbbinghausLLM:
    """艾宾浩斯记忆增强的LLM"""

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", device=None):
        print(f"初始化模型: {model_name}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if device is None else device,
            trust_remote_code=True,
            attn_implementation="eager"
        )

        # 获取设备和模型信息
        self.device = next(self.model.parameters()).device
        self.num_layers = len(self.model.model.layers) if hasattr(self.model, 'model') else 12
        print(f"模型加载到设备: {self.device}, 层数: {self.num_layers}")

        # 初始化记忆管理器
        self.memory_manager = EbbinghausMemoryManager(self.num_layers)
        
        # 性能优化缓存
        self._cached_memory_weights = None
        self._cached_seq_len = 0
        self._last_update_step = -1
        self._cached_attention_mask = None
        self._cached_mask_seq_len = 0

    def prepare_input(self, text: str) -> Dict:
        """准备模型输入"""
        if "Qwen" in self.model.__class__.__name__:
            messages = [{"role": "user", "content": text}]
            formatted_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_text = text

        return self.tokenizer(formatted_text, return_tensors="pt").to(self.device)

    

    def get_detailed_token_weights(self, token_ids: torch.Tensor) -> Dict:
        """获取每个token的详细权重信息"""
        # 避免多次device转换，一次性转换到CPU用于tokenizer
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
        
        # 为每一层获取权重信息
        for layer_idx in range(self.num_layers):
            layer_info = {
                'retention_weights': [],
                'memory_strength': [],
                'time_steps': [],
                'memory_formula': []
            }
            
            for pos in range(seq_len):
                if pos in self.memory_manager.memories[layer_idx]:
                    memory = self.memory_manager.memories[layer_idx][pos]
                    retention = memory.get_retention_weight()
                    
                    layer_info['retention_weights'].append(round(retention, 4))
                    layer_info['memory_strength'].append(round(memory.S, 4))
                    layer_info['time_steps'].append(memory.t)
                    layer_info['memory_formula'].append(f"e^(-{memory.t}/{memory.S:.2f}) = {retention:.4f}")
                else:
                    layer_info['retention_weights'].append(1.0)
                    layer_info['memory_strength'].append(1.0)
                    layer_info['time_steps'].append(0)
                    layer_info['memory_formula'].append("e^(-0/1.00) = 1.0000")
            
            detailed_info['layers_info'][f'layer_{layer_idx}'] = layer_info
        
        return detailed_info

    def print_token_weights_summary(self, token_weights_info: Dict, show_layers: Optional[List[int]] = None):
        """打印token权重摘要"""
        if show_layers is None:
            show_layers = [0, -1]
            
        print(f"\n{'='*80}")
        print("TOKEN权重详细信息")
        print(f"{'='*80}")
        
        tokens = token_weights_info['tokens']
        token_texts = token_weights_info['token_texts']
        
        # 显示token基本信息
        print(f"序列长度: {len(tokens)}")
        print(f"Token列表: {' | '.join([f'{i}:{t}' for i, t in enumerate(tokens[:20])])}")
        if len(tokens) > 20:
            print("... (显示前20个)")
        
        # 显示指定层的权重信息
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
            print(f"{'位置':<4} {'Token':<15} {'强度(S)':<8} {'时间(t)':<7} {'权重(R)':<8} {'公式'}")
            print("-" * 80)
            
            for i in range(min(20, len(tokens))):  # 只显示前20个
                token_display = token_texts[i][:12] if len(token_texts[i]) > 12 else token_texts[i]
                print(f"{i:<4} {token_display:<15} {layer_info['memory_strength'][i]:<8} "
                      f"{layer_info['time_steps'][i]:<7} {layer_info['retention_weights'][i]:<8} "
                      f"{layer_info['memory_formula'][i]}")
            
            if len(tokens) > 20:
                print("... (显示前20个)")
        
        print(f"{'='*80}")


    def _sample_next_token(
        self, 
        logits: torch.Tensor, 
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> int:
        """采样下一个token"""
        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature
            
        # 计算概率
        probs = F.softmax(logits, dim=-1)
        
        if not do_sample:
            # 贪心采样
            return torch.argmax(probs).item()
        
        # Top-k 采样
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
            probs = torch.zeros_like(probs)
            probs.scatter_(0, top_k_indices, top_k_probs)
            probs = probs / probs.sum()
        
        # Top-p (nucleus) 采样
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            
            # 去掉累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()
        
        # 多项式采样
        return torch.multinomial(probs, num_samples=1).item()
    
    def _sample_next_token_fast(
        self, 
        logits: torch.Tensor, 
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> int:
        """优化的采样方法"""
        if not do_sample:
            return torch.argmax(logits).item()
        
        # 原地修改logits避免复制
        if temperature != 1.0:
            logits.div_(temperature)
        
        # 如果有top_k，先过滤
        if top_k > 0 and top_k < logits.size(-1):
            top_k_values, top_k_indices = torch.topk(logits, top_k)
            # 创建新的较小logits tensor
            logits = top_k_values
            probs = F.softmax(logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1).item()
            return top_k_indices[sampled_idx].item()
        
        probs = F.softmax(logits, dim=-1)
        
        # Top-p采样（优化版）
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            
            # 找到截断点
            cutoff_idx = (cumulative_probs <= top_p).sum().item()
            if cutoff_idx < sorted_probs.size(0):
                # 截断并重新归一化
                selected_probs = sorted_probs[:cutoff_idx + 1]
                selected_indices = sorted_indices[:cutoff_idx + 1]
                selected_probs = selected_probs / selected_probs.sum()
                
                sampled_idx = torch.multinomial(selected_probs, num_samples=1).item()
                return selected_indices[sampled_idx].item()
        
        return torch.multinomial(probs, num_samples=1).item()
    
    def _get_memory_weights_fast(self, seq_len: int):
        """快速获取记忆权重（带缓存）"""
        # 只有序列长度改变时才重新计算
        if seq_len != self._cached_seq_len or self._cached_memory_weights is None:
            self._cached_memory_weights = self.memory_manager.get_all_layer_weights_batch(
                seq_len, self.device, self.model.dtype
            )
            self._cached_seq_len = seq_len
        
        return self._cached_memory_weights
    
    def _apply_memory_to_past_kv_fast(self, past_key_values, memory_weights):
        """快速应用记忆权重到past_key_values（原地修改）"""
        if past_key_values is None or not memory_weights:
            return past_key_values
        
        # 批量处理，只对前一半层
        max_layers = min(len(past_key_values), len(memory_weights), self.num_layers // 2)
        
        for layer_idx in range(max_layers):
            weights = memory_weights[layer_idx]
            key_cache, value_cache = past_key_values[layer_idx]
            
            if weights.shape[0] == key_cache.shape[2]:
                # 原地修改，避免新tensor创建
                adjusted_weights = torch.clamp(weights, min=0.8, max=1.0)
                weights_expanded = adjusted_weights.view(1, 1, -1, 1)
                
                # 原地修改key cache
                key_cache.mul_(weights_expanded)
        
        return past_key_values
    
    def _generate_dynamic_attention_mask_fast(self, current_seq_len: int, memory_weights: List[torch.Tensor]):
        """快速生成attention mask（向量化操作）"""
        if not memory_weights:
            return None
        
        # 尝试使用缓存的mask（扩展到当前长度）
        if self._cached_attention_mask is not None and self._cached_mask_seq_len < current_seq_len:
            # 扩展旧mask并计算新部分
            old_len = self._cached_mask_seq_len
            new_mask = self._compute_mask_for_range(memory_weights, old_len, current_seq_len)
            if new_mask is not None:
                # 拼接旧mask和新计算的部分
                self._cached_attention_mask = torch.cat([self._cached_attention_mask[:, :old_len], new_mask], dim=1)
                self._cached_mask_seq_len = current_seq_len
                return self._cached_attention_mask
        
        # 重新计算整个mask
        threshold = 0.001
        min_layers = len(memory_weights) // 2  # 整数除法更快
        
        # 快速筛选有效权重
        valid_weights = []
        for w in memory_weights:
            if w.shape[0] >= current_seq_len:
                valid_weights.append(w[:current_seq_len])
        
        if valid_weights:
            # 使用torch.stack一次性处理所有层
            stacked_weights = torch.stack(valid_weights)  # [num_layers, seq_len]
            
            # 使用tensor操作计算，保持设备一致性
            low_weight_mask = (stacked_weights < threshold).sum(dim=0)  # [seq_len]
            
            # 直接生成float mask，避免多次类型转换
            attention_mask = (low_weight_mask < min_layers).float().unsqueeze(0)
            
            # 缓存结果
            self._cached_attention_mask = attention_mask
            self._cached_mask_seq_len = current_seq_len
            
            return attention_mask
        
        return None
    
    def _compute_mask_for_range(self, memory_weights: List[torch.Tensor], start: int, end: int):
        """计算指定范围的mask"""
        threshold = 0.001
        min_layers = len(memory_weights) // 2
        
        valid_weights = []
        for w in memory_weights:
            if w.shape[0] >= end:
                valid_weights.append(w[start:end])
        
        if valid_weights:
            stacked_weights = torch.stack(valid_weights)
            low_weight_mask = (stacked_weights < threshold).sum(dim=0)
            new_mask = (low_weight_mask < min_layers).float().unsqueeze(0)
            return new_mask
        
        return None
    
    def _apply_memory_to_past_kv(self, past_key_values, memory_weights):
        """将记忆权重应用到past_key_values"""
        if past_key_values is None or not memory_weights:
            return past_key_values
            
        # 修改past_key_values
        for layer_idx in range(min(len(past_key_values), len(memory_weights))):
            if layer_idx >= self.num_layers // 2:  # 只对前一半层应用
                break
                
            weights = memory_weights[layer_idx]
            key_cache, value_cache = past_key_values[layer_idx]
            
            if weights.shape[0] == key_cache.shape[2]:  # 检查维度匹配
                # Soft delete: 权重范围[0.8, 1.0]
                adjusted_weights = torch.clamp(weights, min=0.8, max=1.0)
                weights_expanded = adjusted_weights.view(1, 1, -1, 1)
                
                # 修改key cache
                new_key = key_cache * weights_expanded
                past_key_values[layer_idx] = (new_key, value_cache)
        
        return past_key_values
    
    def _generate_dynamic_attention_mask(self, current_seq_len: int, memory_weights: List[torch.Tensor]):
        """动态生成attention mask"""
        if not memory_weights:
            return None
            
        # 计算需要多少层同时满足条件才mask
        min_layers = int(len(memory_weights) * 0.5)  # 50%的层
        threshold = 0.001
        
        # 统计每个位置在多少层中权重低于阈值
        low_weight_counts = torch.zeros(current_seq_len, device=self.device)
        
        for layer_weights in memory_weights:
            if layer_weights.shape[0] >= current_seq_len:
                low_weight_mask = layer_weights[:current_seq_len] < threshold
                low_weight_counts += low_weight_mask.float()
        
        # 当超过min_layers个层的权重都低于阈值时，才真正mask掉
        sparse_mask = low_weight_counts >= min_layers
        
        # 创建attention mask (1表示保留，0表示mask)
        attention_mask = (~sparse_mask).float().unsqueeze(0)  # [1, seq_len]
        
        return attention_mask

    def _remove_low_weight_tokens(self, past_key_values, memory_weights, full_input_ids, threshold=0.01):
        """直接删除低权重token而不是使用mask"""
        if past_key_values is None or not memory_weights:
            return past_key_values, full_input_ids, []
        
        seq_len = full_input_ids.shape[1]
        
        # 获取所有层的平均权重来决定删除哪些token
        avg_weights = torch.zeros(seq_len, device=self.device)
        valid_layer_count = 0
        
        for layer_weights in memory_weights:
            if layer_weights.shape[0] >= seq_len:
                avg_weights += layer_weights[:seq_len]
                valid_layer_count += 1
        
        if valid_layer_count > 0:
            avg_weights = avg_weights / valid_layer_count
        
        # 找到需要保留的token位置
        keep_mask = avg_weights >= threshold
        keep_indices = torch.where(keep_mask)[0]
        removed_indices = torch.where(~keep_mask)[0].tolist()
        
        if len(keep_indices) == seq_len:
            # 没有token需要删除
            return past_key_values, full_input_ids, []
        
        if len(keep_indices) == 0:
            # 不能删除所有token，保留最后一个
            keep_indices = torch.tensor([seq_len - 1], device=self.device)
            removed_indices = list(range(seq_len - 1))
        
        # 确保 keep_indices 是有序的，避免索引错误
        keep_indices, _ = torch.sort(keep_indices)
        
        # 限制删除的token数量，避免删除太多导致问题
        max_remove = int(seq_len * 0.5)  # 最多删除50%的token
        if len(removed_indices) > max_remove:
            # 只删除权重最低的部分token
            weights_to_remove = avg_weights[~keep_mask]
            _, lowest_indices = torch.topk(weights_to_remove, max_remove, largest=False)
            actual_remove_indices = torch.tensor(removed_indices, device=self.device)[lowest_indices]
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)
            keep_mask[actual_remove_indices] = False
            keep_indices = torch.where(keep_mask)[0]
            keep_indices, _ = torch.sort(keep_indices)
            removed_indices = actual_remove_indices.tolist()
        
        # 更新input_ids
        new_input_ids = full_input_ids[:, keep_indices]
        
        # 安全地更新past_key_values
        new_past_key_values = []
        try:
            for layer_idx, (key_cache, value_cache) in enumerate(past_key_values):
                # 检查维度
                if key_cache.shape[2] != seq_len or value_cache.shape[2] != seq_len:
                    # 维度不匹配，跳过删除
                    return past_key_values, full_input_ids, []
                
                # key_cache和value_cache的形状通常是[batch_size, num_heads, seq_len, head_dim]
                new_key = key_cache[:, :, keep_indices, :]
                new_value = value_cache[:, :, keep_indices, :]
                new_past_key_values.append((new_key, new_value))
        except Exception as e:
            # 如果出现错误，返回原始值
            print(f"删除token时出错: {e}")
            return past_key_values, full_input_ids, []
        
        # 确保返回正确的格式
        new_past_key_values = tuple(new_past_key_values)
        
        return new_past_key_values, new_input_ids, removed_indices
    
    @torch.no_grad()
    def generate_sparse_delete(
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
        deletion_threshold: float = 0.01,
        deletion_interval: int = 5,  # 每隔多少步删除一次
    ):
        """新的稀疏实现：直接删除低权重token而不是mask"""
        # 准备输入
        inputs = self.prepare_input(input_text)
        input_ids = inputs["input_ids"]
        
        # 初始化记忆管理器
        self.memory_manager = EbbinghausMemoryManager(self.num_layers)
        
        start_time = time.time()
        if verbose:
            print(f"开始稀疏删除生成，输入长度: {input_ids.shape[1]}")
        
        generated_ids = []
        full_input_ids = input_ids.clone()  # 保持完整的input_ids历史
        past_key_values = None
        attention_weights_history = [] if return_attention_weights else None
        total_removed = 0
        
        for step in range(max_new_tokens):
            current_seq_len = full_input_ids.shape[1]
            
            # 获取记忆权重
            memory_weights = self.memory_manager.get_all_layer_weights_batch(
                current_seq_len, self.device, self.model.dtype
            )
            
            # 删除低权重token（按间隔进行）
            if step > 0 and step % deletion_interval == 0:  # 按指定间隔删除
                old_past_kv, full_input_ids, removed_indices = self._remove_low_weight_tokens(
                    past_key_values, memory_weights, full_input_ids, deletion_threshold
                )
                
                if removed_indices:
                    total_removed += len(removed_indices)
                    if verbose and step < 5:
                        print(f"Step {step}: 删除了 {len(removed_indices)} 个token, 剩余 {full_input_ids.shape[1]} 个token")
                    
                    # 如果删除了token，重置past_key_values以避免格式问题
                    past_key_values = None
                    
                    # 批量更新记忆管理器（优化版）
                    removed_set = set(removed_indices)  # 使用集合加速查找
                    for layer_idx in range(self.num_layers):
                        layer_memories = self.memory_manager.memories[layer_idx]
                        if not layer_memories:
                            continue
                        
                        # 一次性构建新的记忆映射
                        new_memories = {}
                        offset_map = {}  # 缓存偏移量计算
                        offset = 0
                        
                        for pos in range(full_input_ids.shape[1] + len(removed_indices)):
                            if pos in removed_set:
                                offset += 1
                            else:
                                offset_map[pos] = pos - offset
                        
                        # 批量重映射
                        for old_pos, memory in layer_memories.items():
                            if old_pos not in removed_set and old_pos in offset_map:
                                new_memories[offset_map[old_pos]] = memory
                        
                        self.memory_manager.memories[layer_idx] = new_memories
            
            # 前向传播
            if step == 0 or past_key_values is None:
                # 第一步或删除token后需要传入完整序列
                current_ids = full_input_ids
            else:
                # 只传入最新生成的token
                current_ids = torch.tensor([[generated_ids[-1]]], device=self.device, dtype=torch.long)
            
            outputs = self.model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,  # 必须为True以更新记忆
                return_dict=True
            )
            
            # 更新past_key_values
            past_key_values = outputs.past_key_values
            
            # 更新记忆
            if outputs.attentions is not None:
                layer_attentions = [
                    layer_attn[0, :, -1, :].mean(dim=0, keepdim=True) 
                    for layer_attn in outputs.attentions
                ]
                
                # 更新所有层的记忆
                self.memory_manager.update_all_layer_memories_batch(layer_attentions)
                
                # 记录注意力历史
                if return_attention_weights and attention_weights_history is not None:
                    attention_data = [attn.squeeze().detach() for attn in layer_attentions]
                    attention_weights_history.append({
                        'step': step,
                        'seq_len': full_input_ids.shape[1],
                        'layer_weights': attention_data
                    })
            
            # 获取最后一个token的logits
            logits = outputs.logits[0, -1, :]
            
            # 采样下一个token
            next_token_id = self._sample_next_token_fast(
                logits, temperature, do_sample, top_k, top_p
            )
            
            generated_ids.append(next_token_id)
            
            # 检查是否结束
            if next_token_id == self.tokenizer.eos_token_id and not force_exact_length:
                if verbose:
                    print(f"遇到EOS token，提前结束")
                break
            
            # 将新token添加到完整序列中
            new_token_tensor = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
            full_input_ids = torch.cat([full_input_ids, new_token_tensor], dim=1)
            
            # 时间步进
            self.memory_manager.step_all_memories()
            
            if verbose and (step + 1) % 10 == 0:
                print(f"已生成 {step + 1} tokens, 删除了 {total_removed} 个token")
        
        generation_time = time.time() - start_time
        
        # 构建完整序列用于解码
        original_input_len = input_ids.shape[1]
        decode_ids = torch.cat([
            input_ids[0],
            torch.tensor(generated_ids, device=self.device, dtype=torch.long)
        ])
        
        # 解码
        full_text = self.tokenizer.decode(decode_ids, skip_special_tokens=True)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if verbose:
            print(f"生成完成，用时: {generation_time:.2f}秒")
            print(f"总共删除了 {total_removed} 个token")
        
        # 获取token权重信息
        token_weights_info = None
        if return_attention_weights:
            token_weights_info = self.get_detailed_token_weights(decode_ids)
        
        return {
            'full_text': full_text,
            'generated_text': generated_text,
            'num_tokens': len(generated_ids),
            'generation_time': generation_time,
            'memory_stats': self.memory_manager.get_all_stats(),
            'token_weights': token_weights_info,
            'attention_weights_history': attention_weights_history,
            'total_removed_tokens': total_removed,
            'generation_mode': 'sparse_delete'
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
    ):
        """标准生成方法，使用forward实现"""
        # 准备输入
        inputs = self.prepare_input(input_text)
        input_ids = inputs["input_ids"]
        
        start_time = time.time()
        if verbose:
            print(f"开始基线生成，输入长度: {input_ids.shape[1]}")
        
        generated_ids = []
        current_ids = input_ids
        past_key_values = None
        
        for step in range(max_new_tokens):
            # 前向传播
            outputs = self.model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=True
            )
            
            # 更新past_key_values
            past_key_values = outputs.past_key_values
            
            # 获取最后一个token的logits
            logits = outputs.logits[0, -1, :]
            
            # 采样下一个token
            next_token_id = self._sample_next_token_fast(
                logits, temperature, do_sample, top_k, top_p
            )
            
            generated_ids.append(next_token_id)
            
            # 检查是否结束
            if next_token_id == self.tokenizer.eos_token_id and not force_exact_length:
                if verbose:
                    print(f"遇到EOS token，提前结束")
                break
            
            # 更新输入（只传入新token）
            current_ids = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
            
            if verbose and (step + 1) % 10 == 0:
                print(f"已生成 {step + 1} tokens")
        
        generation_time = time.time() - start_time
        
        # 构建完整序列
        full_ids = torch.cat([
            input_ids[0],
            torch.tensor(generated_ids, device=self.device, dtype=torch.long)
        ])
        
        # 解码
        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if verbose:
            print(f"生成完成，用时: {generation_time:.2f}秒")
        
        return {
            'full_text': full_text,
            'generated_text': generated_text,
            'num_tokens': len(generated_ids),
            'generation_time': generation_time,
            'generation_mode': 'baseline'
        }

    @torch.no_grad()
    def generate_soft_delete(
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
    ):
        """软删除生成方法，动态修改past_key_values"""
        # 准备输入
        inputs = self.prepare_input(input_text)
        input_ids = inputs["input_ids"]
        
        # 初始化记忆管理器
        self.memory_manager = EbbinghausMemoryManager(self.num_layers)
        
        start_time = time.time()
        if verbose:
            print(f"开始软删除生成，输入长度: {input_ids.shape[1]}")
        
        generated_ids = []
        current_ids = input_ids
        past_key_values = None
        attention_weights_history = [] if return_attention_weights else None
        
        for step in range(max_new_tokens):
            # 获取当前序列长度
            if step == 0:
                seq_len = current_ids.shape[1]
            else:
                seq_len = len(generated_ids) + input_ids.shape[1]
            
            # 快速获取和应用记忆权重
            memory_weights = self._get_memory_weights_fast(seq_len)
            
            if past_key_values is not None:
                past_key_values = self._apply_memory_to_past_kv_fast(past_key_values, memory_weights)
            
            # 前向传播
            outputs = self.model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,  # 必须为True以更新记忆
                return_dict=True
            )
            
            # 更新past_key_values
            past_key_values = outputs.past_key_values
            
            # 每个token都必须更新记忆（艾宾浩斯核心逻辑）
            if outputs.attentions is not None:
                # 高效的批量处理attention weights
                layer_attentions = [
                    layer_attn[0, :, -1, :].mean(dim=0, keepdim=True) 
                    for layer_attn in outputs.attentions
                ]
                
                # 更新所有层的记忆
                self.memory_manager.update_all_layer_memories_batch(layer_attentions)
                
                # 记录注意力历史（如果需要）
                if return_attention_weights and attention_weights_history is not None:
                    # 保持GPU tensor，避免CPU转换
                    attention_data = [attn.squeeze().detach() for attn in layer_attentions]
                    attention_weights_history.append({
                        'step': step,
                        'seq_len': seq_len,
                        'layer_weights': attention_data
                    })
            
            # 获取最后一个token的logits
            logits = outputs.logits[0, -1, :]
            
            # 采样下一个token
            next_token_id = self._sample_next_token_fast(
                logits, temperature, do_sample, top_k, top_p
            )
            
            generated_ids.append(next_token_id)
            
            # 检查是否结束
            if next_token_id == self.tokenizer.eos_token_id and not force_exact_length:
                if verbose:
                    print(f"遇到EOS token，提前结束")
                break
            
            # 更新输入（只传入新token）
            current_ids = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
            
            # 每个token都必须进行时间步进（艾宾浩斯核心）
            self.memory_manager.step_all_memories()
            
            # 清除缓存以获取最新的记忆权重
            self._cached_memory_weights = None
            self._cached_attention_mask = None
            
            if verbose and (step + 1) % 10 == 0:
                print(f"已生成 {step + 1} tokens")
        
        generation_time = time.time() - start_time
        
        # 构建完整序列
        full_ids = torch.cat([
            input_ids[0],
            torch.tensor(generated_ids, device=self.device, dtype=torch.long)
        ])
        
        # 解码
        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if verbose:
            print(f"生成完成，用时: {generation_time:.2f}秒")
        
        # 获取token权重信息
        token_weights_info = None
        if return_attention_weights:
            token_weights_info = self.get_detailed_token_weights(full_ids)
        
        return {
            'full_text': full_text,
            'generated_text': generated_text,
            'num_tokens': len(generated_ids),
            'generation_time': generation_time,
            'memory_stats': self.memory_manager.get_all_stats(),
            'token_weights': token_weights_info,
            'attention_weights_history': attention_weights_history,
            'generation_mode': 'soft_delete'
        }

    @torch.no_grad()
    def generate_sparse_attention(
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
    ):
        """稀疏注意力生成方法，动态生成attention_mask"""
        # 准备输入
        inputs = self.prepare_input(input_text)
        input_ids = inputs["input_ids"]
        
        # 初始化记忆管理器
        self.memory_manager = EbbinghausMemoryManager(self.num_layers)
        
        start_time = time.time()
        if verbose:
            print(f"开始稀疏注意力生成，输入长度: {input_ids.shape[1]}")
        
        generated_ids = []
        current_ids = input_ids
        past_key_values = None
        attention_weights_history = [] if return_attention_weights else None
        
        for step in range(max_new_tokens):
            # 获取当前序列长度
            if step == 0:
                seq_len = current_ids.shape[1]
            else:
                seq_len = len(generated_ids) + input_ids.shape[1]
            
            # 快速获取记忆权重和生成mask
            memory_weights = self._get_memory_weights_fast(seq_len)
            attention_mask = self._generate_dynamic_attention_mask_fast(seq_len, memory_weights)
            
            if verbose and step < 3 and attention_mask is not None:
                masked_count = (attention_mask == 0).sum().item()
                print(f"Step {step}: masking {masked_count}/{seq_len} positions")
            
            # 前向传播
            outputs = self.model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=True,  # 必须为True以更新记忆
                return_dict=True
            )
            
            # 更新past_key_values
            past_key_values = outputs.past_key_values
            
            # 每个token都必须更新记忆（艾宾浩斯核心逻辑）
            if outputs.attentions is not None:
                # 高效的批量处理attention weights
                layer_attentions = [
                    layer_attn[0, :, -1, :].mean(dim=0, keepdim=True) 
                    for layer_attn in outputs.attentions
                ]
                
                # 更新所有层的记忆
                self.memory_manager.update_all_layer_memories_batch(layer_attentions)
                
                # 记录注意力历史（如果需要）
                if return_attention_weights and attention_weights_history is not None:
                    # 保持GPU tensor，避免CPU转换
                    attention_data = [attn.squeeze().detach() for attn in layer_attentions]
                    attention_weights_history.append({
                        'step': step,
                        'seq_len': seq_len,
                        'layer_weights': attention_data
                    })
            
            # 获取最后一个token的logits
            logits = outputs.logits[0, -1, :]
            
            # 采样下一个token
            next_token_id = self._sample_next_token_fast(
                logits, temperature, do_sample, top_k, top_p
            )
            
            generated_ids.append(next_token_id)
            
            # 检查是否结束
            if next_token_id == self.tokenizer.eos_token_id and not force_exact_length:
                if verbose:
                    print(f"遇到EOS token，提前结束")
                break
            
            # 更新输入（只传入新token）
            current_ids = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
            
            # 每个token都必须进行时间步进（艾宾浩斯核心）
            self.memory_manager.step_all_memories()
            
            # 清除缓存以获取最新的记忆权重
            self._cached_memory_weights = None
            self._cached_attention_mask = None
            
            if verbose and (step + 1) % 10 == 0:
                print(f"已生成 {step + 1} tokens")
        
        generation_time = time.time() - start_time
        
        # 构建完整序列
        full_ids = torch.cat([
            input_ids[0],
            torch.tensor(generated_ids, device=self.device, dtype=torch.long)
        ])
        
        # 解码
        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if verbose:
            print(f"生成完成，用时: {generation_time:.2f}秒")
        
        # 获取token权重信息
        token_weights_info = None
        if return_attention_weights:
            token_weights_info = self.get_detailed_token_weights(full_ids)
        
        return {
            'full_text': full_text,
            'generated_text': generated_text,
            'num_tokens': len(generated_ids),
            'generation_time': generation_time,
            'memory_stats': self.memory_manager.get_all_stats(),
            'token_weights': token_weights_info,
            'attention_weights_history': attention_weights_history,
            'generation_mode': 'sparse_attention'
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
        generation_mode: str = "baseline",  # "baseline", "soft_delete", "sparse_attention"
        force_exact_length: bool = False,
    ):
        """统一的生成接口"""
        
        if generation_mode == "baseline":
            return self.generate_baseline(
                input_text, max_new_tokens, temperature, do_sample, top_k, top_p, verbose,
                force_exact_length
            )
        elif generation_mode == "soft_delete":
            return self.generate_soft_delete(
                input_text, max_new_tokens, temperature, do_sample, top_k, top_p, 
                return_attention_weights, verbose, force_exact_length
            )
        elif generation_mode == "sparse_attention":
            return self.generate_sparse_attention(
                input_text, max_new_tokens, temperature, do_sample, top_k, top_p, 
                return_attention_weights, verbose, force_exact_length
            )
        elif generation_mode == "sparse_delete":
            return self.generate_sparse_delete(
                input_text, max_new_tokens, temperature, do_sample, top_k, top_p, 
                return_attention_weights, verbose, force_exact_length
            )
        else:
            raise ValueError(f"Unknown generation_mode: {generation_mode}. Available modes: baseline, soft_delete, sparse_attention, sparse_delete")

    @torch.no_grad()
    def generate_standard(
            self,
            input_text: str,
            max_new_tokens: int = 100,
            temperature: float = 0.7,
            do_sample: bool = True,
            top_p: float = 0.9,
            verbose: bool = True,
            **kwargs
    ):
        """标准生成（用于对比）"""
        inputs = self.prepare_input(input_text)

        if verbose:
            print(f"标准生成开始，输入长度: {inputs['input_ids'].shape[1]}")

        start_time = time.time()
        
        # 准备生成参数，过滤掉不支持的参数
        generate_kwargs = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'do_sample': do_sample,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # 只在支持时添加top_p
        if do_sample and top_p < 1.0:
            generate_kwargs['top_p'] = top_p
            
        # 添加其他安全的kwargs
        safe_kwargs = ['num_beams', 'length_penalty', 'repetition_penalty']
        for key, value in kwargs.items():
            if key in safe_kwargs:
                generate_kwargs[key] = value
        
        outputs = self.model.generate(
            **inputs,
            **generate_kwargs
        )
        generation_time = time.time() - start_time

        if verbose:
            print(f"标准生成完成，用时: {generation_time:.2f}秒")

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
        )

        return {
            'full_text': generated_text,
            'generated_text': response_text,
            'num_tokens': outputs[0].shape[0] - inputs["input_ids"].shape[1],
            'generation_time': generation_time
        }