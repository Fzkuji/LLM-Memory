#!/usr/bin/env python3
"""
更公平的性能对比：标准生成 vs 记忆增强生成
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimplifiedGenerate:
    """简化的生成基准，用于公平对比"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    @torch.no_grad()
    def generate_baseline(self, input_text, max_new_tokens=50, temperature=0.7):
        """最小化的生成循环，作为公平基准"""
        # 准备输入
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        generated_ids = []
        current_ids = input_ids
        past_key_values = None
        
        start_time = time.perf_counter()
        
        # 简单的生成循环
        for _ in range(max_new_tokens):
            # 前向传播
            outputs = self.model(
                current_ids[:, -1:] if past_key_values else current_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            
            # 简单采样
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            generated_ids.append(next_token_id)
            current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]], device=self.device)], dim=1)
        
        end_time = time.perf_counter()
        
        return {
            'num_tokens': len(generated_ids),
            'time': end_time - start_time,
            'generated_ids': generated_ids
        }

def fair_performance_comparison():
    """公平的性能对比测试"""
    
    print("🔬 公平性能对比测试")
    print("=" * 60)
    
    # 加载模型
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 创建简化生成器
    simple_gen = SimplifiedGenerate(model, tokenizer)
    
    # 测试文本
    test_text = "解释一下什么是机器学习"
    max_tokens = 50
    
    print(f"\n测试配置:")
    print(f"- 文本: {test_text}")
    print(f"- 生成长度: {max_tokens} tokens")
    print(f"- 使用相同的模型和参数")
    print(f"- 都是简单的Python循环实现")
    
    # 1. 测试简化基准
    print(f"\n📊 测试1: 简化基准生成")
    print("-" * 40)
    
    results_baseline = []
    for i in range(3):
        result = simple_gen.generate_baseline(test_text, max_tokens)
        speed = result['num_tokens'] / result['time']
        results_baseline.append(speed)
        print(f"运行 {i+1}: {result['time']:.3f}s, {speed:.2f} tokens/s")
    
    avg_baseline = sum(results_baseline) / len(results_baseline)
    print(f"平均: {avg_baseline:.2f} tokens/s")
    
    # 2. 测试我们的记忆生成（无注意力）
    print(f"\n📊 测试2: 记忆生成（无注意力）")
    print("-" * 40)
    
    from ebbinghaus_llm import EbbinghausLLM
    ebbinghaus_llm = EbbinghausLLM(model_name)
    
    results_memory = []
    for i in range(3):
        result = ebbinghaus_llm.generate(
            test_text, 
            max_new_tokens=max_tokens,
            return_attention_weights=False,
            verbose=False
        )
        speed = result['num_tokens'] / result['generation_time']
        results_memory.append(speed)
        print(f"运行 {i+1}: {result['generation_time']:.3f}s, {speed:.2f} tokens/s")
    
    avg_memory = sum(results_memory) / len(results_memory)
    print(f"平均: {avg_memory:.2f} tokens/s")
    
    # 3. 分析结果
    print(f"\n📈 性能分析")
    print("=" * 60)
    print(f"简化基准: {avg_baseline:.2f} tokens/s")
    print(f"记忆生成: {avg_memory:.2f} tokens/s")
    print(f"性能差异: {(avg_memory/avg_baseline - 1)*100:+.1f}%")
    
    if avg_memory < avg_baseline:
        print(f"\n⚠️  记忆系统引入了 {(1 - avg_memory/avg_baseline)*100:.1f}% 的额外开销")
        print("这主要来自:")
        print("- 记忆权重计算")
        print("- Cache权重应用") 
        print("- 记忆管理逻辑")
    else:
        print(f"\n✅ 记忆系统在相同条件下表现更好！")
    
    # 4. 内存效率分析
    print(f"\n💾 理论内存效率分析")
    print("=" * 60)
    print("软删除策略:")
    print("- 内存使用: 相同（仍存储完整KV Cache）")
    print("- 计算复杂度: 相同（仍是完整矩阵运算）")
    print("- 实际效果: 降低某些位置的影响权重")
    print("\n如果要真正提高效率，需要:")
    print("- 硬删除：物理上移除KV Cache的某些位置")
    print("- 稀疏注意力：跳过某些计算")
    print("- 量化压缩：减少存储精度")

if __name__ == "__main__":
    fair_performance_comparison()