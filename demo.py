#!/usr/bin/env python
"""
Ebbinghaus记忆增强LLM - 核心演示
支持三种模式：baseline, soft_delete, sparse_attention
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memollm import EbbinghausLLM
import argparse

def main():
    parser = argparse.ArgumentParser(description="Ebbinghaus记忆增强LLM演示")
    parser.add_argument("--mode", choices=["baseline", "soft_delete", "sparse_attention", "sparse_delete"], 
                       default="baseline", help="生成模式")
    parser.add_argument("--prompt", type=str, default="请详细解释现代大语言模型（LLMs）的工作原理，包括从训练到推理的完整流程，涉及的关键技术，以及它们是如何理解和生成人类语言的？",
                       help="输入提示词")
    parser.add_argument("--max_tokens", type=int, default=50, 
                       help="最大生成token数")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", 
                       help="模型名称")
    parser.add_argument("--compare", action="store_true", 
                       help="对比三种模式")
    parser.add_argument("--verbose", action="store_true", 
                       help="详细输出")
    parser.add_argument("--force_exact_length", action="store_true",
                       help="强制生成固定长度的文本（忽略EOS标记）")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EBBINGHAUS 记忆增强 LLM")
    print("=" * 80)
    
    # 初始化模型
    print(f"初始化模型: {args.model}")
    llm = EbbinghausLLM(args.model)
    
    if args.compare:
        # 对比模式
        print(f"\n🔄 对比四种模式 (prompt: {args.prompt})")
        print("=" * 60)
        
        modes = ["baseline", "soft_delete", "sparse_attention", "sparse_delete"]
        results = {}
        
        for mode in modes:
            print(f"\n📍 {mode.upper()} 模式:")
            print("-" * 40)
            
            try:
                result = llm.generate(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    generation_mode=mode,
                    return_attention_weights=(mode != "baseline"),
                    verbose=args.verbose,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    force_exact_length=args.force_exact_length
                )
                
                results[mode] = result
                print(f"生成文本: {result['generated_text']}")
                print(f"Token数: {result['num_tokens']}")
                print(f"耗时: {result['generation_time']:.2f}秒")
                print(f"速度: {result['num_tokens']/result['generation_time']:.2f} tokens/秒")
                
                # 显示记忆统计和删除信息
                if mode != "baseline":
                    if 'memory_stats' in result:
                        layer_0_stats = result['memory_stats'].get('layer_0', {})
                        if layer_0_stats:
                            avg_retention = layer_0_stats.get('avg_retention', 0)
                            num_tokens = layer_0_stats.get('num_tokens', 0)
                            print(f"记忆状态: {num_tokens} tokens, 平均保持率 {avg_retention:.4f}")
                    
                    # 显示删除信息（仅sparse_delete模式）
                    if mode == "sparse_delete" and 'total_removed_tokens' in result:
                        print(f"删除token数: {result['total_removed_tokens']}")
                
            except Exception as e:
                print(f"❌ 错误: {e}")
                results[mode] = None
        
        # 性能总结
        print(f"\n{'='*60}")
        print("性能总结:")
        print(f"{'='*60}")
        
        if all(results.values()):
            for mode in modes:
                speed = results[mode]["num_tokens"] / results[mode]["generation_time"]
                quality = "✅" if len(results[mode]["generated_text"]) > 10 else "⚠️"
                print(f"{mode:15}: {speed:6.2f} tokens/秒 {quality}")
    
    else:
        # 单模式
        print(f"\n🔄 {args.mode.upper()} 模式生成")
        print(f"Prompt: {args.prompt}")
        print("=" * 60)
        
        try:
            result = llm.generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                generation_mode=args.mode,
                return_attention_weights=(args.mode != "baseline"),
                verbose=args.verbose,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                force_exact_length=args.force_exact_length
            )
            
            print(f"\n📝 生成结果:")
            print(result['generated_text'])
            
            print(f"\n📊 性能指标:")
            print(f"- Token数: {result['num_tokens']}")
            print(f"- 耗时: {result['generation_time']:.2f}秒") 
            print(f"- 速度: {result['num_tokens']/result['generation_time']:.2f} tokens/秒")
            
            # 记忆信息
            if args.mode != "baseline" and 'memory_stats' in result:
                print(f"\n🧠 记忆统计:")
                for layer_idx in [0, 11, 23]:
                    layer_key = f'layer_{layer_idx}'
                    if layer_key in result['memory_stats']:
                        stats = result['memory_stats'][layer_key]
                        if stats:
                            print(f"  Layer {layer_idx}: {stats['num_tokens']} tokens, "
                                  f"avg retention {stats['avg_retention']:.4f}")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("演示完成!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()