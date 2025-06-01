#!/usr/bin/env python
"""
Ebbinghaus记忆增强LLM - 调试版本
包含CUDA错误诊断和内存管理优化
"""

import sys
import os

# 设置CUDA调试环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from memollm import EbbinghausLLM
import argparse
import traceback

def check_cuda_status():
    """检查CUDA状态"""
    print("=" * 60)
    print("CUDA 状态检查")
    print("=" * 60)
    
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
        
        # 检查显存
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_cached = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"\n显存信息:")
        print(f"总显存: {mem_total:.2f} GB")
        print(f"已分配: {mem_allocated:.2f} GB")
        print(f"已缓存: {mem_cached:.2f} GB")
        print(f"可用显存: {mem_total - mem_cached:.2f} GB")
        
        # 建议检查
        if mem_total < 8.0:
            print("⚠️  警告: 显存可能不足以运行7B模型")
        if mem_total - mem_cached < 6.0:
            print("⚠️  警告: 可用显存不足，建议使用量化加载")

def main():
    parser = argparse.ArgumentParser(description="Ebbinghaus记忆增强LLM调试版")
    parser.add_argument("--mode", choices=["baseline", "soft_delete", "sparse_attention"], 
                       default="baseline", help="生成模式")
    parser.add_argument("--prompt", type=str, default="请简单介绍什么是人工智能？",
                       help="输入提示词")
    parser.add_argument("--max_tokens", type=int, default=20, 
                       help="最大生成token数")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", 
                       help="模型名称")
    parser.add_argument("--compare", action="store_true", 
                       help="对比三种模式")
    parser.add_argument("--verbose", action="store_true", 
                       help="详细输出")
    parser.add_argument("--force_exact_length", action="store_true",
                       help="强制生成固定长度的文本")
    parser.add_argument("--check_cuda", action="store_true",
                       help="检查CUDA状态")
    
    args = parser.parse_args()
    
    if args.check_cuda:
        check_cuda_status()
        return
    
    print("=" * 80)
    print("EBBINGHAUS 记忆增强 LLM - 调试版")
    print("=" * 80)
    
    # CUDA状态检查
    check_cuda_status()
    
    # 初始化模型
    print(f"\n初始化模型: {args.model}")
    try:
        llm = EbbinghausLLM(args.model)
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        print("\n建议:")
        print("1. 使用更小的模型 (如 Qwen/Qwen2.5-0.5B-Instruct)")
        print("2. 或添加 --model 参数指定更小的模型")
        print("3. 确保有足够的显存")
        return
    
    if args.compare:
        # 对比模式
        print(f"\n🔄 对比三种模式 (prompt: {args.prompt})")
        print("=" * 60)
        
        modes = ["baseline", "soft_delete", "sparse_attention"]
        results = {}
        
        for mode in modes:
            print(f"\n📍 {mode.upper()} 模式:")
            print("-" * 40)
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
                
                # 显示记忆统计
                if mode != "baseline" and 'memory_stats' in result:
                    layer_0_stats = result['memory_stats'].get('layer_0', {})
                    if layer_0_stats:
                        avg_retention = layer_0_stats.get('avg_retention', 0)
                        num_tokens = layer_0_stats.get('num_tokens', 0)
                        print(f"记忆状态: {num_tokens} tokens, 平均保持率 {avg_retention:.4f}")
                
            except Exception as e:
                print(f"❌ 错误: {e}")
                print(f"详细错误信息:")
                traceback.print_exc()
                results[mode] = None
                
                # 提供诊断信息
                if "CUDA" in str(e):
                    print("\nCUDA错误诊断:")
                    print("1. 检查显存是否足够")
                    print("2. 尝试使用更小的max_tokens")
                    print("3. 考虑使用量化模型")
        
        # 性能总结
        print(f"\n{'='*60}")
        print("性能总结:")
        print(f"{'='*60}")
        
        successful_results = {k: v for k, v in results.items() if v is not None}
        if successful_results:
            for mode in modes:
                if mode in successful_results:
                    speed = successful_results[mode]["num_tokens"] / successful_results[mode]["generation_time"]
                    quality = "✅" if len(successful_results[mode]["generated_text"]) > 10 else "⚠️"
                    print(f"{mode:15}: {speed:6.2f} tokens/秒 {quality}")
                else:
                    print(f"{mode:15}: ❌ 失败")
    
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
            print(f"详细错误信息:")
            traceback.print_exc()
            
            # 提供诊断信息
            if "CUDA" in str(e):
                print("\nCUDA错误诊断:")
                print("1. 检查显存是否足够")
                print("2. 尝试使用更小的max_tokens")
                print("3. 考虑使用量化模型")
                print("4. 运行: python debug_cuda.py 进行详细诊断")
    
    print(f"\n{'='*80}")
    print("演示完成!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()