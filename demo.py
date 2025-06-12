#!/usr/bin/env python
"""
Ebbinghaus记忆增强LLM - 核心演示
支持两种模式：baseline, memory_enhanced
memory_enhanced模式现在使用自然记忆权重（0-1），仅在权重低于阈值时进行硬删除
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memollm import EbbinghausLLM
import argparse

def main():
    parser = argparse.ArgumentParser(description="Ebbinghaus记忆增强LLM演示")
    parser.add_argument("--mode", choices=["baseline", "memory_enhanced"], 
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
        print(f"\n🔄 对比两种模式 (prompt: {args.prompt})")
        print("=" * 60)
        
        modes = ["baseline", "memory_enhanced"]
        results = {}
        
        for mode in modes:
            print(f"\n📍 {mode.upper()} 模式:")
            print("-" * 40)
            
            try:
                # 为memory_enhanced模式使用更合理的删除阈值
                kwargs = {}
                if mode == "memory_enhanced":
                    kwargs['hard_delete_threshold'] = 0.05  # 使用更合理的阈值来触发删除
                
                result = llm.generate(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    generation_mode=mode,
                    return_attention_weights=(mode != "baseline"),
                    verbose=args.verbose,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    force_exact_length=args.force_exact_length,
                    **kwargs
                )
                
                results[mode] = result
                print(f"生成文本: {result['generated_text']}")
                print(f"Token数: {result['num_tokens']}")
                print(f"耗时: {result['generation_time']:.2f}秒")
                print(f"速度: {result['num_tokens']/result['generation_time']:.2f} tokens/秒")
                
                # 显示记忆统计和删除信息
                if mode != "baseline":
                    if 'memory_stats' in result:
                        # 收集所有层的统计信息
                        all_retentions = []
                        all_token_counts = []
                        
                        for layer_idx in range(24):  # 假设24层
                            layer_key = f'layer_{layer_idx}'
                            if layer_key in result['memory_stats']:
                                stats = result['memory_stats'][layer_key]
                                if stats and 'num_tokens' in stats and 'avg_retention' in stats:
                                    all_retentions.append(stats['avg_retention'])
                                    all_token_counts.append(stats['num_tokens'])
                        
                        # 计算整体统计
                        if all_retentions:
                            avg_retention = sum(all_retentions) / len(all_retentions)
                            avg_tokens = sum(all_token_counts) / len(all_token_counts)
                            print(f"记忆状态: 平均{avg_tokens:.1f} tokens, 平均保持率 {avg_retention:.4f}")
                    
                    # 显示memory_enhanced模式的cache删除率
                    if mode == "memory_enhanced":
                        cache_deletion_percentage = result.get('cache_deletion_percentage', 0)
                        total_expected_tokens = result.get('total_expected_tokens', 0)
                        total_actual_tokens = result.get('total_actual_tokens', 0)
                        
                        if cache_deletion_percentage > 0:
                            total_cache_deletions = result.get('total_cache_deletions', 0)
                            total_cache_entries = result.get('total_cache_entries', 0)
                            min_cache = result.get('min_cache_length', 0)
                            max_cache = result.get('max_cache_length', 0)
                            print(f"🗑️  Cache删除率: {cache_deletion_percentage:.2f}% ({total_cache_deletions}/{total_cache_entries})")
                            if min_cache != max_cache:
                                print(f"🔀  每层独立: 最短{min_cache}个token, 最长{max_cache}个token")
                        
                        print(f"📊 Tokens: 期望{total_expected_tokens}, 实际平均{total_actual_tokens:.1f}")
                        
                        # 显示删除事件数量
                        deletion_events = result.get('deletion_events', [])
                        if deletion_events:
                            print(f"📋 删除事件: {len(deletion_events)} 次")
                    
                
            except Exception as e:
                print(f"❌ {mode}模式错误: {e}")
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

        # 为memory_enhanced模式使用更合理的删除阈值
        kwargs = {}
        if args.mode == "memory_enhanced":
            kwargs['hard_delete_threshold'] = 0.05  # 测试阈值为0的情况

        result = llm.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            generation_mode=args.mode,
            return_attention_weights=(args.mode != "baseline"),
            verbose=args.verbose,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            force_exact_length=args.force_exact_length,
            **kwargs
        )

        print(f"\n📝 生成结果:")
        print(result['generated_text'])

        print(f"\n📊 性能指标:")
        print(f"- Token数: {result['num_tokens']}")
        print(f"- 耗时: {result['generation_time']:.2f}秒")
        print(f"- 速度: {result['num_tokens']/result['generation_time']:.2f} tokens/秒")

        # 记忆信息 - 直接基于cache长度统计
        if args.mode == "memory_enhanced":
            layer_cache_lengths = result.get('layer_cache_lengths', [])
            if layer_cache_lengths:
                print(f"\n🧠 记忆统计 (基于实际cache长度):")

                # 计算整体统计
                avg_length = sum(layer_cache_lengths) / len(layer_cache_lengths)
                min_length = min(layer_cache_lengths)
                max_length = max(layer_cache_lengths)

                print(f"  平均cache长度: {avg_length:.1f} tokens")
                print(f"  cache长度范围: {min_length}-{max_length} tokens")

                # 找出cache最短和最长的层
                min_idx = layer_cache_lengths.index(min_length)
                max_idx = layer_cache_lengths.index(max_length)

                print(f"  cache最短层: Layer {min_idx} ({min_length} tokens)")
                print(f"  cache最长层: Layer {max_idx} ({max_length} tokens)")

                # 显示所有层的cache长度
                print(f"\n  各层cache长度详情:")
                for i in range(0, len(layer_cache_lengths), 6):  # 每行显示6层
                    layer_group = []
                    for j in range(i, min(i+6, len(layer_cache_lengths))):
                        layer_group.append(f"L{j}:{layer_cache_lengths[j]}")
                    print(f"    {' '.join(layer_group)}")

        # 显示memory_enhanced模式的详细cache删除信息
        if args.mode == "memory_enhanced":
            cache_deletion_percentage = result.get('cache_deletion_percentage', 0)
            total_expected_tokens = result.get('total_expected_tokens', 0)
            total_actual_tokens = result.get('total_actual_tokens', 0)
            layer_cache_lengths = result.get('layer_cache_lengths', [])

            print(f"\n📊 Token统计:")
            print(f"  期望tokens: {total_expected_tokens}")
            print(f"  实际平均tokens: {total_actual_tokens:.1f}")

            # 显示每层cache长度的分布
            if layer_cache_lengths:
                min_cache = result.get('min_cache_length', min(layer_cache_lengths))
                max_cache = result.get('max_cache_length', max(layer_cache_lengths))
                print(f"  各层cache长度: 最小{min_cache}, 最大{max_cache}")

                # 显示前几层和后几层的cache长度
                if len(layer_cache_lengths) >= 6:
                    print(f"  前3层cache: {layer_cache_lengths[:3]}")
                    print(f"  后3层cache: {layer_cache_lengths[-3:]}")

                # 显示缓存不一致性（每层独立管理的特征）
                unique_lengths = len(set(layer_cache_lengths))
                if min_cache != max_cache:
                    print(f"  🔀 每层独立管理: {unique_lengths}种不同长度, 范围{min_cache}-{max_cache}")
                else:
                    print(f"  ⚠️  所有层长度相同({min_cache}) - 可能需要更高删除阈值或transformers兼容性限制")

            if cache_deletion_percentage > 0:
                print(f"\n🗑️  Cache删除统计:")
                total_cache_deletions = result.get('total_cache_deletions', 0)
                total_cache_entries = result.get('total_cache_entries', 0)
                min_cache = result.get('min_cache_length', 0)
                max_cache = result.get('max_cache_length', 0)

                print(f"  Cache删除率: {cache_deletion_percentage:.2f}%")
                print(f"  Cache统计: {total_cache_deletions}/{total_cache_entries} 条目被删除")
                if min_cache != max_cache:
                    print(f"  每层独立管理: 最短{min_cache}个token, 最长{max_cache}个token")

                # 显示删除事件详情
                deletion_events = result.get('deletion_events', [])
                if deletion_events:
                    print(f"  删除事件: {len(deletion_events)} 次")
                    total_deletions = sum(e.get('total_deletions', 0) for e in deletion_events)
                    avg_cache_per_event = total_cache_deletions / len(deletion_events) if deletion_events else 0
                    print(f"  总共删除cache条目: {total_deletions} 个")
                    print(f"  平均每次删除cache: {avg_cache_per_event:.1f} 条目")

                    # 显示每层删除差异
                    print(f"  🔍 每层删除差异:")
                    total_per_layer = [0] * 24  # 假设24层
                    for event in deletion_events:
                        per_layer = event.get('per_layer_deletions', [])
                        for i, count in enumerate(per_layer):
                            if i < len(total_per_layer):
                                total_per_layer[i] += count

                    # 显示差异最大的几层
                    layer_diffs = [(i, count) for i, count in enumerate(total_per_layer) if count > 0]
                    layer_diffs.sort(key=lambda x: x[1], reverse=True)

                    if layer_diffs:
                        top_layers = layer_diffs[:5]  # 显示删除最多的5层
                        layer_info = [f"L{idx}:{count}" for idx, count in top_layers]
                        print(f"    删除最多的层: {', '.join(layer_info)}")
            else:
                print(f"\n🗑️  没有触发cache删除（所有token权重高于阈值）")

    print(f"\n{'='*80}")
    print("演示完成!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()