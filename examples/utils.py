"""实验工具模块 - 提供可视化和对比测试功能"""

import time
from typing import Dict, Optional
from memollm import EbbinghausLLM
from .visualization import create_comprehensive_memory_report


def run_experiment(
    llm: EbbinghausLLM,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9,
    # 实验选项
    compare_with_standard: bool = False,
    save_visualizations: bool = False,
    visualization_prefix: str = "experiment",
    verbose: bool = True,
) -> Dict:
    """
    运行实验，可选择是否进行对比和可视化
    
    Args:
        llm: EbbinghausLLM实例
        prompt: 输入文本
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        do_sample: 是否采样
        top_p: 核采样参数
        compare_with_standard: 是否与标准生成对比
        save_visualizations: 是否保存可视化结果
        visualization_prefix: 可视化文件名前缀
        verbose: 是否打印详细信息
        
    Returns:
        包含实验结果的字典
    """
    results = {}
    
    # 运行标准生成（如果需要对比）
    if compare_with_standard:
        if verbose:
            print("\n[标准生成]")
            print(f"输入: {prompt}")
        try:
            standard_result = llm.generate_standard(
                prompt, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p
            )
            results['standard'] = standard_result
            
            if verbose:
                print(f"输出: {standard_result['generated_text']}")
                print(f"Token数: {standard_result['num_tokens']}")
                print(f"用时: {standard_result['generation_time']:.2f}秒")
        except Exception as e:
            if verbose:
                print(f"标准生成失败: {e}")
            results['standard'] = None
    
    # 运行艾宾浩斯生成
    if verbose:
        print("\n[艾宾浩斯生成]")
        print(f"输入: {prompt}")
    
    # 决定是否需要attention weights
    # 默认总是启用注意力权重以更新记忆强度
    need_attention_weights = True  # 原来是 save_visualizations
    
    try:
        ebb_result = llm.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            return_attention_weights=need_attention_weights,
            verbose=verbose
        )
        results['ebbinghaus'] = ebb_result
        
        if verbose:
            print(f"输出: {ebb_result['generated_text']}")
            print(f"Token数: {ebb_result['num_tokens']}")
            print(f"用时: {ebb_result['generation_time']:.2f}秒")
            
            # 显示记忆统计
            if ebb_result['memory_stats']:
                print(f"\n记忆统计 (前3层):")
                for layer_name in list(ebb_result['memory_stats'].keys())[:3]:
                    stats = ebb_result['memory_stats'][layer_name]
                    if stats:
                        print(f"  {layer_name}:")
                        print(f"    - Token数: {stats['num_tokens']}")
                        print(f"    - 平均强度: {stats['avg_strength']:.3f}")
                        print(f"    - 平均时间: {stats['avg_time']:.1f}")
                        print(f"    - 平均保持率: {stats['avg_retention']:.3f}")
        
        # 保存可视化（如果需要）
        if save_visualizations and ebb_result.get('token_weights'):
            if verbose:
                print(f"\n生成可视化...")
            try:
                create_comprehensive_memory_report(
                    ebb_result['token_weights'],
                    visualization_prefix,
                    ebb_result.get('attention_weights_history')
                )
                results['visualization_saved'] = True
                if verbose:
                    print(f"可视化已保存，前缀: {visualization_prefix}")
            except Exception as e:
                if verbose:
                    print(f"可视化失败: {e}")
                results['visualization_saved'] = False
                
    except Exception as e:
        if verbose:
            print(f"艾宾浩斯生成失败: {e}")
            import traceback
            traceback.print_exc()
        results['ebbinghaus'] = None
    
    # 计算对比结果（如果有）
    if compare_with_standard and results.get('standard') and results.get('ebbinghaus'):
        results['comparison'] = {
            'time_ratio': results['ebbinghaus']['generation_time'] / results['standard']['generation_time'],
            'token_diff': results['ebbinghaus']['num_tokens'] - results['standard']['num_tokens'],
            'same_output': results['ebbinghaus']['generated_text'] == results['standard']['generated_text']
        }
        
        if verbose:
            print(f"\n[对比结果]")
            print(f"时间比率: {results['comparison']['time_ratio']:.2f}x")
            print(f"Token数差异: {results['comparison']['token_diff']}")
            print(f"输出相同: {results['comparison']['same_output']}")
    
    return results


def quick_test(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "什么是机器学习？",
    max_new_tokens: int = 30,
    compare: bool = True,
    visualize: bool = False,
    verbose: bool = True
):
    """快速测试函数"""
    if verbose:
        print("=== 艾宾浩斯记忆增强LLM测试 ===")
        print(f"模型: {model_name}")
        print(f"Prompt: {prompt}")
        print(f"最大Token数: {max_new_tokens}")
        print("=" * 50)
    
    # 初始化模型
    llm = EbbinghausLLM(model_name)
    
    # 运行实验
    results = run_experiment(
        llm=llm,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        compare_with_standard=compare,
        save_visualizations=visualize,
        visualization_prefix=f"quick_test_{int(time.time())}",
        verbose=verbose
    )
    
    return results


def batch_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompts: list = None,
    max_new_tokens: int = 50,
    compare: bool = True,
    visualize_first_n: int = 0,
    verbose: bool = True
):
    """批量实验函数"""
    if prompts is None:
        prompts = [
            "什么是机器学习？",
            "Python的优点是什么？",
            "解释一下深度学习。"
        ]
    
    if verbose:
        print(f"=== 批量实验 ({len(prompts)} 个测试) ===\n")
    
    # 初始化模型
    llm = EbbinghausLLM(model_name)
    
    all_results = []
    
    for i, prompt in enumerate(prompts):
        if verbose:
            print(f"\n{'='*70}")
            print(f"测试 {i+1}/{len(prompts)}: {prompt}")
            print('='*70)
        
        # 决定是否为这个测试保存可视化
        save_viz = i < visualize_first_n
        
        results = run_experiment(
            llm=llm,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            compare_with_standard=compare,
            save_visualizations=save_viz,
            visualization_prefix=f"batch_test_{i+1}",
            verbose=verbose
        )
        
        all_results.append({
            'prompt': prompt,
            'results': results
        })
    
    # 汇总统计
    if verbose and compare:
        print(f"\n{'='*70}")
        print("汇总统计")
        print('='*70)
        
        time_ratios = []
        for item in all_results:
            if item['results'].get('comparison'):
                time_ratios.append(item['results']['comparison']['time_ratio'])
        
        if time_ratios:
            import numpy as np
            print(f"平均时间比率: {np.mean(time_ratios):.2f}x")
            print(f"时间比率标准差: {np.std(time_ratios):.2f}")
    
    return all_results