#!/usr/bin/env python3
"""纯推理脚本 - 不包含任何可视化或对比功能"""

import argparse
from memollm import EbbinghausLLM


def main():
    parser = argparse.ArgumentParser(description='艾宾浩斯记忆增强LLM推理')
    parser.add_argument('prompt', type=str, help='输入文本')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help='模型名称')
    parser.add_argument('--max-tokens', type=int, default=100, help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.7, help='采样温度')
    parser.add_argument('--top-p', type=float, default=0.9, help='核采样参数')
    parser.add_argument('--no-sample', action='store_true', help='使用贪婪解码')
    parser.add_argument('--quiet', action='store_true', help='只输出生成的文本')
    
    args = parser.parse_args()
    
    # 初始化模型
    if not args.quiet:
        print(f"加载模型: {args.model}")
    
    llm = EbbinghausLLM(args.model)
    
    # 生成文本
    result = llm.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=not args.no_sample,
        top_p=args.top_p,
        return_attention_weights=False,  # 不需要权重信息
        verbose=not args.quiet
    )
    
    # 输出结果
    if args.quiet:
        print(result['generated_text'])
    else:
        print(f"\n生成文本: {result['generated_text']}")
        print(f"\nToken数: {result['num_tokens']}")
        print(f"用时: {result['generation_time']:.2f}秒")


if __name__ == "__main__":
    main()