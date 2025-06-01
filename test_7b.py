#!/usr/bin/env python
"""
快速测试7B模型的脚本
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
sys.path.append('.')

from memollm import EbbinghausLLM

def test_7b_model():
    """测试7B模型加载和简单生成"""
    print("测试7B模型...")
    
    try:
        # 使用更简单的prompt和更少的tokens
        llm = EbbinghausLLM("Qwen/Qwen2.5-7B-Instruct")
        
        result = llm.generate(
            "你好",
            max_new_tokens=5,
            generation_mode="baseline",
            temperature=0.7
        )
        
        print("✅ 7B模型测试成功!")
        print(f"生成文本: {result['generated_text']}")
        print(f"速度: {result['num_tokens']/result['generation_time']:.2f} tokens/秒")
        
    except Exception as e:
        print(f"❌ 7B模型测试失败: {e}")
        print("\n建议解决方案:")
        print("1. 显存不足 - 使用量化版本")
        print("2. 尝试运行: python debug_cuda.py")
        print("3. 使用0.5B模型进行测试")

if __name__ == "__main__":
    test_7b_model()