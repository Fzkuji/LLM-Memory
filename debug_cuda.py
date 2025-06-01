#!/usr/bin/env python
"""
CUDA调试脚本 - 诊断7B模型的CUDA错误
"""

import torch
import os
import sys

# 设置CUDA调试环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

print("=" * 60)
print("CUDA 调试信息")
print("=" * 60)

# 检查CUDA可用性
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

# 测试简单的tensor操作
print("\n测试基础CUDA操作...")
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✅ 基础CUDA操作正常")
except Exception as e:
    print(f"❌ 基础CUDA操作失败: {e}")

# 测试模型加载
print("\n测试模型加载...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_names = [
    "Qwen/Qwen2.5-0.5B-Instruct",  # 0.5B - 应该正常工作
    "Qwen/Qwen2.5-7B-Instruct"      # 7B - 可能导致问题
]

for model_name in model_names:
    print(f"\n尝试加载: {model_name}")
    try:
        # 使用不同的加载策略
        print("策略1: 默认加载...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✅ 默认加载成功")
            del model
            torch.cuda.empty_cache()
        except Exception as e1:
            print(f"❌ 默认加载失败: {e1}")
            
        print("策略2: CPU加载后移动...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            model = model.cuda()
            print("✅ CPU加载后移动成功")
            del model
            torch.cuda.empty_cache()
        except Exception as e2:
            print(f"❌ CPU加载后移动失败: {e2}")
            
        print("策略3: 8bit量化加载...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto"
            )
            print("✅ 8bit量化加载成功")
            del model
            torch.cuda.empty_cache()
        except Exception as e3:
            print(f"❌ 8bit量化加载失败: {e3}")
            
    except Exception as e:
        print(f"❌ 模型加载完全失败: {e}")
        
print("\n" + "=" * 60)
print("调试完成")
print("=" * 60)