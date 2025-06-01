#!/usr/bin/env python
"""
MemOLLM 运行脚本
用于从根目录运行各种功能
"""

import sys
import os

# 确保可以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("MemOLLM - 记忆增强语言模型")
    print("=" * 50)
    print("\n可用命令:")
    print("1. python run.py demo - 运行演示")
    print("2. python run.py test [prompt] [--visualize] - 运行测试（可选可视化）")
    print("3. python run.py inference <prompt> - 推理生成")
    print("4. python run.py performance - 性能测试")
    print("5. python run.py quick - 快速测试")
    print("\n示例:")
    print('  python run.py test --visualize')
    print('  python run.py test "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?" --visualize')
    
    if len(sys.argv) < 2:
        print("\n请提供命令参数")
        return
    
    command = sys.argv[1]
    
    if command == "demo":
        from scripts.demo import test_with_options
        test_with_options()
    
    elif command == "test":
        # 检查是否有 --visualize 参数
        visualize = "--visualize" in sys.argv or "-v" in sys.argv
        
        # 获取prompt（过滤掉 --visualize）
        args = [arg for arg in sys.argv[2:] if arg not in ["--visualize", "-v"]]
        prompt = args[0] if args else "什么是机器学习？"
        
        from scripts.utils import quick_test
        quick_test(prompt=prompt, visualize=visualize)
    
    elif command == "inference":
        if len(sys.argv) < 3:
            print("请提供推理文本")
            return
        from scripts.inference import main as inference_main
        # 修改sys.argv以传递参数
        sys.argv = ['inference.py'] + sys.argv[2:]
        inference_main()
    
    elif command == "performance":
        from scripts.test_performance import test_performance
        test_performance()
    
    elif command == "quick":
        from scripts.quick_test_simple import quick_performance_test
        quick_performance_test()
    
    else:
        print(f"未知命令: {command}")

if __name__ == "__main__":
    main()