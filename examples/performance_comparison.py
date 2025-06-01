import time
import torch
from memollm import EbbinghausLLM

def compare_performance():
    """对比不同生成方法的性能"""
    
    print("初始化模型...")
    llm = EbbinghausLLM("Qwen/Qwen2.5-0.5B-Instruct")
    
    test_text = "请介绍一下人工智能的发展历史，包括其主要里程碑。"
    max_tokens = 100
    
    print(f"\n测试文本: {test_text}")
    print(f"生成长度: {max_tokens} tokens")
    print("=" * 80)
    
    methods = [
        ("标准生成", lambda: llm.generate_standard(test_text, max_tokens, verbose=False)),
        ("禁用记忆", lambda: llm.generate(test_text, max_tokens, enable_memory=False, verbose=False)),
        ("优化记忆生成(无注意力)", lambda: llm.generate(test_text, max_tokens, return_attention_weights=False, verbose=False)),
        ("优化记忆生成(有注意力)", lambda: llm.generate(test_text, max_tokens, return_attention_weights=True, verbose=False))
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        print(f"\n测试: {method_name}")
        print("-" * 40)
        
        # 预热
        try:
            method_func()
        except Exception as e:
            print(f"预热失败: {e}")
            continue
        
        # 正式测试 - 多次运行取平均
        times = []
        tokens_generated = []
        
        for i in range(3):
            try:
                start_time = time.time()
                result = method_func()
                end_time = time.time()
                
                gen_time = end_time - start_time
                num_tokens = result.get('num_tokens', 0)
                
                times.append(gen_time)
                tokens_generated.append(num_tokens)
                
                print(f"  运行 {i+1}: {gen_time:.3f}s, {num_tokens} tokens, {num_tokens/gen_time:.2f} tokens/s")
                
            except Exception as e:
                print(f"  运行 {i+1} 失败: {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            avg_tokens = sum(tokens_generated) / len(tokens_generated)
            avg_speed = avg_tokens / avg_time
            
            results[method_name] = {
                'avg_time': avg_time,
                'avg_tokens': avg_tokens,
                'avg_speed': avg_speed
            }
            
            print(f"  平均: {avg_time:.3f}s, {avg_tokens:.0f} tokens, {avg_speed:.2f} tokens/s")
    
    # 性能对比总结
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)
    
    if "标准生成" in results:
        baseline = results["标准生成"]
        print(f"基准 (标准生成): {baseline['avg_speed']:.2f} tokens/s")
        print()
        
        for method_name, result in results.items():
            if method_name == "标准生成":
                continue
                
            speed_ratio = result['avg_speed'] / baseline['avg_speed']
            time_ratio = result['avg_time'] / baseline['avg_time']
            
            print(f"{method_name}:")
            print(f"  速度: {result['avg_speed']:.2f} tokens/s ({speed_ratio:.2f}x)")
            print(f"  时间: {result['avg_time']:.3f}s ({time_ratio:.2f}x)")
            print()
    
    # GPU内存使用
    if torch.cuda.is_available():
        print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == "__main__":
    compare_performance()