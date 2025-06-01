import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import matplotlib.colors as mcolors


def visualize_token_weights(token_weights_info: Dict, save_path='token_weights_detailed.png', 
                           layers_to_show: List[int] = [0, 1, -1]):
    """可视化token权重的详细信息"""
    
    if not token_weights_info or 'tokens' not in token_weights_info:
        print("没有token权重数据可以可视化")
        return
    
    tokens = token_weights_info['tokens']
    token_texts = token_weights_info['token_texts']
    layers_info = token_weights_info['layers_info']
    
    # 限制显示的token数量
    max_tokens = min(30, len(tokens))
    display_tokens = tokens[:max_tokens]
    display_texts = [text[:8] for text in token_texts[:max_tokens]]
    
    # 处理要显示的层
    num_layers = len(layers_info)
    actual_layers = []
    for layer_idx in layers_to_show:
        if layer_idx == -1:
            layer_idx = num_layers - 1
        if f'layer_{layer_idx}' in layers_info:
            actual_layers.append(layer_idx)
    
    # 创建子图
    fig, axes = plt.subplots(len(actual_layers), 3, figsize=(15, 4*len(actual_layers)))
    if len(actual_layers) == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(actual_layers)))
    
    for row_idx, layer_idx in enumerate(actual_layers):
        layer_key = f'layer_{layer_idx}'
        layer_info = layers_info[layer_key]
        
        retention_weights = layer_info['retention_weights'][:max_tokens]
        memory_strength = layer_info['memory_strength'][:max_tokens]
        time_steps = layer_info['time_steps'][:max_tokens]
        
        # 1. 记忆保持率 (Retention Weights)
        ax1 = axes[row_idx, 0]
        bars1 = ax1.bar(range(max_tokens), retention_weights, 
                       color=colors[row_idx], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_title(f'Layer {layer_idx}: Memory Retention (R = e^(-t/S))', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Retention Weight (R)')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars1, retention_weights)):
            if i % 3 == 0:  # 只显示部分标签以避免拥挤
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 记忆强度 (Memory Strength)
        ax2 = axes[row_idx, 1]
        bars2 = ax2.bar(range(max_tokens), memory_strength, 
                       color=colors[row_idx], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_title(f'Layer {layer_idx}: Memory Strength (S)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Memory Strength (S)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 时间步数 (Time Steps)
        ax3 = axes[row_idx, 2]
        bars3 = ax3.bar(range(max_tokens), time_steps, 
                       color=colors[row_idx], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.set_title(f'Layer {layer_idx}: Time Steps (t)', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Time Steps (t)')
        ax3.grid(True, alpha=0.3)
        
        # 设置x轴标签（只在最后一行）
        if row_idx == len(actual_layers) - 1:
            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel('Token Position')
                ax.set_xticks(range(0, max_tokens, max(1, max_tokens//10)))
                ax.set_xticklabels([f'{i}\n{display_texts[i]}' for i in range(0, max_tokens, max(1, max_tokens//10))], 
                                 rotation=45, ha='right', fontsize=8)
        else:
            for ax in [ax1, ax2, ax3]:
                ax.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Token权重可视化已保存到: {save_path}")


def visualize_memory_heatmap(token_weights_info: Dict, save_path='memory_heatmap.png',
                            max_layers: int = 8, max_tokens: int = 50):
    """创建记忆权重的热力图"""
    
    if not token_weights_info or 'layers_info' not in token_weights_info:
        print("没有层级数据可以可视化")
        return
    
    layers_info = token_weights_info['layers_info']
    tokens = token_weights_info['tokens']
    token_texts = token_weights_info['token_texts']
    
    # 限制显示范围
    num_layers = min(max_layers, len(layers_info))
    num_tokens = min(max_tokens, len(tokens))
    
    # 构建权重矩阵
    retention_matrix = np.zeros((num_layers, num_tokens))
    strength_matrix = np.zeros((num_layers, num_tokens))
    
    for layer_idx in range(num_layers):
        layer_key = f'layer_{layer_idx}'
        if layer_key in layers_info:
            layer_info = layers_info[layer_key]
            retention_matrix[layer_idx, :] = layer_info['retention_weights'][:num_tokens]
            strength_matrix[layer_idx, :] = layer_info['memory_strength'][:num_tokens]
    
    # 创建热力图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # 1. 记忆保持率热力图
    im1 = ax1.imshow(retention_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Memory Retention Heatmap (R = e^(-t/S))', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Layer Index')
    ax1.set_yticks(range(0, num_layers, max(1, num_layers//8)))
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Retention Weight (R)', rotation=270, labelpad=20)
    
    # 2. 记忆强度热力图
    im2 = ax2.imshow(strength_matrix, cmap='plasma', aspect='auto')
    ax2.set_title('Memory Strength Heatmap (S)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Layer Index')
    ax2.set_xlabel('Token Position')
    ax2.set_yticks(range(0, num_layers, max(1, num_layers//8)))
    
    # 设置x轴标签
    step = max(1, num_tokens//15)
    ax2.set_xticks(range(0, num_tokens, step))
    ax2.set_xticklabels([f'{i}\n{token_texts[i][:6]}' for i in range(0, num_tokens, step)], 
                       rotation=45, ha='right', fontsize=8)
    
    # 添加颜色条
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Memory Strength (S)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"记忆热力图已保存到: {save_path}")


def visualize_attention_weights(attention_weights_history: List, token_weights_info: Dict, 
                               save_path='attention_weights_detailed.png', max_layers: int = 4):
    """可视化每个token每层的attention weights - 显示生成过程中的attention演化"""
    
    if not attention_weights_history:
        print("没有attention weights历史数据可以可视化")
        return
    
    tokens = token_weights_info['tokens']
    token_texts = token_weights_info['token_texts']
    
    # 获取输入长度（第一步的序列长度）
    input_length = attention_weights_history[0]['seq_len'] - 1 if attention_weights_history else 0
    total_tokens = len(tokens)
    
    # 确定要显示的层数
    num_layers = min(max_layers, len(attention_weights_history[0]['layer_weights']) if attention_weights_history else 0)
    
    # 创建子图
    fig, axes = plt.subplots(num_layers, 1, figsize=(16, 4*num_layers))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        # 构建该层的完整attention矩阵 [query_tokens, key_tokens]
        # 初始化为NaN（表示未计算的位置）
        full_attention_matrix = np.full((total_tokens, total_tokens), np.nan)
        
        # 填充输入部分的attention（这部分我们没有，所以保持为NaN）
        
        # 填充生成过程中的attention
        for step_idx, step_data in enumerate(attention_weights_history):
            if layer_idx < len(step_data['layer_weights']):
                layer_weights = step_data['layer_weights'][layer_idx]
                # 当前生成的token位置
                current_pos = input_length + step_idx
                if current_pos < total_tokens:
                    # 填充这个token对之前所有token的attention
                    for i, weight in enumerate(layer_weights[:current_pos + 1]):
                        if i < total_tokens:
                            full_attention_matrix[current_pos, i] = weight
        
        # 创建masked array来处理NaN值
        masked_matrix = np.ma.masked_invalid(full_attention_matrix)
        
        # 创建自定义colormap，将masked区域设为灰色
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='lightgray')  # 将NaN值显示为浅灰色
        
        # 创建热力图
        im = ax.imshow(masked_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Layer {layer_idx}: Attention Weights Matrix (Query → Key)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Query Token Position')
        
        # 设置y轴标签
        if total_tokens <= 20:
            ax.set_yticks(range(0, total_tokens, max(1, total_tokens//10)))
        else:
            ax.set_yticks(range(0, total_tokens, max(1, total_tokens//20)))
        
        # 设置x轴标签
        if layer_idx == num_layers - 1:  # 只在最后一个子图显示x轴标签
            ax.set_xlabel('Key Token Position')
            if total_tokens <= 20:
                ax.set_xticks(range(0, total_tokens, max(1, total_tokens//10)))
            else:
                ax.set_xticks(range(0, total_tokens, max(1, total_tokens//20)))
        else:
            ax.set_xticks([])
        
        # 添加对角线和输入/生成分界线
        ax.axline((0, 0), (1, 1), transform=ax.transAxes, linestyle='--', color='red', alpha=0.3, linewidth=1)
        ax.axhline(y=input_length, color='white', linestyle=':', alpha=0.5, label='Input/Generate boundary')
        ax.axvline(x=input_length, color='white', linestyle=':', alpha=0.5)
        
        # 添加区域标注
        if layer_idx == 0:  # 只在第一个图上标注
            # 输入区域标注
            ax.text(input_length/2, input_length/2, 'Input tokens\n(not recorded)', 
                   ha='center', va='center', fontsize=10, color='black', alpha=0.7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # 生成区域标注  
            gen_center = (input_length + total_tokens) / 2
            ax.text(gen_center/2, gen_center, 'Generated tokens\nattention', 
                   ha='center', va='center', fontsize=10, color='white', alpha=0.9)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Attention weights矩阵可视化已保存到: {save_path}")


def visualize_attention_evolution(attention_weights_history: List, token_weights_info: Dict,
                                save_path='attention_evolution.png', layers_to_show: List[int] = None):
    """可视化生成过程中attention的演化 - 每个生成步骤显示当前token对之前所有token的attention"""
    
    if not attention_weights_history:
        print("没有attention weights历史数据可以可视化")
        return
    
    if layers_to_show is None:
        layers_to_show = [0, -1]
        
    tokens = token_weights_info['tokens']
    token_texts = token_weights_info['token_texts']
    
    # 确定要显示的层
    num_layers = len(attention_weights_history[0]['layer_weights']) if attention_weights_history else 0
    actual_layers = []
    for layer_idx in layers_to_show:
        if layer_idx == -1:
            layer_idx = num_layers - 1
        if 0 <= layer_idx < num_layers:
            actual_layers.append(layer_idx)
    
    if not actual_layers:
        print("没有有效的层可以显示")
        return
    
    # 限制显示的步骤数
    max_steps = min(6, len(attention_weights_history))
    steps_to_show = np.linspace(0, len(attention_weights_history)-1, max_steps, dtype=int)
    
    # 创建子图网格
    fig, axes = plt.subplots(len(actual_layers), max_steps, 
                           figsize=(3.5*max_steps, 3.5*len(actual_layers)))
    
    if len(actual_layers) == 1:
        axes = axes.reshape(1, -1)
    if max_steps == 1:
        axes = axes.reshape(-1, 1)
    
    for row_idx, layer_idx in enumerate(actual_layers):
        for col_idx, step_idx in enumerate(steps_to_show):
            ax = axes[row_idx, col_idx]
            
            step_data = attention_weights_history[step_idx]
            layer_weights = step_data['layer_weights'][layer_idx]
            generated_token = step_data.get('generated_token', '?')
            seq_len = step_data['seq_len']
            
            # 创建attention向量的可视化
            weights = np.array(layer_weights[:seq_len])
            
            # 条形图显示
            positions = range(len(weights))
            bars = ax.bar(positions, weights, color='steelblue', alpha=0.7)
            
            # 高亮显示高attention的token
            threshold = weights.mean() + weights.std()
            for i, (bar, w) in enumerate(zip(bars, weights)):
                if w > threshold:
                    bar.set_color('darkred')
                    bar.set_alpha(0.9)
                    if i < len(token_texts):
                        ax.text(i, w + 0.01, token_texts[i][:8], 
                               ha='center', va='bottom', fontsize=8, rotation=45)
            
            # 设置标题和标签
            ax.set_title(f'L{layer_idx} Step {step_idx}: "{generated_token}"', fontsize=10)
            ax.set_ylim(0, max(0.1, weights.max() * 1.1))
            
            if row_idx == len(actual_layers) - 1:
                ax.set_xlabel('Token Position', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel('Attention Weight', fontsize=9)
            
            # 简化x轴标签
            if len(weights) > 20:
                ax.set_xticks(range(0, len(weights), max(1, len(weights)//10)))
            
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Attention Weights Evolution During Generation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Attention演化可视化已保存到: {save_path}")


def create_comprehensive_memory_report(token_weights_info: Dict, 
                                     save_prefix='memory_analysis',
                                     attention_weights_history: List = None):
    """创建综合的记忆分析报告"""
    
    # 1. 详细的token权重图
    visualize_token_weights(token_weights_info, 
                           f'{save_prefix}_detailed.png', 
                           layers_to_show=[0, 1, 2, -1])
    
    # 2. 热力图
    visualize_memory_heatmap(token_weights_info, 
                            f'{save_prefix}_heatmap.png')
    
    # 3. Attention weights可视化
    if attention_weights_history:
        # 3a. 完整的attention矩阵
        visualize_attention_weights(attention_weights_history, token_weights_info,
                                   f'{save_prefix}_attention_matrix.png')
        
        # 3b. Attention演化过程
        visualize_attention_evolution(attention_weights_history, token_weights_info,
                                    f'{save_prefix}_attention_evolution.png')
    
    # 4. 统计摘要
    print("\n" + "="*60)
    print("记忆分析统计摘要")
    print("="*60)
    
    tokens = token_weights_info['tokens']
    layers_info = token_weights_info['layers_info']
    
    print(f"总Token数: {len(tokens)}")
    print(f"总层数: {len(layers_info)}")
    
    # 计算跨层统计
    all_retentions = []
    all_strengths = []
    all_times = []
    
    for layer_key, layer_info in layers_info.items():
        all_retentions.extend(layer_info['retention_weights'])
        all_strengths.extend(layer_info['memory_strength'])
        all_times.extend(layer_info['time_steps'])
    
    print(f"\n全局统计:")
    print(f"  平均保持率: {np.mean(all_retentions):.4f}")
    print(f"  平均记忆强度: {np.mean(all_strengths):.4f}")
    print(f"  平均时间步: {np.mean(all_times):.2f}")
    print(f"  保持率标准差: {np.std(all_retentions):.4f}")
    
    # 找出最重要和最被遗忘的token
    avg_retentions_per_token = []
    for i in range(len(tokens)):
        token_retentions = [layers_info[f'layer_{j}']['retention_weights'][i] 
                           for j in range(len(layers_info)) 
                           if i < len(layers_info[f'layer_{j}']['retention_weights'])]
        avg_retentions_per_token.append(np.mean(token_retentions))
    
    # 排序找出极值
    sorted_indices = np.argsort(avg_retentions_per_token)
    
    print(f"\n最被遗忘的Token (平均保持率最低):")
    for i in sorted_indices[:5]:
        print(f"  位置{i}: '{token_weights_info['token_texts'][i]}' - 平均保持率: {avg_retentions_per_token[i]:.4f}")
    
    print(f"\n最被记住的Token (平均保持率最高):")
    for i in sorted_indices[-5:]:
        print(f"  位置{i}: '{token_weights_info['token_texts'][i]}' - 平均保持率: {avg_retentions_per_token[i]:.4f}")
    
    print("="*60)