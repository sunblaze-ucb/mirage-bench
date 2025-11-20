#!/usr/bin/env python3
"""
统计每个环境下的测试数据数量（排除OSWorld）并计算百分比
"""
import os
from pathlib import Path

def count_test_data():
    dataset_all_dir = Path("/Users/bytedance/vscode/mirage-bench/dataset_all")
    
    if not dataset_all_dir.exists():
        print(f"Directory {dataset_all_dir} does not exist")
        return
    
    # 用于存储每个环境的总计数
    env_counts = {}
    
    # 遍历第一层类别
    for category_dir in dataset_all_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        # 遍历第二层环境
        for env_dir in category_dir.iterdir():
            if not env_dir.is_dir():
                continue
                
            env_name = env_dir.name
            
            # 跳过OSWorld环境
            if env_name.lower() == "osworld":
                continue
                
            # 统计该环境下的json文件数量
            json_count = len(list(env_dir.glob("*.json")))
            
            # 累积到环境总计数
            if env_name in env_counts:
                env_counts[env_name] += json_count
            else:
                env_counts[env_name] = json_count
    
    # 计算总数
    total = sum(env_counts.values())
    
    # 打印结果
    print("\n=== 环境测试数据统计（排除OSWorld） ===")
    print("=" * 60)
    
    if total == 0:
        print("没有找到测试数据")
        return
    
    # 按环境名称排序
    for env_name in sorted(env_counts.keys()):
        count = env_counts[env_name]
        percentage = (count / total) * 100
        print(f"{env_name}: {count} 个测试文件 ({percentage:.1f}%)")
    
    print("=" * 60)
    print(f"总测试文件数: {total}")

if __name__ == "__main__":
    print("统计每个环境下的测试数据数量...")
    print("=" * 50)
    count_test_data()
    print("=" * 50)
    print("统计完成")