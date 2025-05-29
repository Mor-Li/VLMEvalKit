#!/usr/bin/env python3
"""
计算文件MD5的工具脚本
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from vlmeval.smp.file import md5

def calculate_file_md5(file_path):
    """计算指定文件的MD5值"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    print(f"🔍 正在计算文件MD5: {file_path}")
    md5_value = md5(file_path)
    print(f"✅ MD5值: {md5_value}")
    return md5_value

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python calculate_md5.py <文件路径>")
        print("示例: python calculate_md5.py /path/to/MMSI_bench.tsv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    calculate_file_md5(file_path) 