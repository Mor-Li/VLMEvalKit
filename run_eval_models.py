#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# python run.py --model  Claude3-7V_Sonnet_Internal Claude3-7V_Sonnet_Internal_thinking GeminiPro2-5-0506 DoubaoVL --data MSR_Bench_Circular  --reuse       

# python run.py --model  Claude3-7V_Sonnet_Internal Claude3-7V_Sonnet_Internal_thinking NVILA-8B  DoubaoVL --data MSR_Bench_Circular  --reuse   
# python run.py --model NVILA-8B --data MSR_Bench_Circular  --reuse   

# 传统的方式的提取的答案的方法
# python scripts/summarize.py --model InternVL2_5-1B InternVL2_5-2B InternVL2_5-4B InternVL2_5-8B InternVL2_5-26B InternVL2_5-38B InternVL2_5-78B InternVL3-1B InternVL3-2B InternVL3-8B InternVL3-9B InternVL3-14B InternVL3-38B InternVL3-78B Qwen2.5-VL-3B-Instruct Qwen2.5-VL-7B-Instruct Qwen2.5-VL-32B-Instruct Qwen2.5-VL-72B-Instruct llava_onevision_qwen2_0.5b_ov llava_onevision_qwen2_7b_ov llava_onevision_qwen2_72b_ov Llama-3.2-11B-Vision-Instruct deepseek_vl2_tiny deepseek_vl2_small deepseek_vl2 NVILA-8B NVILA-15B  --data MSR_Bench_Circular 

# 用api的提取出来的答案的api的model 对于claude 和gemini

# python scripts/summarize.py --model Claude3-7V_Sonnet_Internal Claude3-7V_Sonnet_Internal_thinking NVILA-8B  --data MSR_Bench_Circular 


import os
import sys
import subprocess
import time
import pandas as pd
import argparse
import random

# 项目根目录
PROJECT_DIR = "/fs-computility/mllm1/limo/workspace/VLMEvalKit"

# 要评测的数据集
DATASET = "MSR_Bench"
# DATASET = "MMBench_DEV_EN_V11"
DATASET = "MSR_Bench MMBench_DEV_EN_V11 MSR_Bench_Circular"
# DATASET = "MSR_Bench MSR_Bench_Circular"
# DATASET = "MSR_Bench_Circular"
DATASET = "MMBench_DEV_EN_V11 MSR_Bench_Circular"
DATASET = "MMBench_DEV_EN_V11"
DATASET = "MSR_Bench"
DATASET = "MSR_Bench_Circular"

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='运行评测任务')
    parser.add_argument('--debug', action='store_true', help='测试模式：仅检查配置，不提交任务')
    parser.add_argument('--queue', type=str, default='both', help='指定队列名称: mllm1, llmeval_volc 或 both（随机选择）')
    return parser.parse_args()

# 基础命令模板
CMD_TEMPLATE = (
    f"cd {PROJECT_DIR} && conda activate ENV_NAME && python run.py --data {DATASET} --model MODEL_NAME --verbose"
)

TORCH_RUN_CMD_TEMPLATE = (
    f"cd {PROJECT_DIR} && conda activate ENV_NAME && torchrun --nproc-per-node=NUMBER_OF_PROC  run.py --data {DATASET}  --model MODEL_NAME  --verbose --reuse"
)

TORCH_RUN_MODELS = [
    "NVILA-8B",
    "NVILA-15B",
    'llava_onevision_qwen2_72b_ov'
]

api_models = [
    "Claude3-7V_Sonnet_Internal",
    "Claude3-7V_Sonnet_Internal_thinking",
    "GeminiPro2-5-0506", # 需要开代理
    "DoubaoVL",
]
    
    
# 模型列表 (按照提供的清单)
MODELS = [
    # 1. InternVL2.5 系列（所有大小，包括 MPO 版本）
    "InternVL2_5-1B",
    "InternVL2_5-2B",
    "InternVL2_5-4B",
    "InternVL2_5-8B",
    "InternVL2_5-26B",
    "InternVL2_5-38B",
    "InternVL2_5-78B",

    # 2. InternVL3 系列（所有大小）
    "InternVL3-1B",
    "InternVL3-2B",
    "InternVL3-8B",
    "InternVL3-9B",
    "InternVL3-14B",
    "InternVL3-38B",
    "InternVL3-78B",

    # 3. Qwen2.5-VL 系列（所有大小, 去掉AWQ版本）
    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct",

    # 4. LLaVA-OneVision 系列（所有大小）
    "llava_onevision_qwen2_0.5b_ov",
    "llava_onevision_qwen2_7b_ov",
    "llava_onevision_qwen2_72b_ov",

    # 5. Efficient-Large-Model/VILA 系列
    "NVILA-8B",
    # "NVILA-15B", # 正在下载path
    # "Llama-3-VILA1.5-8b",
    # "VILA1.5-13b",
    # "VILA1.5-40b",

    # 6. meta-llama/Llama-3.2-11B-Vision-Instruct
    "Llama-3.2-11B-Vision-Instruct", # 正在下载path

    # 7. deepseek-ai/deepseek-vl2 系列（所有大小）
    "deepseek_vl2_tiny",
    "deepseek_vl2_small",
    "deepseek_vl2",
       
    # API models
    *api_models,
]

# MODELS = [    "InternVL3-14B","llava_onevision_qwen2_0.5b_ov","llava_onevision_qwen2_72b_ov"
# ]

# MODELS = [    
#     # "NVILA-8B", # 正在下载path
#     "NVILA-15B", # 正在下载path"
# ]
# MODELS = [    
#     "llava_onevision_qwen2_0.5b_ov"
# ]
# MODELS = [    
#     "llava_onevision_qwen2_72b_ov"
# ]


    
# MODELS = [   *api_models
# ]

# print(f"python scripts/summarize.py --model {' '.join(MODELS)} --data {DATASET}")
# # eval 的命令
# print(f"python run.py --model {' '.join(MODELS)} --data {DATASET} --mode eval --reuse")

# raise Exception("stop")
# python scripts/summarize.py --model InternVL2_5-1B InternVL2_5-2B InternVL2_5-4B InternVL2_5-8B InternVL2_5-26B InternVL2_5-38B InternVL2_5-78B InternVL3-1B InternVL3-2B InternVL3-8B InternVL3-9B InternVL3-14B InternVL3-38B InternVL3-78B Qwen2.5-VL-3B-Instruct Qwen2.5-VL-7B-Instruct Qwen2.5-VL-32B-Instruct Qwen2.5-VL-72B-Instruct llava_onevision_qwen2_0.5b_ov llava_onevision_qwen2_7b_ov llava_onevision_qwen2_72b_ov Llama-3.2-11B-Vision-Instruct deepseek_vl2_tiny deepseek_vl2_small deepseek_vl2 NVILA-8B NVILA-15B gpt-4.1-2025-04-14 DoubaoVL --data MMBench_DEV_EN_V11 MSR_Bench MSR_Bench_Circular 

# python scripts/summarize.py --model InternVL2_5-1B InternVL2_5-2B InternVL2_5-4B InternVL2_5-8B InternVL2_5-26B InternVL2_5-38B InternVL2_5-78B  --data MMBench_DEV_EN_V11 MSR_Bench MSR_Bench_Circular 

# 检查映射文件是否存在
def check_map_files():
    print("检查映射文件...")
    required_files = [
        os.path.join(PROJECT_DIR, "images/model_map/model_1g.txt"),
        os.path.join(PROJECT_DIR, "images/model_map/model_2g.txt"),
        os.path.join(PROJECT_DIR, "images/model_map/model_4g.txt"),
    ]
    for f in required_files:
        if not os.path.isfile(f):
            print(f"错误: 映射文件不存在: {f}")
            sys.exit(1)
    print("映射文件检查完成！\n")

check_map_files()

def get_env(model):
    # 依次查找1g, 2g, 4g文件
    for fname in ["model_1g.txt", "model_2g.txt", "model_4g.txt"]:
        path = os.path.join(PROJECT_DIR, "images/model_map", fname)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] == model:
                    return parts[1]
    return "th25tr446"  # 默认环境
# pip install google-generativeai

def get_gpu_count(model):
    # 如果是API模型，则不需要卡，返回0
    if model in api_models:
        return 0

    # 优先查找4g, 2g文件
    path_4g = os.path.join(PROJECT_DIR, "images/model_map/model_4g.txt")
    path_2g = os.path.join(PROJECT_DIR, "images/model_map/model_2g.txt")
    if os.path.isfile(path_4g):
        with open(path_4g, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1 and parts[0] == model:
                    return 4
    if os.path.isfile(path_2g):
        with open(path_2g, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1 and parts[0] == model:
                    return 2
    if "26B" in model or "32B" in model:
        return 2
    if any(x in model for x in ["20B", "76B", "78B", "72B"]):
        return 4
    if "15B" in model:
        return 2
    return 1

def get_all_conda_envs():
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            check=True
        )
        envs = []
        for line in result.stdout.splitlines():
            if line.strip() and not line.startswith("#"):
                env_name = line.split()[0]
                envs.append(env_name)
        return set(envs)
    except Exception as e:
        print("无法获取conda环境列表:", e)
        return set()

def check_env_exists(env, all_envs=None):
    if all_envs is None:
        all_envs = get_all_conda_envs()
    return env in all_envs

# 收集所有模型的配置信息
def collect_model_info():
    all_envs = get_all_conda_envs()
    rows = []
    for model in MODELS:
        env = get_env(model)
        gpu_count = get_gpu_count(model)
        # gpu_count = 8
        env_exists = "✓" if check_env_exists(env, all_envs) else "✗"
        rows.append({
            "模型名称": model,
            "环境": env,
            "GPU数量": gpu_count,
            "环境存在": env_exists
        })
    return pd.DataFrame(rows)

def print_model_table(df):
    print("==== 模型和环境配置检查 ====")
    print(df.to_string(index=False))
    print("")

def py_volcrun(task_cmd, num_gpus=8, queue_name="mllm1", task_name="paramnoise_task"):
    """
    Python版本的volcrun.sh
    """
    import datetime
    # 添加时间戳确保任务名唯一
    now = datetime.datetime.now()
    task_name = f"{task_name}_{now.strftime('%m%d%H%M')}"
    volc_tools_py = "/fs-computility/mllm1/limo/workspace/VLMEvalKit/volc_tools.py"
    image = "vemlp-cn-beijing.cr.volces.com/preset-images/cuda:12.4.1"
    args = [
        "python", volc_tools_py,
        "--task-cmd", task_cmd,
        "--log-level", "DEBUG",
        "--num-gpus", str(num_gpus),
        "--num-replicas", "1",
        "--task-name", task_name,
        "--queue-name", queue_name,
        "--image", image,
        "--yes"
    ]
    return subprocess.run(args, check=True)

def main():
    # 解析命令行参数
    args = parse_args()
    
    df = collect_model_info()
    print_model_table(df)

    if args.debug:
        print("测试模式：配置检查完成，未提交任务。")
        print("不使用--debug参数以实际提交任务。")
        sys.exit(0)

    # 准备队列列表
    available_queues = ["mllm1", "llmeval_volc"]
    if args.queue == 'both':
        queue_list = available_queues
    else:
        if args.queue in available_queues:
            queue_list = [args.queue]
        else:
            print(f"警告: 未知队列 {args.queue}，使用默认队列 mllm1")
            queue_list = ["mllm1"]

    print("开始提交任务...")
    for idx, row in df.iterrows():
        model = row["模型名称"]
        env = row["环境"]
        gpu_count = row["GPU数量"]
        env_exists = row["环境存在"]
        if env_exists != "✓":
            print(f"警告: 环境 {env} 不存在，请先运行克隆脚本，跳过模型 {model}")
            continue
        
        # 检查是否是需要使用torchrun的模型
        if model in TORCH_RUN_MODELS:
            # 对于torchrun模型，总是申请8张卡，并计算nproc-per-node参数
            total_gpus = 8
            nproc_per_node = total_gpus // gpu_count
            # 确保至少有1个进程
            nproc_per_node = max(1, nproc_per_node)
            
            # 对于torch run模型，将数据集按空格分开，为每个数据集提交单独的任务
            datasets = DATASET.split()
            for dataset in datasets:
                dataset_cmd = TORCH_RUN_CMD_TEMPLATE.replace("ENV_NAME", env).replace("MODEL_NAME", model).replace("NUMBER_OF_PROC", str(nproc_per_node))
                # 替换模板中的数据集部分
                dataset_cmd = dataset_cmd.replace(f"--data {DATASET}", f"--data {dataset}")
                print(f"提交torchrun任务: {model} (环境: {env}, 总GPU: {total_gpus}, nproc-per-node: {nproc_per_node}, 数据集: {dataset})")
                
                try:
                    random.seed(idx)
                    queue_name = random.choice(queue_list)
                    print(f"使用队列: {queue_name}")
                    # 替换点号为下划线，确保符合任务名称规范
                    sanitized_model_name = model.replace(".", "_")
                    task_name = f"eval_{sanitized_model_name}_{dataset}"
                    py_volcrun(dataset_cmd, num_gpus=total_gpus, queue_name=queue_name, task_name=task_name)
                except Exception as e:
                    print(f"提交任务失败: {model} 数据集 {dataset}, 错误: {e}")
                time.sleep(2)
        else:
            cmd = CMD_TEMPLATE.replace("ENV_NAME", env).replace("MODEL_NAME", model)
            print(f"提交普通任务: {model} (环境: {env}, GPU: {gpu_count})")
            
            # 直接用python版本的volcrun
            try:
                random.seed(idx)
                queue_name = random.choice(queue_list)
                print(f"使用队列: {queue_name}")
                # 替换点号为下划线，确保符合任务名称规范
                sanitized_model_name = model.replace(".", "_")
                py_volcrun(cmd, num_gpus=gpu_count, queue_name=queue_name, task_name=f"eval_{sanitized_model_name}")
            except Exception as e:
                print(f"提交任务失败: {model}, 错误: {e}")
            time.sleep(2)
    print("所有评测任务已提交")

if __name__ == "__main__":
    main()