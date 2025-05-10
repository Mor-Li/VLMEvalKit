import os
import re
import pandas as pd
import os.path as osp
import json
import numpy as np
import warnings
from tqdm import tqdm
from .image_mcq import ImageMCQDataset
from .utils import DEBUG_MESSAGE, build_judge
from ..smp import LMUDataRoot, file_size, load, dump, decode_base64_to_image_file, listinstr, gpt_key_set
import string
import glob

class MSRBenchDataset(ImageMCQDataset):
    """
    MSR Bench Dataset class for multiple-choice questions with multiple images.
    支持多图片的多选题评测数据集，图片以JSON数组格式存储在image字段中。
    """
    TYPE = 'MCQ'
    
    # 使用本地TSV文件路径，不进行网络下载
    # MSR_BENCH_TSV = '/fs-computility/mllm1/shared/LMUData/msr_bench_fanal_version_5_5_cat_option_to_qs.tsv'
    MSR_BENCH_TSV = '/fs-computility/mllm1/shared/LMUData/msr_bench_fanal_version_5_5_cat_option_to_qs_fixed.tsv'
    # MSR_BENCH_TSV = '/fs-computility/mllm1/shared/LMUData/msr_bench_en_3_sample_from_fanal_version_cat_option_to_qs.tsv'
    
    # DATASET_URL = {
    #     'MSR_Bench': 'file:///fs-computility/mllm1/shared/LMUData/msr_bench_cat_option_to_qs.tsv'
    # }
    DATASET_MD5 = {
        'MSR_Bench': ''  # 如果有MD5校验，可以在这里添加
    }
    
    @classmethod
    def supported_datasets(cls):
        return ['MSR_Bench']
    
    def load_data(self, dataset):
        """
        重写load_data方法，直接加载本地TSV文件
        """
        if dataset == 'MSR_Bench':
            tsv_path = self.__class__.MSR_BENCH_TSV
            if not osp.exists(tsv_path):
                raise FileNotFoundError(f"MSR_Bench TSV文件不存在: {tsv_path}")
            
            data = pd.read_csv(tsv_path, sep='\t')
            # 确保必要的列存在
            assert 'index' in data.columns, "TSV文件缺少'index'列"
            assert 'question' in data.columns, "TSV文件缺少'question'列"
            
            return data
        else:
            # 对于其他数据集，使用父类的方法
            return super().load_data(dataset)
    
    def dump_image(self, line):
        """
        处理图片字段，支持多张图片。
        如果image字段是JSON数组格式，则解析并处理每张图片。
        """
        # 处理image_path字段
        if 'image_path' in line and isinstance(line['image_path'], str):
            tgt_path = line['image_path']
            if not isinstance(tgt_path, list):
                tgt_path = [tgt_path]
            return tgt_path
        
        # 处理image字段    
        if 'image' in line:
            # 获取image字段的值，确保它是一个字符串
            img_field = line['image']
            if isinstance(img_field, (pd.Series, np.ndarray)):
                # 如果是Series或数组，取第一个元素
                if len(img_field) > 0:
                    img_str = img_field.iloc[0] if hasattr(img_field, 'iloc') else img_field[0]
                else:
                    return None
            else:
                img_str = img_field
                
            # 处理已经是列表类型的图片数据
            if isinstance(img_str, list):
                paths = []
                # 处理每张图片
                for i, img_base64 in enumerate(img_str):
                    img_path = os.path.join(self.img_root, f"{line['index']}_{i}.jpg")
                    decode_base64_to_image_file(img_base64, img_path)
                    paths.append(img_path)
                return paths
                
            # 确保img_str是字符串
            if not isinstance(img_str, str):
                return None
                
            # 检查是否是JSON数组格式的多图片
            if img_str.startswith('[') and img_str.endswith(']'):
                try:
                    # 尝试解析JSON数组
                    img_list = json.loads(img_str)
                    paths = []
                    
                    # 处理每张图片
                    for i, img_base64 in enumerate(img_list):
                        img_path = os.path.join(self.img_root, f"{line['index']}_{i}.jpg")
                        decode_base64_to_image_file(img_base64, img_path)
                        paths.append(img_path)
                    
                    return paths
                except json.JSONDecodeError:
                    # 如果解析失败，按单图片处理
                    pass
            
            # 单图片处理
            img_path = os.path.join(self.img_root, f"{line['index']}.jpg")
            decode_base64_to_image_file(img_str, img_path)
            return img_path
            
        return None
    
    def build_prompt(self, line):
        """
        构建提示，支持多图片输入。
        在新的TSV格式中，选项已经包含在question字段中，不需要再从单独的列提取选项。
        """
        if isinstance(line, int):
            line = self.data.iloc[line]
            
        # 处理图片（支持多图）
        tgt_path = self.dump_image(line)
        
        # 构建文本提示 - 在新格式中，question字段已经包含了选项，不需要再拼接
        question = line['question']
        # 添加post_prompt，引导模型以正确格式回答
        post_prompt = "Answer with the option's letter from the given choices directly. Enclose the option's letter within ``."
        prompt = f'{question}\n{post_prompt}'
        
        # 构建多模态消息
        msgs = []
        if isinstance(tgt_path, list):
            # 处理多张图片
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            # 处理单张图片
            msgs = [dict(type='image', value=tgt_path)]
        
        # 添加文本提示
        msgs.append(dict(type='text', value=prompt))
        return msgs
    
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        评估模型预测结果。
        使用extract_single_choice_with_word_boundary函数提取预测的选项。
        """
        from ..smp.file import load
        data = load(eval_file)
        
        # 确保预测值和答案都是字符串类型
        data['prediction'] = [str(x) if x is not None else None for x in data['prediction']]
        data['answer'] = [str(x) if x is not None else None for x in data['answer']]
        
        # 计算准确率
        correct = 0
        total = 0
        
        # 添加预测结果列
        data['extracted_pred'] = None
        data['score'] = 0.0
        
        for idx, row in data.iterrows():
            gt = row['answer']
            pred = row['prediction']
            
            # 使用提供的函数提取选项
            extracted_pred = cls.extract_single_choice_with_word_boundary(pred)
            
            # 记录提取的预测结果
            data.at[idx, 'extracted_pred'] = extracted_pred
            
            # 如果提取到了有效选项，进行得分计算
            if extracted_pred is not None:
                answer = gt.lower().replace("\n", " ").strip()
                predict = extracted_pred.lower().replace("\n", " ").strip()
                try:
                    if answer == predict[0]:
                        data.at[idx, 'score'] = 1.0
                        correct += 1
                    elif predict[0] == "(" and answer == predict[1]:
                        data.at[idx, 'score'] = 1.0
                        correct += 1
                    elif predict[0:7] == "option " and answer == predict[7]:
                        data.at[idx, 'score'] = 1.0
                        correct += 1
                    elif predict[0:14] == "the answer is " and answer == predict[14]:
                        data.at[idx, 'score'] = 1.0
                        correct += 1
                except Exception as e:
                    pass
                
            total += 1
            
        accuracy = correct / total if total > 0 else 0
        print("MSR_Bench 评测结果：")
        print(f"总样本数: {total}")
        print(f"正确样本数: {correct}")
        print(f"准确率: {accuracy:.2%}")
        
        # 分类别计算准确率
        category_acc = {}
        if 'category' in data.columns:
            for category in data['category'].unique():
                cat_data = data[data['category'] == category]
                cat_correct = sum(cat_data['score'] == 1.0)
                cat_total = len(cat_data)
                        
                category_acc[category] = cat_correct / cat_total if cat_total > 0 else 0
                
        results = {
            'overall': accuracy,
            'categories': category_acc
        }
        
        # 保存详细评测结果
        score_file = eval_file.replace('.xlsx', '_score.xlsx')
        data.to_excel(score_file)
        
        return pd.DataFrame([results])
    
    @staticmethod
    def extract_single_choice_with_word_boundary(pred):
        """
        从预测文本中提取选项，并与正确答案比较。
        返回提取到的选项，如果没有找到则返回None。
        """
        if pred is None:
            return None
            
        # 确保pred是字符串类型
        try:
            pred = str(pred)
        except:
            return None
            
        pattern_1 = r'``([^`]*)``'
        match = re.search(pattern_1, pred)
        if match:
            pred = match.group(1)  # 提取反引号之间的内容

        pattern_2 = r'`([^`]*)`'
        match = re.search(pattern_2, pred)
        if match:
            pred = match.group(1)  # 提取双反引号之间的内容

        pattern_3 = r'\b[A-D]\b(?!\s[a-zA-Z])'
        match = re.search(pattern_3, pred)
        if match:
            pred = match.group()  # 提取孤立的大写字母（排除"A bike"，不定冠词+空格+单词的情况）
        else:
            return None  # 如果没有匹配，返回 None
            
        return pred 

    @staticmethod
    def extract_single_choice_with_llm(pred, question=None, cache_dir=None):
        """
        Extract a single choice answer from a prediction using an LLM.
        Uses concurrency and caching to improve performance.
        
        Args:
            pred (str): The prediction text to extract a choice from
            question (str, optional): The question text containing options. Default is None.
            cache_dir (str, optional): Directory for caching results. Default is 'output_dir'.
        
        Returns:
            str: The extracted choice (A, B, C, D, or Z for no match)
        """
        import hashlib
        import os
        import json
        from concurrent.futures import ProcessPoolExecutor
        from .utils import build_judge
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.environ.get('OUTPUT_DIR', 'output_dir')
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create cache key from question and prediction
        def get_cache_path(question, pred):
            combined = (question or "") + (pred or "")
            hash_key = hashlib.md5(combined.encode()).hexdigest()
            return os.path.join(cache_dir, f"choice_cache_{hash_key}.json")
        
        cache_path = get_cache_path(question, pred)
        
        # Check cache first
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    return cached_data.get('choice', 'Z')
            except:
                # If there's any error with the cache, proceed without it
                pass
        
        # Build the model using build_judge
        model = build_judge(model='chatgpt-0125')
        
        # Build the prompt for the LLM
        prompt = (
            'You are an AI assistant who will help me to match '
            'an answer with several options of a single-choice question. '
            'You are provided with a question and an answer, '
            'and you need to find which option is most similar to the answer. '
            'If the meaning of all options are significantly different from the answer, output Z. '
            'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. '
            'Do not explain your reasoning, just output the letter directly.\n'
            'Example 1: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
            'Answer: a cute teddy bear\nYour output: A\n'
            'Example 2: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
            'Answer: Spider\nYour output: Z\n'
            'Example 3: \n'
            f'Question: {question}\nAnswer: {pred}\nYour output: '
        )
        
        retry = 3
        while retry:
            try:
                ans = model.generate(prompt)
                # Try to extract the choice from the answer
                choice = None
                
                # First look for a single letter answer
                match = re.search(r'\b([A-DZ])\b', ans)
                if match:
                    choice = match.group(1)
                
                # Also look for patterns like "(A)" or "A."
                if not choice:
                    match = re.search(r'[\(\s]([A-DZ])[\)\.\s]', ans)
                    if match:
                        choice = match.group(1)
                
                if choice and choice in "ABCDZ":
                    # Save to cache
                    try:
                        with open(cache_path, 'w') as f:
                            json.dump({
                                'choice': choice,
                                'full_response': ans,
                                'prompt': prompt
                            }, f)
                    except:
                        # If caching fails, just continue
                        pass
                    
                    return choice
            except:
                pass
            
            retry -= 1
        
        # If all attempts failed, return Z
        # Save failure to cache too
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'choice': 'Z',
                    'full_response': 'Failed to extract',
                    'prompt': prompt
                }, f)
        except:
            pass
            
        return "Z"

    @staticmethod
    def _process_single_item(row_dict, cache_dir='output_dir'):
        """
        处理单个项目的工作函数，用于并行处理
        
        Args:
            row_dict (dict): 数据行字典
            cache_dir (str): 缓存目录
            
        Returns:
            str: 提取的选项
        """
        import time
        import random
        import sys
        import os
        import hashlib
        import json
        import re
        
        try:
            pred = row_dict['prediction']
            question = row_dict['question']
            
            # 构建缓存路径
            def get_cache_path(question, pred):
                # 规范化输入，删除所有空白字符，确保每次生成相同的哈希值
                combined = ((question or "") + (pred or "")).strip()
                # 仅使用前1000个字符计算哈希，避免超长文本
                combined = combined[:1000]
                hash_key = hashlib.md5(combined.encode('utf-8', errors='ignore')).hexdigest()
                return os.path.join(cache_dir, f"choice_cache_{hash_key}.json")
            
            cache_path = get_cache_path(question, pred)
            
            # 检查缓存 - 添加详细日志
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        choice = cached_data.get('choice', None)
                        if choice and choice in "ABCDZ":
                            print(f"✅ 缓存命中: {cache_path}")
                            return choice
                        else:
                            print(f"❌ 缓存格式无效: {cache_path}")
                except Exception as e:
                    print(f"❌ 读取缓存出错 {cache_path}: {e}")
            else:
                print(f"⚠️ 缓存不存在: {cache_path}")
            
            # 添加随机延迟以避免API速率限制
            time.sleep(random.uniform(0, 0.2))
            
            # 直接调用静态方法处理
            from .utils import build_judge
            
            # 构建模型
            model = build_judge(model='chatgpt-0125')
            
            # 构建提示
            prompt = (
                'You are an AI assistant who will help me to match '
                'an answer with several options of a single-choice question. '
                'You are provided with a question and an answer, '
                'and you need to find which option is most similar to the answer. '
                'If the meaning of all options are significantly different from the answer, output Z. '
                'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. '
                'Do not explain your reasoning, just output the letter directly.\n'
                'Example 1: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
                'Answer: a cute teddy bear\nYour output: A\n'
                'Example 2: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
                'Answer: Spider\nYour output: Z\n'
                'Example 3: \n'
                f'Question: {question}\nAnswer: {pred}\nYour output: '
            )
            
            # 调用模型
            retry = 3
            while retry:
                try:
                    ans = model.generate(prompt)
                    # 提取选项
                    choice = None
                    
                    # 首先查找单个字母答案
                    match = re.search(r'\b([A-DZ])\b', ans)
                    if match:
                        choice = match.group(1)
                    
                    # 还查找类似"(A)"或"A."的模式
                    if not choice:
                        match = re.search(r'[\(\s]([A-DZ])[\)\.\s]', ans)
                        if match:
                            choice = match.group(1)
                    
                    if choice and choice in "ABCDZ":
                        # 保存到缓存
                        try:
                            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                            with open(cache_path, 'w', encoding='utf-8') as f:
                                cache_data = {
                                    'choice': choice,
                                    'full_response': ans,
                                    'prompt': prompt
                                }
                                json.dump(cache_data, f, ensure_ascii=False)
                                print(f"✅ 缓存已保存: {cache_path}")
                        except Exception as e:
                            print(f"❌ 保存缓存失败 {cache_path}: {e}")
                        
                        return choice
                except Exception as e:
                    print(f"❌ 调用模型出错: {e}")
                
                retry -= 1
            
            # 如果所有尝试都失败，返回Z
            # 保存失败结果到缓存
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'choice': 'Z',
                        'full_response': '调用失败',
                        'prompt': prompt
                    }, f, ensure_ascii=False)
                    print(f"⚠️ 保存默认缓存(Z): {cache_path}")
            except Exception as e:
                print(f"❌ 保存默认缓存失败: {e}")
            
            return "Z"
        except Exception as e:
            print(f"❌ 处理项目时出错: {e}")
            return "Z"  # 出错时返回默认值

    @staticmethod
    def batch_extract_choices_with_llm(data_rows, num_processes=32, cache_dir='output_dir', use_single_thread=False):
        """
        并发处理多个预测
        
        Args:
            data_rows: 包含预测和问题的数据行
            num_processes: 要使用的并发进程数
            cache_dir: 缓存结果的目录
            use_single_thread: 是否强制使用单线程处理
            
        Returns:
            list: 提取的选项列表
        """
        import os
        import multiprocessing as mp
        from functools import partial
        import time
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 使用进程池来并发处理项目
        results = []
        total = len(data_rows)
        
        # 为每行创建简单的字典，只包含必要信息
        row_dicts = []
        for row in data_rows:
            row_dicts.append({
                'prediction': row['prediction'],
                'question': row['question']
            })
        
        # 如果指定了单线程或进程数为1，则使用单线程处理
        if use_single_thread or num_processes <= 1:
            print(f"使用单线程处理 {total} 个样本...")
            results = []
            start_time = time.time()
            for i, row in enumerate(row_dicts):
                result = MSRBenchDataset._process_single_item(row, cache_dir=cache_dir)
                results.append(result)
                if (i+1) % max(1, total//20) == 0:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / (i+1)) * (total - (i+1))
                    print(f"单线程进度: {i+1}/{total} ({(i+1)/total*100:.1f}%) - 已用时: {elapsed:.1f}秒, 剩余时间: {remaining:.1f}秒")
            return results
        
        # 创建进程共享的处理函数
        process_func = partial(MSRBenchDataset._process_single_item, cache_dir=cache_dir)
        
        # 对于Windows，需要使用if __name__=='__main__'来避免子进程递归创建
        # 但在当前环境中，我们直接使用multiprocessing
        print(f"启动 {num_processes} 个进程处理 {total} 个样本")
        
        try:
            # 设置启动方法为 forkserver 或 spawn 以提高兼容性
            if hasattr(mp, 'get_context'):
                try:
                    # 优先使用 forkserver，它在Linux上通常更可靠
                    ctx = mp.get_context('forkserver')
                    pool = ctx.Pool(processes=num_processes)
                    print("使用 forkserver 方式启动进程池")
                except ValueError:
                    try:
                        # 如果不支持 forkserver，则尝试 spawn
                        ctx = mp.get_context('spawn')
                        pool = ctx.Pool(processes=num_processes)
                        print("使用 spawn 方式启动进程池")
                    except ValueError:
                        # 如果都不支持，则使用默认方式
                        pool = mp.Pool(processes=num_processes)
                        print("使用默认方式启动进程池")
            else:
                # 如果无法设置上下文，则使用默认池
                pool = mp.Pool(processes=num_processes)
                print("使用标准进程池")
            
            # 使用 imap 可以按顺序得到结果，同时支持并行处理
            for i, result in enumerate(pool.imap(process_func, row_dicts, chunksize=max(1, total // (num_processes * 4)))):
                results.append(result)
                processed = i + 1
                if processed % max(1, total//20) == 0:  # 每5%更新一次进度
                    print(f"进度: {processed}/{total} ({processed/total*100:.1f}%)")
            
            pool.close()
            pool.join()
            
        except Exception as e:
            print(f"并行处理出错: {e}")
            # 发生错误时，尝试单线程处理
            print("尝试单线程处理...")
            results = []
            for i, row in enumerate(row_dicts):
                result = MSRBenchDataset._process_single_item(row, cache_dir=cache_dir)
                results.append(result)
                if (i+1) % max(1, total//20) == 0:
                    print(f"单线程进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
        
        print(f"完成所有处理: {len(results)}/{total}")
        return results

class MSRBenchCircular(MSRBenchDataset):
    """
    MSR Bench Circular Dataset class.
    Uses circular evaluation method for multiple-choice questions.
    选项嵌入在question字段中，使用circular evaluation方法进行评估。
    """
    TYPE = 'MCQ'
    
    @classmethod
    def supported_datasets(cls):
        return ['MSR_Bench_Circular']
    
    def extract_options_from_question(self, question):
        """
        从question文本中提取选项，返回(主问题文本, 选项dict)
        """
        # 分割出 Options: 后面的内容
        parts = question.split("Options:", 1)
        if len(parts) < 2:
            return question.strip(), {}
        
        question_text = parts[0].strip()
        options_text = parts[1].strip()
        
        # 通用的选项提取模式，适用于逗号分隔或空格分隔的情况
        pattern = r'([A-D])\s*:\s*(.*?)(?=\s+[A-D]\s*:|,\s*[A-D]\s*:|$)'
        matches = re.findall(pattern, options_text)
        options = {m[0]: m[1].strip() for m in matches}
        
        return question_text, options

    def build_question_with_options(self, question_text, options):
        """
        重新拼接question和options为原格式
        """
        options_str = "Options: " + ", ".join([f"{k}: {v}" for k, v in options.items()])
        return f"{question_text}\n{options_str}"

    def load_data(self, dataset):
        """
        加载数据并自动生成 circular 变体，每题4种选项顺序。
        """
        if dataset == 'MSR_Bench_Circular':
            tsv_path = MSRBenchDataset.MSR_BENCH_TSV
            if not osp.exists(tsv_path):
                raise FileNotFoundError(f"MSR_Bench TSV文件不存在: {tsv_path}")

            data = pd.read_csv(tsv_path, sep='\t')
            assert 'index' in data.columns, "TSV文件缺少'index'列"
            assert 'question' in data.columns, "TSV文件缺少'question'列"

            cp4 = ['ABCD', 'BCDA', 'CDAB', 'DABC']
            new_rows = []

            for _, row in tqdm(data.iterrows(), desc="Processing data", total=len(data)):
                question_text, options = self.extract_options_from_question(row['question'])
                answer = row['answer'] if 'answer' in row else None

                # 跳过没有选项的题
                if not options or answer not in options:
                    # import ipdb; ipdb.set_trace()
                    print(f"跳过没有选项的题: {row['index']}")
                    print(f"选项: {options}")
                    print(f"答案: {answer}")
                    print(f"row['question']: {row['question']}")
                    continue

                for i, order in enumerate(cp4):
                    # 重新排列选项
                    new_options = {k: options[o] for k, o in zip('ABCD', order) if o in options}
                    # 计算新答案
                    if answer in order:
                        new_answer = 'ABCD'[order.index(answer)]
                    else:
                        new_answer = answer  # fallback

                    # 构造新行
                    new_row = row.copy()
                    # 重新拼接question
                    new_row['question'] = self.build_question_with_options(question_text, new_options)
                    new_row['answer'] = new_answer
                    new_row['index'] = int(row['index']) + i * 1000000
                    new_row['g_index'] = row['index']  # 用于分组
                    new_rows.append(new_row)

            new_data = pd.DataFrame(new_rows)
            return new_data

        else:
            return super(MSRBenchCircular, self).load_data(dataset)
    
    def evaluate(self, eval_file, cache_dir='output_dir', num_processes=32, use_single_thread=False, **judge_kwargs):
        """
        评估方法，同时计算循环评估和传统评估的结果
        
        Args:
            eval_file: 评估数据文件路径
            cache_dir: 缓存目录路径
            num_processes: 并行处理的进程数
            use_single_thread: 是否使用单线程处理
            **judge_kwargs: 其他参数
        """
        from ..smp.file import load, dump
        from .utils.multiple_choice import report_acc
        import pandas as pd
        import numpy as np
        import os
        import glob
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 检查缓存目录中已有的缓存文件数量
        cache_files = glob.glob(os.path.join(cache_dir, "choice_cache_*.json"))
        print(f"❗ 缓存目录 '{cache_dir}' 中已有 {len(cache_files)} 个缓存文件")
        
        suffix = eval_file.split('.')[-1]
        
        # 加载和预处理评估数据
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['index'] = [int(x) for x in data['index']]
        data['prediction'] = [str(x) for x in data['prediction']]
        
        # 确保数据中有g_index字段
        if 'g_index' not in data.columns:
            data['g_index'] = [int(x % 1e6) for x in data['index']]
            
        # 使用LLM提取选项，通过并发处理提高速度
        print(f"❗ 使用 {num_processes} 个并行进程提取选项，缓存目录：{cache_dir}")
        print(f"❗ {'使用单线程模式' if use_single_thread else '使用多进程模式'}")
        
        # 提取前统计缓存文件数量
        cache_files_before = len(glob.glob(os.path.join(cache_dir, "choice_cache_*.json")))
        
        data['extracted_pred'] = MSRBenchDataset.batch_extract_choices_with_llm(
            data.to_dict('records'), 
            num_processes=num_processes,
            cache_dir=cache_dir,
            use_single_thread=use_single_thread
        )
        
        # 提取后统计缓存文件数量
        cache_files_after = len(glob.glob(os.path.join(cache_dir, "choice_cache_*.json")))
        new_cache_files = cache_files_after - cache_files_before
        
        print(f"❗ 已完成 {len(data)} 个样本的选项提取")
        print(f"❗ 新增 {new_cache_files} 个缓存文件，共有 {cache_files_after} 个缓存文件")
        
        # ----- 循环评估 (Circular Evaluation) -----
        print(f"🔄 开始计算循环评估 (Circular Evaluation) 结果...")
        
        # 分组评估
        groups = data.groupby('g_index')
        circular_results = []
        
        for g_index, group in tqdm(groups, desc="Processing groups", total=len(groups)):
            # 验证 g_index 确实小于 10^6
            assert g_index < 1e6, f"g_index {g_index} 不小于 10^6，数据有误"
            
            # 创建基本结果行
            result_row = {
                'index': int(g_index),  # 使用g_index作为主index
                'category': group['category'].iloc[0],
                'hit': 0,
                'log': ''
            }
            
            # 检查是否所有预测都正确
            all_correct = True
            log_parts = []
            
            for _, row in group.iterrows():
                pred = row['extracted_pred']
                ans = row['answer']
                
                # 记录当前行预测结果
                correct = (pred == ans)
                log_parts.append(f"Index {row['index']}: 预测={pred}, 答案={ans}, 正确={correct}")
                
                if not correct:
                    all_correct = False
            
            # 如果所有预测都正确，则这题算对
            result_row['hit'] = 1 if all_correct else 0
            result_row['log'] = '\n'.join(log_parts)
            
            circular_results.append(result_row)
            
        # 创建循环评估结果DataFrame
        circular_df = pd.DataFrame(circular_results)
        
        # ----- 传统评估 (Vanilla Evaluation) -----
        print(f"🔍 开始计算传统评估 (Vanilla Evaluation) 结果...")
        
        # 创建传统评估的结果列表
        vanilla_results = []
        
        for g_index, group in tqdm(groups, desc="Processing original items", total=len(groups)):
            # 验证 g_index 确实小于 10^6
            assert g_index < 1e6, f"g_index {g_index} 不小于 10^6，数据有误"
            
            # 找到组内index最小的行（原始题目）
            original_row = group.loc[group['index'].idxmin()]
            
            # 创建结果行
            result_row = {
                'index': int(g_index),  # 使用g_index作为主index
                'category': original_row['category'],
                'answer': original_row['answer'],
                'prediction': original_row['prediction'],
                'extracted_pred': original_row['extracted_pred'],
                'hit': 1 if original_row['extracted_pred'] == original_row['answer'] else 0,
                'log': f"Index {original_row['index']}: 预测={original_row['extracted_pred']}, 答案={original_row['answer']}"
            }
            
            vanilla_results.append(result_row)
        
        # 创建传统评估结果DataFrame
        vanilla_df = pd.DataFrame(vanilla_results)
        
        # ----- 保存结果 -----
        # 保存循环评估详细结果
        circular_detailed_file = eval_file.replace(f'.{suffix}', f'_circular_result.{suffix}')
        dump(circular_df, circular_detailed_file)
        
        # 保存传统评估详细结果
        vanilla_detailed_file = eval_file.replace(f'.{suffix}', f'_vanilla_result.{suffix}')
        dump(vanilla_df, vanilla_detailed_file)
        
        # 计算循环评估准确率
        circular_acc = {}
        circular_acc['Overall'] = np.mean(circular_df['hit'])
        
        # 计算传统评估准确率
        vanilla_acc = {}
        vanilla_acc['Overall'] = np.mean(vanilla_df['hit'])
        
        # 按类别计算准确率
        if 'category' in circular_df.columns:
            categories = circular_df['category'].unique()
            for category in categories:
                # 循环评估
                circ_cat_data = circular_df[circular_df['category'] == category]
                circular_acc[category] = np.mean(circ_cat_data['hit'])
                
                # 传统评估
                van_cat_data = vanilla_df[vanilla_df['category'] == category]
                vanilla_acc[category] = np.mean(van_cat_data['hit'])
        
        # 创建报告格式
        combined_acc = {}
        for key in circular_acc.keys():
            combined_acc[key] = {
                'Circular': circular_acc[key],
                'Vanilla': vanilla_acc[key]
            }
        
        combined_df = pd.DataFrame(combined_acc).T
        combined_df.index.name = 'Category'
        
        # 保存准确率结果
        score_file = eval_file.replace(f'.{suffix}', f'_combined_acc.csv')
        dump(combined_df, score_file)
        
        # 输出最终结果
        print(f"\n====== MSR_Bench 评测结果 ======")
        print(f"总样本组数: {len(circular_df)}")
        
        print(f"\n📊 传统评估 (Vanilla) - 单题正确率: {vanilla_acc['Overall']:.2%}")
        print(f"📊 循环评估 (Circular) - 全题组正确率: {circular_acc['Overall']:.2%}")
        
        print(f"\n📊 各类别准确率:")
        for cat in categories:
            print(f"{cat}:")
            print(f"  传统评估: {vanilla_acc[cat]:.2%}")
            print(f"  循环评估: {circular_acc[cat]:.2%}")
        
        # 返回两种评估结果的DataFrame
        return combined_df 
