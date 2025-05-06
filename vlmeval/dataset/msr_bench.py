import os
import re
import pandas as pd
import os.path as osp
import json
import numpy as np
import warnings
from .image_mcq import ImageMCQDataset
from .utils import DEBUG_MESSAGE, build_judge
from ..smp import LMUDataRoot, file_size, load, dump, decode_base64_to_image_file, listinstr, gpt_key_set
import string

class MSRBenchDataset(ImageMCQDataset):
    """
    MSR Bench Dataset class for multiple-choice questions with multiple images.
    支持多图片的多选题评测数据集，图片以JSON数组格式存储在image字段中。
    """
    TYPE = 'MCQ'
    
    # 使用本地TSV文件路径，不进行网络下载
    MSR_BENCH_TSV = '/fs-computility/mllm1/shared/LMUData/msr_bench_fanal_version_5_5_cat_option_to_qs.tsv'
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
            extracted_pred = cls.extract_single_choice_with_word_boundary(pred, gt)
            
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
    def extract_single_choice_with_word_boundary(pred, gt):
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
        从问题文本中提取选项。
        格式：问题文本 + "Options:" + "A: 选项1, B: 选项2, ..."
        """
        # 检查是否有"Options:"部分
        parts = question.split("Options:", 1)
        if len(parts) < 2:
            # 如果没有找到"Options:"，返回原始问题文本和空选项
            return parts[0].strip(), {}
        
        # 提取问题和选项部分
        question_text = parts[0].strip()
        options_text = parts[1].strip()
        
        # 提取选项
        options = {}
        # 使用正则表达式提取选项
        pattern = r'([A-D])\s*:\s*([^,]*?)(?:,\s*[A-D]\s*:|$)'
        matches = re.findall(pattern, options_text)
        
        for key, value in matches:
            options[key] = value.strip()
        
        return question_text, options
    
    def build_prompt(self, line):
        """
        构建提示，支持多图片输入，并从question中提取选项。
        """
        if isinstance(line, int):
            line = self.data.iloc[line]
            
        # 处理图片（支持多图）
        tgt_path = self.dump_image(line)
        
        # 从question中提取选项
        question_text, options = self.extract_options_from_question(line['question'])
        
        # 构建标准MCQ格式的提示
        prompt = ''
        prompt += f'Question: {question_text}\n'
        
        if options:
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above.\n'
        
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
    
    def preprocess_data(self):
        """
        预处理数据：从question中提取选项，并添加到数据中的A、B、C、D列
        """
        for idx, row in self.data.iterrows():
            question_text, options = self.extract_options_from_question(row['question'])
            # 更新数据中的选项列
            for key, value in options.items():
                self.data.at[idx, key] = value
    
    def post_build(self, dataset):
        """
        在加载完数据后，进行数据预处理
        """
        self.preprocess_data()
    
    def evaluate(self, eval_file, **judge_kwargs):
        """
        使用circular evaluation方法评估模型预测结果。
        """
        from .utils.multiple_choice import mcq_circular_eval, report_acc
        
        nproc = judge_kwargs.pop('nproc', 4)
        suffix = eval_file.split('.')[-1]
        
        # 处理模型设置
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model
        
        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None
        
        # 结果文件路径
        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')
        
        # 加载和预处理评估数据
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['index'] = [int(x) for x in data['index']]  # 确保index是整数
        data['prediction'] = [str(x) for x in data['prediction']]
        
        # 统一列名大小写
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)
        
        # 确保评估数据与训练数据匹配
        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )
        
        # 使用circular评估方法
        data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        
        # 保存评估结果
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        
        # 计算准确率
        acc = report_acc(data)
        
        # 保存准确率结果
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)
        
        return acc 