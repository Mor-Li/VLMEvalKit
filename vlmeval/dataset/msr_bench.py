import os
import re
import pandas as pd
import os.path as osp
import json
import numpy as np
from .image_mcq import ImageMCQDataset
from ..smp import LMUDataRoot, file_size, load, decode_base64_to_image_file
import string

class MSRBenchDataset(ImageMCQDataset):
    """
    MSR Bench Dataset class for multiple-choice questions with multiple images.
    支持多图片的多选题评测数据集，图片以JSON数组格式存储在image字段中。
    """
    TYPE = 'MCQ'
    
    # 使用本地TSV文件路径，不进行网络下载
    MSR_BENCH_TSV = '/fs-computility/mllm1/shared/LMUData/msr_bench_cat_option_to_qs.tsv'
    
    DATASET_URL = {
        'MSR_Bench': 'file:///fs-computility/mllm1/shared/LMUData/msr_bench_cat_option_to_qs.tsv'
    }
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