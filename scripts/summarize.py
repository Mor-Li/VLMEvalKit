from vlmeval.smp import *
from vlmeval.dataset import SUPPORTED_DATASETS

# 存储数据集各个类别的样本数
category_counts = {}

def get_score(model, dataset):
    global category_counts
    
    file_name = f'outputs/{model}/{model}_{dataset}'
    if listinstr([
        'CCBench', 'MMBench', 'SEEDBench_IMG', 'MMMU', 'ScienceQA', 
        'AI2D_TEST', 'MMStar', 'RealWorldQA', 'BLINK', 'VisOnlyQA-VLMEvalKit'
    ], dataset):
        file_name += '_acc.csv'
    elif dataset == 'MSR_Bench_Circular':
        file_name += '_simple_acc.csv'
    elif listinstr(['MME', 'Hallusion', 'LLaVABench'], dataset):
        file_name += '_score.csv'
    elif listinstr(['MMVet', 'MathVista'], dataset):
        file_name += '_gpt-4-turbo_score.csv'
    elif listinstr(['COCO', 'OCRBench'], dataset):
        file_name += '_score.json'
    elif listinstr(['Spatial457'], dataset):
        file_name += '_score.json'
    elif listinstr(['MSR_Bench'], dataset):
        file_name += '_score.xlsx'
    else:
        raise NotImplementedError
    if not osp.exists(file_name):
        print(f"文件未找到: {file_name}")
        return {}
    
    data = load(file_name)
    ret = {}
    if dataset == 'CCBench':
        ret[dataset] = data['Overall'][0] * 100
    elif dataset == 'MMBench':
        for n, a in zip(data['split'], data['Overall']):
            if n == 'dev':
                ret['MMBench_DEV_EN'] = a * 100
            elif n == 'test':
                ret['MMBench_TEST_EN'] = a * 100
    elif dataset == 'MMBench_CN':
        for n, a in zip(data['split'], data['Overall']):
            if n == 'dev':
                ret['MMBench_DEV_CN'] = a * 100
            elif n == 'test':
                ret['MMBench_TEST_CN'] = a * 100
    elif listinstr(['SEEDBench', 'ScienceQA', 'MMBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'BLINK'], dataset):
        ret[dataset] = data['Overall'][0] * 100
    elif 'MME' == dataset:
        ret[dataset] = data['perception'][0] + data['reasoning'][0]
    elif 'MMVet' == dataset:
        data = data[data['Category'] == 'Overall']
        ret[dataset] = float(data.iloc[0]['acc'])
    elif 'HallusionBench' == dataset:
        data = data[data['split'] == 'Overall']
        for met in ['aAcc', 'qAcc', 'fAcc']:
            ret[dataset + f' ({met})'] = float(data.iloc[0][met])
    elif 'MMMU' in dataset:
        data = data[data['split'] == 'validation']
        ret['MMMU (val)'] = float(data.iloc[0]['Overall']) * 100
    elif 'MathVista' in dataset:
        data = data[data['Task&Skill'] == 'Overall']
        ret[dataset] = float(data.iloc[0]['acc'])
    elif 'LLaVABench' in dataset:
        data = data[data['split'] == 'overall'].iloc[0]
        ret[dataset] = float(data['Relative Score (main)'])
    elif 'OCRBench' in dataset:
        ret[dataset] = data['Final Score']
    elif dataset == "VisOnlyQA-VLMEvalKit":
        for n, a in zip(data['split'], data['Overall']):
            ret[f'VisOnlyQA-VLMEvalKit_{n}'] = a * 100
    elif 'Spatial457' in dataset:
        ret["All"] = data["score"] * 100
        for level in ["L1_single", "L2_objects", "L3_2d_spatial", "L4_occ",
                        "L4_pose", "L5_6d_spatial", "L5_collision"]:
            ret[f"{dataset} - {level}"] = data[f"{level}_score"] * 100
    elif dataset == 'MSR_Bench_Circular':
        # Handle circular evaluation results similar to other circular benchmarks
        ret[dataset] = data['Overall'][0] * 100
        # Add category-specific results if available
        if len(data.columns) > 1:
            for col in data.columns:
                if col != 'Overall' and col != 'split':
                    ret[f'{dataset} - {col}'] = data[col][0] * 100
    elif dataset == 'MSR_Bench':
        # Calculate overall accuracy from the score column (0 or 1 for each question)
        if 'score' in data.columns:
            # Overall accuracy is the mean of all scores
            overall_acc = data['score'].mean() * 100
            ret[dataset] = overall_acc
            
            # 计算每个类别的样本数量
            if 'category' in data.columns and dataset not in category_counts:
                category_counts[dataset] = data['category'].value_counts().to_dict()
            
            # Calculate category-wise accuracies
            if 'category' in data.columns:
                category_results = data.groupby('category')['score'].mean() * 100
                for cat, score in category_results.items():
                    if not pd.isna(cat) and cat != 'nan':
                        ret[f'{dataset} - {cat}'] = score
    return ret

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', default=[])
    parser.add_argument("--model", type=str, nargs='+', required=True)
    args = parser.parse_args()
    return args

def gen_table(models, datasets):
    res = defaultdict(dict)
    for m in models:
        for d in datasets:
            try:
                res[m].update(get_score(m, d))
            except Exception as e:
                logging.warning(f'{type(e)}: {e}')
                logging.warning(f'Missing Results for Model {m} x Dataset {d}')
    keys = []
    for m in models:
        for d in res[m]:
            keys.append(d)
    keys = list(set(keys))
    keys.sort()
    final = defaultdict(list)
    for m in models:
        final['Model'].append(m)
        for k in keys:
            if k in res[m]:
                final[k].append(res[m][k])
            else:
                final[k].append(None)
    final = pd.DataFrame(final)
    
    # 显示类别样本数量信息
    if category_counts:
        print("=== 数据集类别样本统计 ===")
        for dataset, counts in category_counts.items():
            print(f"\n{dataset} 总样本数: {sum(counts.values())}")
            category_info = []
            for cat, count in sorted(counts.items()):
                if not pd.isna(cat) and cat != 'nan':
                    category_info.append(f"{cat}: {count}题")
            
            # 每行打印3个类别信息
            for i in range(0, len(category_info), 3):
                print("  ".join(category_info[i:i+3]))
        print("\n=== 模型性能对比 ===")
    
    dump(final, 'summ.csv')
    
    # 重命名列名，使表格更易读
    columns_mapping = {}
    for col in final.columns:
        if col == 'Model':
            continue
        if ' - ' in col:
            dataset, category = col.split(' - ', 1)
            columns_mapping[col] = category
        else:
            columns_mapping[col] = col
    
    final_display = final.rename(columns=columns_mapping)
    
    if len(final) >= len(final.iloc[0].keys()):
        print(tabulate(final_display, headers='keys', showindex=True))
    else:
        print(tabulate(final_display.T, headers='keys', showindex=True))
    
if __name__ == '__main__':
    args = parse_args()
    if args.data == []:
        args.data = list(SUPPORTED_DATASETS)
    gen_table(args.model, args.data)