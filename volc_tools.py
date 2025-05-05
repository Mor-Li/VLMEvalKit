import os  # 导入操作系统模块
import subprocess  # 导入子进程模块
import uuid  # 导入UUID模块
import yaml  # 导入YAML模块
import argparse  # 导入命令行参数解析模块
from typing import Dict, Optional  # 导入类型提示
from dataclasses import dataclass, field, asdict  # 导入数据类相关模块
from loguru import logger  # 导入日志模块

# 配置loguru日志记录器
logger.remove()  # 移除默认处理器
logger.add(
    lambda msg: print(msg, flush=True),  # 只打印到控制台
    colorize=True,  # 颜色化输出
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"  # 控制台日志格式
)

def get_current_conda_env() -> str:
    """获取当前conda环境的路径。"""
    try:
        # 从环境中获取CONDA_PREFIX
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            logger.debug(f"从CONDA_PREFIX找到的conda环境: {conda_prefix}")
            return conda_prefix
        
        # 如果未设置CONDA_PREFIX，尝试从conda命令获取
        logger.debug("未找到CONDA_PREFIX，尝试使用conda info命令")
        result = subprocess.run(
            'conda info --envs | grep "*" | awk \'{print $NF}\'',  # 获取当前激活的conda环境
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            env_path = result.stdout.strip()
            logger.debug(f"从命令找到的conda环境: {env_path}")
            return env_path
        
    except Exception as e:
        logger.warning(f"检测conda环境失败: {e}")
    
    # 如果检测失败，返回默认环境
    default_env = '/fs-computility/llm/shared/llmeval/share_envs/oc-v034-ld-v061'
    logger.warning(f"使用默认的conda环境: {default_env}")
    return default_env

@dataclass
class VolcanoConfig:
    """Volcano部署的配置。"""
    bashrc_path: str = '/fs-computility/mllm1/limo/.bashrc'  # bashrc文件路径
    conda_env_name: str = field(default_factory=get_current_conda_env)  # conda环境名称
    huggingface_cache: str = '/fs-computility/llm/shared/mllm/hub'  # Hugging Face缓存路径
    torch_cache: str = '/fs-computility/llm/shared/llmeval/torch'  # Torch缓存路径
    volc_cfg_file: str = '/fs-computility/mllm1/limo/workspace/opencompass/configs/internal/volc_infer_limo.yaml'  # Volcano配置文件路径
    task_name: str = 'compassjudger-1-32B'  # 任务名称
    queue_name: str = 'llmeval_volc'  # 队列名称
    extra_envs: list = field(default_factory=lambda: [  # 额外的环境变量
        'COMPASS_DATA_CACHE=/fs-computility/llm/shared/llmeval/datasets/compass_data_cache',
        'TORCH_HOME=/fs-computility/llm/shared/llmeval/torch',
        'TIKTOKEN_CACHE_DIR=/fs-computility/llm/shared/llmeval/share_tiktoken',
        'VLLM_WORKER_MULTIPROC_METHOD=spawn',
        'LMUData=/fs-computility/mllm1/shared/LMUData',
        'MKL_SERVICE_FORCE_INTEL=1',
    ])
    image: str = "fs-computility-cn-beijing.cr.volces.com/devinstance-archive/moli-opencompass:1.4"  # Docker镜像

class VolcanoDeployment:
    """处理将ML任务部署到Volcano基础设施。"""
    
    def __init__(self, config: Optional[Dict] = None):
        """使用配置初始化部署。"""
        self.config = VolcanoConfig(**config) if config else VolcanoConfig()  # 初始化配置
        self.pwd = os.getcwd()  # 获取当前工作目录
        logger.info("使用配置初始化VolcanoDeployment:")
        logger.info(f"工作目录: {self.pwd}")
        for key, value in asdict(self.config).items():
            logger.info(f"{key}: {value}")
        
    def choose_flavor(self, num_gpus: int, num_replicas: int = 1) -> Dict:
        """根据GPU需求选择合适的机器类型。"""
        flavor_map = {  # 机器类型映射
            0: 'ml.r3i.large',
            1: 'ml.pni2l.3xlarge',
            2: 'ml.pni2l.7xlarge',
            4: 'ml.pni2l.14xlarge',
            8: 'ml.hpcpni2l.28xlarge'
        }
        
        if num_gpus > 8:
            logger.error(f"不支持{num_gpus}个GPU的配置")
            raise NotImplementedError(f"不支持{num_gpus}个GPU的配置")
        
        for max_gpus, flavor in sorted(flavor_map.items()):
            if num_gpus <= max_gpus:
                selected_flavor = flavor  # 选择合适的机器类型
                break
        
        logger.info(f"为{num_gpus}个GPU选择的机器类型: {selected_flavor}")
        logger.info(f"副本数量: {num_replicas}")
                
        with open(self.config.volc_cfg_file) as fp:
            volc_cfg = yaml.safe_load(fp)  # 加载YAML配置文件
            
        for role_spec in volc_cfg['TaskRoleSpecs']:
            if role_spec['RoleName'] == 'worker':
                role_spec['Flavor'] = selected_flavor  # 设置机器类型
                role_spec['RoleReplicas'] = num_replicas  # 设置副本数量
                
        return volc_cfg
    
    def build_shell_command(self, task_cmd: str) -> str:
        """构建包含所有必要环境设置的shell命令。"""
        logger.debug("构建shell命令")
        cmd_parts = [
            f'source {self.config.bashrc_path}',  # 加载bashrc文件
        ]
        
        # 从环境中获取CONDA_EXE
        conda_exe = os.environ.get("CONDA_EXE", None)
        assert conda_exe, f"CONDA_EXE为None，请确保conda在当前环境中存在"
        conda_activate = conda_exe.replace("bin/conda", "bin/activate")  # 获取conda激活命令

        # 根据路径或名称处理conda环境激活
        if os.path.exists(self.config.conda_env_name):
            logger.debug(f"使用路径激活conda环境: {self.config.conda_env_name}")
            cmd_parts.append(f'source {conda_activate} {self.config.conda_env_name}')
        else:
            logger.debug(f"使用名称激活conda环境: {self.config.conda_env_name}")
            cmd_parts.append(f'source {conda_activate} {self.config.conda_env_name}')
        
        cmd_parts.extend([
            f'export PYTHONPATH={self.pwd}:$PYTHONPATH',  # 设置PYTHONPATH
            f'export HF_HUB_CACHE={self.config.huggingface_cache}',  # 设置Hugging Face缓存
            f'export HUGGINGFACE_HUB_CACHE={self.config.huggingface_cache}',  # 设置Hugging Face缓存
            f'export TORCH_HOME={self.config.torch_cache}'  # 设置Torch缓存
        ])
        
        offline_vars = [  # 离线模式环境变量
            # 'HF_DATASETS_OFFLINE=1',
            # 'TRANSFORMERS_OFFLINE=1',
            # 'HF_EVALUATE_OFFLINE=1',
            # 'HF_HUB_OFFLINE=1',
            'HF_ENDPOINT=https://hf-mirror.com'
        ]
        cmd_parts.extend([f'export {var}' for var in offline_vars])  # 添加离线模式环境变量
        
        if self.config.extra_envs:
            cmd_parts.extend([f'export {env}' for env in self.config.extra_envs])  # 添加额外环境变量
        
        cmd_parts.extend([
            f'cd {self.pwd}',  # 切换到工作目录
            task_cmd  # 添加任务命令
        ])
        
        full_cmd = '; '.join(cmd_parts)  # 生成完整命令
        logger.debug(f"生成的shell命令: {full_cmd}")
        return full_cmd
    
    def deploy(self, task_cmd: str, num_gpus: int = 4,  num_replicas: int = 1) -> subprocess.CompletedProcess:
        """将任务部署到Volcano基础设施。"""
        logger.info(f"开始部署，使用{num_gpus}个GPU")
        logger.info(f"任务命令: {task_cmd}")
        
        try:
            volcano_cfg = self.choose_flavor(num_gpus, num_replicas)  # 选择机器类型
            
            os.makedirs(f'{self.pwd}/tmp', exist_ok=True)  # 创建临时目录
            tmp_cfg_file = f'{self.pwd}/tmp/{uuid.uuid4()}_cfg.yaml'  # 生成临时配置文件名
            logger.debug(f"创建临时配置文件: {tmp_cfg_file}")
            
            with open(tmp_cfg_file, 'w') as fp:
                yaml.dump(volcano_cfg, fp, sort_keys=False)  # 写入YAML配置
            
            shell_cmd = self.build_shell_command(task_cmd)  # 构建shell命令
            
            submit_cmd = (  # 提交命令
                'volc ml_task submit'
                f" --conf '{tmp_cfg_file}'"
                f" --entrypoint '{shell_cmd}'"
                f' --task_name {self.config.task_name}'
                f' --resource_queue_name {self.config.queue_name}'
                f' --image {self.config.image}'
            )
            
            logger.info("提交Volcano任务")
            logger.debug(f"提交命令: {submit_cmd}")
            
            result = subprocess.run(
                submit_cmd,
                shell=True,
                text=True,
                capture_output=True,
                check=True
            )
            
            logger.info("任务提交成功")
            return result
            
        except Exception as e:
            logger.error(f"部署失败: {str(e)}")
            raise
        finally:
            # 删除本地生成的volc的log文件
            for fname in os.listdir('.'):
                if fname.startswith('volcano_deploy_') and fname.endswith('.log'):
                    try:
                        os.remove(fname)
                        logger.debug(f"已删除本地log文件: {fname}")
                    except Exception as ex:
                        logger.warning(f"删除log文件失败: {fname}, 错误: {ex}")
            # if os.path.exists(tmp_cfg_file):
            #     logger.debug(f"清理临时配置文件: {tmp_cfg_file}")
            #     os.remove(tmp_cfg_file)

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='将ML任务部署到Volcano基础设施')
    
    # 必需参数
    parser.add_argument('--task-cmd', required=True, help='要执行的主要任务命令')
    
    # 可选参数
    parser.add_argument('--num-gpus', type=int, default=4, help='所需的GPU数量（默认: 4）')
    parser.add_argument('--num-replicas', type=int, default=1, help='副本数量（默认: 1）')
    parser.add_argument('--task-name', help='覆盖默认任务名称')
    parser.add_argument('--queue-name', help='覆盖默认队列名称')
    parser.add_argument('--image', help="覆盖默认镜像")
    parser.add_argument('--conda-env', help='要使用的conda环境（默认: 当前环境）')
    parser.add_argument('--extra-envs', nargs='+', help='额外的环境变量，格式为KEY=VALUE')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                      help='设置日志级别（默认: INFO）')
    parser.add_argument('--yes', action='store_true', help='自动确认部署，不提示')
    
    return parser.parse_args()

def main():
    """主执行函数。"""
    args = parse_args()  # 解析命令行参数
    
    # 设置日志级别
    logger.remove()  # 移除现有处理器
    logger.add(
        lambda msg: print(msg, flush=True),  # 只打印到控制台
        colorize=True,
        level=args.log_level,  # 控制台日志级别
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"  # 控制台日志格式
    )
    
    logger.info("开始Volcano部署脚本")
    
    # 获取当前conda环境
    current_env = get_current_conda_env()
    logger.info(f"当前conda环境: {current_env}")
    
    # 准备配置覆盖
    config_overrides = {}
    if args.task_name:
        config_overrides['task_name'] = args.task_name
    if args.queue_name:
        config_overrides['queue_name'] = args.queue_name
    if args.conda_env:
        config_overrides['conda_env_name'] = args.conda_env
    if args.image:
        config_overrides['image'] = args.image
    if args.extra_envs:
        default_config = VolcanoConfig()
        config_overrides['extra_envs'] = default_config.extra_envs + args.extra_envs
    
    # 初始化部署
    deployer = VolcanoDeployment(config_overrides if config_overrides else None)
    
    # 打印部署配置
    logger.info("\n部署配置摘要:")
    logger.info(f"任务命令: {args.task_cmd}")
    logger.info(f"GPU数量: {args.num_gpus}")
    logger.info(f"conda环境: {deployer.config.conda_env_name}")
    logger.info(f"任务名称: {deployer.config.task_name}")
    logger.info(f"队列名称: {deployer.config.queue_name}")
    logger.info(f"镜像名称: {deployer.config.image}")
    if args.extra_envs:
        logger.info(f"额外环境变量: {args.extra_envs}")
    
    # 确认部署
    if not args.yes:
        confirm = input("\n继续部署吗？ [y/N]: ")
        if confirm.lower() != 'y':
            logger.warning("用户取消了部署")
            return
    else:
        logger.info("自动确认部署（--yes标志）")
    
    # 执行部署
    try:
        result = deployer.deploy(args.task_cmd, num_gpus=args.num_gpus, num_replicas=args.num_replicas)
        
        # 打印部署结果
        if result.returncode == 0:
            logger.success("部署成功完成")
        else:
            logger.error("部署失败")
            
        if result.stdout:
            logger.info(f"输出: {result.stdout}")
        if result.stderr:
            logger.warning(f"错误: {result.stderr}")
            
    except Exception as e:
        logger.exception("部署失败，出现异常")
        raise

if __name__ == "__main__":
    main()  # 执行主函数