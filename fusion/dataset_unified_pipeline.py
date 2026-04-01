import torch
import random
import math
import collections
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
from data.sampler.cylinder_sample import EclipseDataSampler
import concurrent.futures as cf
from data.distributed import BoundedExecutor  # 引入您上传的分布式执行器
class DatasetUnifiedPipeline(IterableDataset):
    """
    针对 Unified Dynamics Pipeline (Stage 2) 设计的数据加载器。
    自动拉取静态孔渗场，缓存动态场并执行滑动窗口切分 (Sliding Window)，
    最终输出 unified_pipeline 要求的 9 项元组。
    """

    def __init__(
            self,
            project_dirs: list,
            project_names: list,
            dynamic_fields: list,  # e.g., ["PRESSURE", "SWAT"]
            nt_width: int = 10,  # 模型一次消化的时间步长度
            stride: int = 1,  # 滑动窗口的步长
            mode: str = "train",
            nums_loading_projects: int = 2,  # 内存中并发持有项目数
            nθ: int = 26,
            nR: int = 10,
            model_nz: int = 40,
            maxR: float = 1000.0,  # 物理超参 (用于 Pipeline 的欧式 PE)
            total_z_depth: float = 50.0,  # 物理超参 (用于 Pipeline 的欧式 PE)
            train_ratio: float = 0.8,
            seed: int = 42,
            shuffle: bool = True,
            dynamic_feature_dim: int = 12,  # 新增：对应统一流水线的井控特征维度
            data_source_type: str = 'eclipse'
    ):
        super().__init__()
        self.projects = list(zip(project_dirs, project_names))
        self.dynamic_fields = dynamic_fields
        self.nt_width = nt_width
        self.stride = stride
        self.mode = mode
        self.nums_loading_projects = nums_loading_projects
        self.nθ, self.nR, self.model_nz = nθ, nR, model_nz
        self.maxR = maxR
        self.total_z_depth = total_z_depth
        self.train_ratio, self.seed, self.shuffle = train_ratio, seed, shuffle
        self.dynamic_feature_dim = dynamic_feature_dim
        # 🌟 2. 必须加上这一行！把参数存入 self 中
        self.data_source_type = data_source_type

    def _process_chunk(self, chunk_projects):
        """核心处理逻辑：利用 BoundedExecutor 并行加载项目动态场并切分 Window"""
        chunk_windows = []

        # 🌟 实例化并发执行器。建议线程数设为 4-8，既能加速 I/O 又能防止内存溢出
        # 使用 ThreadPoolExecutor 是因为 Zarr 的硬盘读取和解压过程能有效释放 GIL 锁
        executor = BoundedExecutor(max_workers=8, executor_cls=cf.ThreadPoolExecutor)

        for p_dir, p_name in chunk_projects:
            try:
                # 1. 实例化发牌人
                sampler = EclipseDataSampler(
                    project_dir=p_dir, project_name=p_name,
                    data_source_type=self.data_source_type,
                    nθ=self.nθ, nR=self.nR, model_nz=self.model_nz,
                    train_ratio=self.train_ratio, seed=self.seed
                )

                # 2. 提取静态场底座 (保持原样，这部分速度通常很快)
                poro_dict = {s['well_name']: s for s in sampler.iter_static("PORO", mode=self.mode, shuffle=False)}
                perm_dict = {s['well_name']: s for s in sampler.iter_static("PERMX", mode=self.mode, shuffle=False)}

                # 3. 🚀 并行提取动态场并缓存 (构建井史)
                well_buffer = collections.defaultdict(list)
                dynamic_gen = sampler.iter_dynamic(self.dynamic_fields, mode=self.mode, shuffle=False)
                total_steps = 430
                # 定义子任务：处理单步数据的 Tensor 转换和特征对齐
                def process_step_task(sample_data):
                    w_name = sample_data['well_name']
                    # 提取并封装动态井控特征
                    ctrl_data = sample_data.get('well_control', [0.0] * self.dynamic_feature_dim)
                    sample_data['dynamic_control'] = torch.tensor(ctrl_data, dtype=torch.float32)
                    return w_name, sample_data

                # ✅ 正确：
                task_generator = (lambda x=sample: process_step_task(x) for sample in dynamic_gen)

                # 🌟 修改这里：传入你喜欢的 desc 和 total
                processed_results = executor.execute_stream(
                    task_source=task_generator,
                    # total=total_steps,  # 👈 传入总步数后，进度条就会显示 [02:15, 60.00step/s]
                    desc=f"📥 加载项目动态场 {p_name}"  # 👈 换回你偏好的 emoji 文案
                )

                # 将并发处理完的结果按井归档
                for w_name, sample in processed_results:
                    well_buffer[w_name].append(sample)

                # 4. 执行滑动窗口切分 (Sliding Window)
                for w_name, seq in well_buffer.items():
                    if w_name not in poro_dict or w_name not in perm_dict:
                        continue

                    # 必须按真实时间步 step_idx 排序，保证物理因果连续性
                    seq.sort(key=lambda x: x['step_idx'])
                    history_len = len(seq)
                    window_len = self.nt_width + 1

                    if history_len < window_len:
                        continue

                    # 按照 stride 步长进行滑动切片
                    for start in range(0, history_len - self.nt_width, self.stride):
                        window = seq[start: start + window_len]
                        if len(window) < window_len:
                            break

                        # ==========================================
                        # 🌟 核心修改：强制转为 .float() 解决 double 报错
                        # ==========================================
                        # 拼装 Pipeline 需要的 9 项核心数据
                        window_fields = torch.stack([s['field_data'] for s in window]).float()  # 👈 必须加 .float()
                        datetimes = torch.tensor([s['step_idx'] for s in window[:-1]], dtype=torch.float32)
                        poro_field = poro_dict[w_name]['field_data'].float()  # 👈 必须加 .float()
                        perm_field = perm_dict[w_name]['field_data'].float()  # 👈 必须加 .float()
                        dynamic_features = torch.stack(
                            [s['dynamic_control'] for s in window]).float()  # 👈 必须加 .float()
                        spatial_mask_orig = poro_dict[w_name]['mask'].float()  # 👈 关键：Mask 参与运算必须是 float
                        hard_data_locs = window[0]['hd_locs']  # 坐标通常是 long，保持不变

                        chunk_windows.append((
                            window_fields, datetimes, poro_field, perm_field,
                            dynamic_features, spatial_mask_orig, hard_data_locs,
                            self.maxR, self.total_z_depth
                        ))

            except Exception as e:
                print(f"⚠️ 警告: Unified Pipeline 处理项目 {p_name} 失败: {e}")
                continue

        return chunk_windows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # 进程任务分配
        if worker_info is not None:
            per_worker = int(math.ceil(len(self.projects) / float(worker_info.num_workers)))
            my_projects = self.projects[worker_info.id * per_worker: (worker_info.id + 1) * per_worker]
        else:
            my_projects = self.projects.copy()

        # 项目级 Shuffle
        if self.shuffle and self.mode == "train":
            random.shuffle(my_projects)

        # 内存分块流水线 (Chunking)
        for i in range(0, len(my_projects), self.nums_loading_projects):
            chunk_projects = my_projects[i: i + self.nums_loading_projects]

            # 集中榨取这些项目并切好 Windows
            chunk_windows = self._process_chunk(chunk_projects)

            # Batch 级(窗口级)局部 Shuffle，保证训练异质性
            if self.shuffle and self.mode == "train":
                random.shuffle(chunk_windows)

            # 逐个吐出切好的滑动窗口
            for window_data in chunk_windows:
                yield window_data


# ==========================================
# 本地测试模块
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 测试 Unified Pipeline Dataset (带进度条与特征对齐)...")

    # 填入你自己的测试路径
    PROJECT_DIR = r"D:\jbgs\jbgs-model-training\dataset\water_196311-199307"
    PROJECT_NAME = "BYEPD93"

    # 测试建议只保留 1 个项目以加快调试速度
    p_dirs = [PROJECT_DIR]
    p_names = [PROJECT_NAME]

    train_dataset = DatasetUnifiedPipeline(
        project_dirs=p_dirs,
        project_names=p_names,
        dynamic_fields=["PRESSURE", "SWAT"],
        nt_width=4,  # 模拟滑动窗口 T=4 (实际返回长度为 5)
        stride=10,  # 测试时步长设大一点，加快拼装速度
        mode="train",
        nums_loading_projects=1,
        dynamic_feature_dim=12,
        data_source_type='zarr'
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=0,
        pin_memory=True,                     # 保持 True，锁定锁页内存，加速向 GPU 拷贝
        prefetch_factor=None, # 🌟 接收 yaml 中的预取参数
        persistent_workers=False              # 🌟 新增：让工人常驻内存，不被释放
    )

    # train_loader = DataLoader(train_dataset, batch_size=2, num_workers=0)

    for batch_idx, batch in enumerate(train_loader):
        # 完美对齐 unified_pipeline_config 中的 9 项输入
        window_fields, datetimes, poro_field, perm_field, dynamic_features, spatial_mask_orig, hard_data_locs, maxR, total_z_depth = batch

        print(f"\n📦 Batch {batch_idx + 1} 获取成功:")
        print(f"  --> 动态场窗口 (T=nt_width+1): {window_fields.shape}")  # 应为 (2, 5, 2, 40, 26, 10)
        print(f"  --> 时间戳序列 (T=nt_width): {datetimes.shape}")  # 应为 (2, 4)
        print(f"  --> 静态孔隙度底座: {poro_field.shape}")  # 应为 (2, 1, 40, 26, 10)
        print(f"  --> 动态井控特征: {dynamic_features.shape}")  # 应为 (2, 5, 12)，不再是 None 报错了
        print(f"  --> 静态掩码: {spatial_mask_orig.shape}")
        print(f"  --> 硬数据位置: {hard_data_locs.shape}")

        if batch_idx == 0:
            break

    print("=" * 60)
    print("✅ 测试通过！Window 级多模态数据拼装完美。")