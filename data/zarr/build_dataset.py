# data/zarr/build_dataset.py
import os
# 🌟 必须放在最顶端！锁定所有底层 C++ 数学库的线程数，防止多进程环境下的线程大爆炸与崩溃
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import numpy as np
import pandas as pd
import concurrent.futures as cf
from pathlib import Path
from collections import defaultdict

import zarr
try:
    from zarr.codecs import BloscCodec, BloscShuffle
    import numcodecs
    numcodecs.blosc.use_threads = False
    numcodecs.blosc.set_nthreads(1)
except ImportError:
    raise ImportError("必须使用 Zarr 3.x 版本！请升级 zarr (例如: pip install zarr>=3.0.0)。")

from data.distributed import BoundedExecutor
from data.eclipse.well_data import EclipseWellData 

def compute_chunk_shape(shape, dtype, max_mb=2, avg_compress_ratio=1/2):
    itemsize = np.dtype(dtype).itemsize
    max_bytes = int(max_mb / avg_compress_ratio) * 1024 * 1024
    max_elems = max_bytes // itemsize
    chunks = [1] * len(shape)
    remaining = max_elems
    for axis in reversed(range(len(shape))):
        dim = shape[axis]
        if remaining >= dim:
            chunks[axis] = dim
            remaining //= dim if dim > 0 else 1
        else:
            chunks[axis] = max(1, remaining)
            break
    return tuple(chunks)

def _create_zarr_array(group, name, data):
    chunks = compute_chunk_shape(data.shape, data.dtype)
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)
    group.create_array(name=name, data=data, chunks=chunks, compressors=[compressor], overwrite=True)

# =============================================================================
# 原子任务 1：基础网格计算 (就地写入 static.attrs)
# =============================================================================
class GridTask:
    def __init__(self, zarr_root_path, egrid_path, proj_name):
        self.zarr_root_path = str(zarr_root_path)
        self.egrid_path = str(egrid_path)
        self.proj_name = proj_name

    def __call__(self):
        from opm.io.ecl import EclFile
        from data.numba_utils import build_act_remap, get_boundary_quads_numba, compute_bounds_numba
        
        efile = EclFile(self.egrid_path)

        def _ecl_get(target_kw):
            for index, kw in enumerate(efile.arrays):
                if kw[0] == target_kw: return efile[index]
            return None

        gridhead = np.array(_ecl_get("GRIDHEAD"))
        nx, ny, nz = int(gridhead[1]), int(gridhead[2]), int(gridhead[3])
        
        coord = np.array(_ecl_get("COORD"), dtype=np.float64).ravel()
        zcorn = -np.array(_ecl_get("ZCORN"), dtype=np.float64).ravel()
        
        actnum_raw = _ecl_get("ACTNUM")
        if actnum_raw is not None:
            actnum = np.array(actnum_raw).astype(bool).reshape(nz, ny, nx)
        else:
            actnum = np.ones((nz, ny, nx), dtype=bool)

        pillars_per_block = (ny + 1) * (nx + 1)
        block_size = pillars_per_block * 6
        nblocks = coord.size // block_size
        if nblocks not in (1, nz): nblocks = 1
            
        coord_block = coord[:block_size].reshape(ny + 1, nx + 1, 6)
        xt, yt = coord_block[..., 0], coord_block[..., 1]
        xb, yb = coord_block[..., 3], coord_block[..., 4]
        radial = bool(((np.abs(yt) <= 360) & (np.abs(yb) <= 360)).mean() > 0.98 and ((xt >= 0) & (xb >= 0)).mean() > 0.98)

        act_remap = build_act_remap(actnum, nx, ny, nz)
        quads, _ = get_boundary_quads_numba(coord, zcorn, actnum, act_remap, nx, ny, nz, nblocks, radial, 1e-12)
        mins, maxs = compute_bounds_numba(quads)
        origin = tuple((mins + maxs) / 2.0)
        sz, sy, sx = maxs - mins

        root = zarr.open_group(self.zarr_root_path, mode='r+')
        _create_zarr_array(root, 'coord_1d', coord)
        _create_zarr_array(root, 'zcorn_1d', zcorn)
        _create_zarr_array(root, 'actnum_3d', actnum)    

        # 🌟 就地落盘：写入子组专属 attrs，杜绝并发冲突
        static_grp = root['static']
        static_grp.attrs['frame_dim'] = (nz, ny, nx)
        static_grp.attrs['frame_size'] = (sz, sy, sx)
        static_grp.attrs['origin'] = origin
        static_grp.attrs['nblocks'] = int(nblocks)
        static_grp.attrs['radial'] = radial

        return {'msg': f"✅ [{self.proj_name}] 网格几何 (.EGRID) 解析与压盘完成"}

# =============================================================================
# 原子任务 2：单一场静态属性提取
# =============================================================================
class StaticFieldTask:
    def __init__(self, zarr_root_path, init_path, field_name, proj_name):
        self.zarr_root_path = str(zarr_root_path)
        self.init_path = str(init_path)
        self.field_name = field_name
        self.proj_name = proj_name

    def __call__(self):
        from opm.io.ecl import EclFile
        efile = EclFile(self.init_path)
        
        def _ecl_get(target_kw):
            for index, kw in enumerate(efile.arrays):
                if kw[0] == target_kw: return efile[index]
            return None

        raw_data = _ecl_get(self.field_name)
        if raw_data is None:
            raise ValueError(f"在 {self.init_path} 中未找到静态场: {self.field_name}")

        data = np.array(raw_data, dtype=np.float64).ravel()
        
        root = zarr.open_group(self.zarr_root_path, mode='r+')
        _create_zarr_array(root['static'], self.field_name, data)

        return {'msg': f"✅ [{self.proj_name}] 静态场 {self.field_name} 压盘完成"}

# =============================================================================
# 原子任务 3：单一时间步动态场提取 (就地写入 step.attrs)
# =============================================================================
class DynamicStepTask:
    def __init__(self, zarr_root_path, x_file_path, dynamic_fields, proj_name):
        self.zarr_root_path = str(zarr_root_path)
        self.x_file_path = str(x_file_path)
        self.dynamic_fields = dynamic_fields
        self.proj_name = proj_name

    def __call__(self):
        from opm.io.ecl import ERst
        rst = ERst(self.x_file_path)
        step = rst.report_steps[0]

        try:
            intehead = rst["INTEHEAD", step]
            day, month, year = int(intehead[64]), int(intehead[65]), int(intehead[66])
            from datetime import datetime
            dt = datetime(year, month, day).isoformat() if (year > 0 and month > 0 and day > 0) else None
        except Exception:
            dt = None
            
        root = zarr.open_group(self.zarr_root_path, mode='r+')
        step_grp = root['dynamic'].create_group(f"Step_{step}", overwrite=True)

        for f in self.dynamic_fields:
            try:
                data = np.array(rst[f, step]).ravel()
                _create_zarr_array(step_grp, f, data)
            except Exception:
                pass 

        # 🌟 就地落盘：将当前物理时间存入该 Step 的属性中
        step_grp.attrs['datetime'] = dt

        return {'msg': f"✅ [{self.proj_name}] 动态场 Step {step:03d} 压盘完成"}

# =============================================================================
# 原子任务 4：Summary 碎纸机提取 (完美解决单线程 100 秒瓶颈)
# =============================================================================
class SmryChunkTask:
    def __init__(self, zarr_root_path, smspec_path, target_keys, chunk_id, proj_name):
        self.zarr_root_path = Path(zarr_root_path)
        self.smspec_path = str(smspec_path)
        self.target_keys = target_keys
        self.chunk_id = chunk_id
        self.proj_name = proj_name

    def __call__(self):
        from opm.io.ecl import ESmry
        smry = ESmry(self.smspec_path)

        # 加载公共时间轴
        df_time = pd.read_parquet(self.zarr_root_path / "df_time.parquet")

        entity_dict = {}
        for k in self.target_keys:
            prop, entity = k.split(':', 1)
            if entity not in entity_dict: entity_dict[entity] = {}
            entity_dict[entity][prop] = smry[k]

        dfs = []
        for entity, prop_data in entity_dict.items():
            df_e = pd.DataFrame(prop_data)
            df_e['well'] = entity
            dfs.append(df_e)

        if dfs:
            df_well = pd.concat(dfs, ignore_index=True)
            # 补齐时间列以兼容下游画图
            time_cols_repeated = pd.concat([df_time] * len(dfs), ignore_index=True)
            for col in df_time.columns: 
                df_well[col] = time_cols_repeated[col]
                
            base_cols = list(df_time.columns) + ['well']
            other_cols = [c for c in df_well.columns if c not in base_cols]
            
            # 🌟 独立保存为 Parquet 切片
            chunk_path = self.zarr_root_path / "smry_well_chunks" / f"chunk_{self.chunk_id}.parquet"
            df_well[base_cols + other_cols].to_parquet(chunk_path, engine="pyarrow")

        return {'msg': f"✅ [{self.proj_name}] 产量摘要 Chunk {self.chunk_id} 提取完成"}

# =============================================================================
# 原子任务 5：井控与轨迹提取 (就地写入 trajectories.attrs)
# =============================================================================
class WellDataTask:
    def __init__(self, zarr_root_path, source_dir, proj_name):
        self.zarr_root_path = Path(zarr_root_path)
        self.source_dir = Path(source_dir)
        self.proj_name = proj_name

    def __call__(self):
        from opm.io.ecl import EclFile
        well = EclipseWellData(str(self.source_dir), self.proj_name)
        
        egrid_path = self.source_dir / f"{self.proj_name}.EGRID"
        actnum = None
        nz, ny, nx = 0, 0, 0
        if egrid_path.exists():
            efile = EclFile(str(egrid_path))
            def _ecl_get(target_kw):
                for i, kw in enumerate(efile.arrays):
                    if kw[0] == target_kw: return efile[i]
                return None
            
            gridhead = np.array(_ecl_get("GRIDHEAD"))
            if gridhead is not None:
                nx, ny, nz = int(gridhead[1]), int(gridhead[2]), int(gridhead[3])
                actnum_raw = _ecl_get("ACTNUM")
                if actnum_raw is not None:
                    actnum = np.array(actnum_raw).astype(bool).reshape(nz, ny, nx)
                else:
                    actnum = np.ones((nz, ny, nx), dtype=bool)

        active_completions = []
        wells_to_delete = set()
        
        for w_name, w_info in well.static_wells.items():
            comps = w_info.get('completions', [])
            has_active = False
            for comp in comps:
                i, j, k = comp['I'] - 1, comp['J'] - 1, comp['K'] - 1
                if actnum is not None:
                    if 0 <= k < nz and 0 <= j < ny and 0 <= i < nx and actnum[k, j, i]:
                        has_active = True
                        active_completions.append({'well': w_name, 'I': comp['I'], 'J': comp['J'], 'K': comp['K']})
                else:
                    has_active = True 
                    active_completions.append({'well': w_name, 'I': comp['I'], 'J': comp['J'], 'K': comp['K']})
            if not has_active:
                wells_to_delete.add(w_name)

        if active_completions:
            pd.DataFrame(active_completions).to_parquet(self.zarr_root_path / "df_completions.parquet", engine="pyarrow")

        if well.df_events is not None and not well.df_events.empty:
            df_events = well.df_events
            if wells_to_delete:
                df_events = df_events[~df_events['well'].isin(wells_to_delete)]
            df_events.to_parquet(self.zarr_root_path / "df_events.parquet", engine="pyarrow")

        root = zarr.open_group(str(self.zarr_root_path), mode='a')
        trj_grp = root['trajectories']
        
        # 🌟 就地落盘：将日期列表存入子组 attrs，彻底干掉主进程收集
        trj_grp.attrs['dates'] = [d.isoformat() for d in well.dates]

        if well.trj_data:
            for w_name, trj_array in well.trj_data.items():
                if w_name not in wells_to_delete:
                    _create_zarr_array(trj_grp, w_name, np.array(trj_array, dtype=np.float32))

        return {'msg': f"✅ [{self.proj_name}] 井控数据 (.DATA) 解析与落盘完成"}


# =============================================================================
# 任务发生器 (支持 Summary 分块并发)
# =============================================================================
def zarr_file_level_generator(source_dir, proj_name, target_zarr_dir, static_fields, dynamic_fields, smry_well_props):
    root = zarr.open_group(str(target_zarr_dir), mode='w')
    root.create_group('static', overwrite=True)
    root.create_group('dynamic', overwrite=True)
    root.create_group('trajectories', overwrite=True)

    # print("\n📂 正在解析:", source_dir, proj_name)

    # 🌟 Summary 并发切割逻辑
    smspec_path = source_dir / f"{proj_name}.SMSPEC"
    if smspec_path.exists():
        try:
            from opm.io.ecl import ESmry
            smry = ESmry(str(smspec_path))
            all_keys = list(smry.keys())
            time_days = smry['TIME'] if 'TIME' in all_keys else np.arange(len(smry))
            
            df_time = pd.DataFrame({'step_idx': np.arange(len(time_days)), 'time_days': time_days})
            if all(k in all_keys for k in ['YEAR', 'MONTH', 'DAY']):
                df_time['date'] = pd.to_datetime({'year': smry['YEAR'], 'month': smry['MONTH'], 'day': smry['DAY']}, errors='coerce')
            
            # 公共时间轴落盘，供子进程读取
            df_time.to_parquet(target_zarr_dir / "df_time.parquet", engine="pyarrow")

            # 🌟 修复：补回全场属性 (Field Properties) 的压盘逻辑！
            field_keys = [k for k in all_keys if ':' not in k and k.startswith('F')]
            if field_keys:
                df_field = pd.concat([df_time, pd.DataFrame({k: smry[k] for k in field_keys})], axis=1)
                df_field.to_parquet(target_zarr_dir / "df_field.parquet", engine="pyarrow")
            
            # 筛选你指定的 W 级别属性
            target_props = set(smry_well_props)
            well_keys = [k for k in all_keys if ':' in k and k.split(':')[0] in target_props]
            
            # 按井进行分块
            well_to_keys = defaultdict(list)
            for k in well_keys: well_to_keys[k.split(':')[1]].append(k)
                
            wells = list(well_to_keys.keys())
            chunk_size = 50 # 每 50 口井切一个分块任务
            (target_zarr_dir / "smry_well_chunks").mkdir(exist_ok=True)
            
            for i in range(0, len(wells), chunk_size):
                chunk_wells = wells[i:i+chunk_size]
                chunk_keys = []
                for w in chunk_wells: chunk_keys.extend(well_to_keys[w])
                yield SmryChunkTask(target_zarr_dir, smspec_path, chunk_keys, i//chunk_size, proj_name)
        except Exception as e:
            print(f"  ⚠️ [报错] 产量摘要分解失败: {e}")
    else:
        print(f"  ⚠️ [跳过] 未找到产量摘要文件: {smspec_path.name}")

    data_path = source_dir / f"{proj_name}.DATA"
    if data_path.exists(): yield WellDataTask(target_zarr_dir, source_dir, proj_name)

    egrid_path = source_dir / f"{proj_name}.EGRID"
    if egrid_path.exists(): yield GridTask(target_zarr_dir, egrid_path, proj_name)
        
    init_path = source_dir / f"{proj_name}.INIT"
    if init_path.exists():
        for field in static_fields: yield StaticFieldTask(target_zarr_dir, init_path, field, proj_name)

    pattern = re.compile(f"^{re.escape(proj_name)}\\.X\\d+$", re.IGNORECASE)
    x_files = [f for f in os.listdir(source_dir) if pattern.match(f)]
    if x_files:
        for f in x_files: yield DynamicStepTask(target_zarr_dir, source_dir / f, dynamic_fields, proj_name)


if __name__ == "__main__":
    import time
    print("="*80)
    print("🚀 启动 Zarr 数据集构建 (无管道负载、彻底解耦、Summary切片并发)...")
    
    projects = [
        {"dir": r"E:\tocug\1c\water_196311-199307", "name": "BYEPD93"},
        {"dir": r"E:\tocug\4c\2z\sim", "name": "1_E100"},
        {"dir": r"E:\tocug\4c\2z\sim", "name": "1-2_E100"},
        {"dir": r"E:\tocug\4c\4-6\x4-6sm20230518", "name": "X4-6X_E100"},
    ]

    OUTPUT_DIR = Path("D:/jbgs/jbgs-model-training/dataset")
    STATIC_FIELDS = ["PORO", "PERMX"]
    DYNAMIC_FIELDS = ["PRESSURE", "SWAT"]
    # 🌟 新增：只提取你关心的井级别属性，大幅降低读取负担
    SMRY_WELL_PROPERTIES = ["WOPR", "WWPR", "WGPR", "WBHP", "WWCT"]
    
    start_time = time.time()
    executor = BoundedExecutor(max_workers=24, executor_cls=cf.ProcessPoolExecutor)
    tasks = []
    
    for p in projects:
        proj_name = p["name"]
        source_dir = Path(p["dir"])
        target_zarr_dir = Path(OUTPUT_DIR) / source_dir.name / f"{proj_name}.zarr"
        target_zarr_dir.mkdir(parents=True, exist_ok=True)
        
        tasks.extend(list(zarr_file_level_generator(source_dir, proj_name, target_zarr_dir, STATIC_FIELDS, DYNAMIC_FIELDS, SMRY_WELL_PROPERTIES)))
        
    if tasks:

        results = executor.execute_stream(task_source=tasks, total_est=len(tasks), desc=f"ETL: ")
        
        # 🌟 主进程收网：没有任何字典收集，纯净至极！
        for res in results:
            if '⚠️' in res.get('msg', '') or '❌' in res.get('msg', ''):
                print(f"  {res['msg']}")

    print(f"\n⏱️ 全局处理总耗时: {time.time() - start_time:.2f} 秒")
    print("="*80)