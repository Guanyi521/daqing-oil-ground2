import time
import numpy as np
import pandas as pd
import concurrent.futures as cf
from pathlib import Path
from datetime import datetime

# 导入原生 Eclipse 接口
from data.eclipse.model_data import EclipseModelData
from data.eclipse.well_data import EclipseWellData
from data.eclipse.well_smry import EclipseWellSmry

# 导入极速 Zarr 接口
from data.zarr.zarr_data import ZarrModelData, ZarrWellData, ZarrWellSmry
from data.distributed import BoundedExecutor


# =============================================================================
# 测试任务 1：严格遍历 IModelData 契约
# =============================================================================
class TestModelDataTask:
    def __init__(self, eclipse_dir, zarr_parent_dir, proj_name):
        self.eclipse_dir = Path(eclipse_dir)
        self.zarr_parent_dir = Path(zarr_parent_dir)
        self.proj_name = proj_name

    def __call__(self):
        errors = []
        try:
            ecl = EclipseModelData(str(self.eclipse_dir), self.proj_name)
            zarr = ZarrModelData(str(self.zarr_parent_dir), self.proj_name)
            
            # 1. get_3Dframe_dim
            if ecl.get_3Dframe_dim() != zarr.get_3Dframe_dim(): errors.append("get_3Dframe_dim 不一致")
            
            # 2. get_3Dframe_size
            if not np.allclose(ecl.get_3Dframe_size(), zarr.get_3Dframe_size()): 
                errors.append("get_3Dframe_size 不一致")
                
            # 3. get_model_actnum
            e_act, z_act = ecl.get_model_actnum(), zarr.get_model_actnum()
            if not np.array_equal(e_act, z_act): errors.append("get_model_actnum 不一致")
                
            # 4. get_origin
            if not np.allclose(ecl.get_origin(), zarr.get_origin()): errors.append("get_origin 不一致")

            # 5. get_top_2d_centers
            e_cx, e_cy = ecl.get_top_2d_centers()
            z_cx, z_cy = zarr.get_top_2d_centers()
            if not np.allclose(e_cx, z_cx) or not np.allclose(e_cy, z_cy): errors.append("get_top_2d_centers 不一致")

            # 6. get_cell_centers
            nz, ny, nx = zarr.nz, zarr.ny, zarr.nx
            ti, tj, tk = [0, nx//2], [0, ny//2], [0, nz//2]
            if not np.allclose(ecl.get_cell_centers(ti, tj, tk), zarr.get_cell_centers(ti, tj, tk)):
                errors.append("get_cell_centers 不一致")

            # 7. get_static_fields
            avail_static = list(zarr.root['static'].keys())
            if avail_static:
                tf = avail_static[0]
                if not np.allclose(ecl.get_static_fields([tf]), zarr.get_static_fields([tf]), equal_nan=True): 
                    errors.append(f"get_static_fields ('{tf}') 不一致")

            # 8. get_dynamic_steps
            if ecl.get_dynamic_steps() != zarr.get_dynamic_steps(): errors.append("get_dynamic_steps 不一致")

            # 9. get_dynamic_datetimes (比较 key 和时间对象是否相等)
            e_dts, z_dts = ecl.get_dynamic_datetimes(), zarr.get_dynamic_datetimes()
            if e_dts != z_dts: errors.append("get_dynamic_datetimes 不一致")

            # 10. get_dynamic_fields
            dyn_steps = zarr.get_dynamic_steps()
            if dyn_steps:
                ts = dyn_steps[-1]
                avail_dyn = list(zarr.root['dynamic'][f'Step_{ts}'].keys())
                if avail_dyn:
                    tdf = avail_dyn[0]
                    if not np.allclose(ecl.get_dynamic_fields(ts, [tdf]), zarr.get_dynamic_fields(ts, [tdf]), equal_nan=True):
                        errors.append(f"get_dynamic_fields ('{tdf}') 不一致")

        except Exception as e:
            errors.append(f"运行时崩溃: {e}")

        if errors:
            return {'msg': f"❌ [{self.proj_name}] IModelData 失败:\n    " + "\n    ".join(errors)}
        return {'msg': f"✅ [{self.proj_name}] IModelData 测试通过"}


# =============================================================================
# 测试任务 2：严格遍历 IWellData 契约
# =============================================================================
class TestWellDataTask:
    def __init__(self, eclipse_dir, zarr_parent_dir, proj_name):
        self.eclipse_dir = Path(eclipse_dir)
        self.zarr_parent_dir = Path(zarr_parent_dir)
        self.proj_name = proj_name

    def __call__(self):
        errors = []
        try:
            ecl = EclipseWellData(str(self.eclipse_dir), self.proj_name)
            zarr = ZarrWellData(str(self.zarr_parent_dir), self.proj_name)

            # 1. 属性 static_wells 验证 (过滤死井比对)
            survived_wells = list(zarr.static_wells.keys())
            for w in survived_wells:
                e_comps = ecl.static_wells[w]['completions']
                z_comps = zarr.static_wells[w]['completions']
                if len(z_comps) > len(e_comps): 
                    errors.append(f"static_wells: 井 {w} Zarr出现了不存在的射孔")
                    break
                for zc in z_comps:
                    if not any(ec['I'] == zc['I'] and ec['J'] == zc['J'] and ec['K'] == zc['K'] for ec in e_comps):
                        errors.append(f"static_wells: 井 {w} 坐标不匹配")
                        break

            # 2. 属性 trj_data 验证
            for w, trj in zarr.trj_data.items():
                if w in ecl.trj_data and not np.allclose(trj, ecl.trj_data[w]):
                    errors.append(f"trj_data: 井 {w} 轨迹数据不一致")

            test_dts = zarr.dates[:3] if len(zarr.dates) >= 3 else [0]
            
            # 3. get_steps_info
            if len(zarr.get_steps_info(test_dts)) > len(ecl.get_steps_info(test_dts)):
                errors.append("get_steps_info 不一致 (Zarr 记录超出 Eclipse)")

            # 4. get_step_to_active_wells_map
            e_map, z_map = ecl.get_step_to_active_wells_map(test_dts), zarr.get_step_to_active_wells_map(test_dts)
            for step in e_map:
                if step in z_map and not set(z_map[step]).issubset(set(e_map[step])):
                    errors.append(f"get_step_to_active_wells_map: Step {step} 状态不匹配")
                    
            # 5. get_wells_by_step_idx_and_name
            test_w = survived_wells[0] if survived_wells else None
            if test_w:
                e_df = ecl.get_wells_by_step_idx_and_name(test_dts[0], test_w, ['step_idx', 'well'])
                z_df = zarr.get_wells_by_step_idx_and_name(test_dts[0], test_w, ['step_idx', 'well'])
                if len(z_df) > len(e_df):
                    errors.append("get_wells_by_step_idx_and_name 不一致")

        except Exception as e:
            errors.append(f"运行时崩溃: {e}")

        if errors:
            return {'msg': f"❌ [{self.proj_name}] IWellData 失败:\n    " + "\n    ".join(errors)}
        return {'msg': f"✅ [{self.proj_name}] IWellData 测试通过"}


# =============================================================================
# 测试任务 3：严格遍历 IWellSmry 契约
# =============================================================================
class TestWellSmryTask:
    def __init__(self, eclipse_dir, zarr_parent_dir, proj_name, test_props):
        self.eclipse_dir = Path(eclipse_dir)
        self.zarr_parent_dir = Path(zarr_parent_dir)
        self.proj_name = proj_name
        self.test_props = test_props

    def __call__(self):
        errors = []
        try:
            # 初始化 (注意：如果是你说的极其耗时的原生类，测试这一步的确会耗时)
            ecl = EclipseWellSmry(str(self.eclipse_dir), self.proj_name, lazy=False)
            zarr = ZarrWellSmry(str(self.zarr_parent_dir), self.proj_name)

            # 1. 验证 DataFrame 属性 df_field 和 df_well 是否加载
            if zarr.df_field is None: errors.append("df_field 属性为空")
            if zarr.df_well is None: errors.append("df_well 属性为空")

            # 2. get_well_data 契约测试 (要求返回 pd.DataFrame)
            if ecl.df_well is not None and zarr.df_well is not None and not zarr.df_well.empty:
                test_w = zarr.df_well['well'].iloc[0]
                
                # 调用基类接口获取 DataFrame
                e_df = ecl.get_well_data(test_w, self.test_props)
                z_df = zarr.get_well_data(test_w, self.test_props)
                
                if not isinstance(z_df, pd.DataFrame):
                    errors.append(f"get_well_data 没有返回 DataFrame，返回了 {type(z_df)}")
                elif not e_df.empty and not z_df.empty:
                    # 将行排序并重置索引，以便按列对齐比对
                    e_df = e_df.sort_values('step_idx').reset_index(drop=True)
                    z_df = z_df.sort_values('step_idx').reset_index(drop=True)
                    
                    for prop in self.test_props:
                        if prop in z_df.columns and prop in e_df.columns:
                            if not np.allclose(e_df[prop].values, z_df[prop].values, equal_nan=True, rtol=1e-4):
                                errors.append(f"get_well_data: 属性 {prop} 数值比对不一致")

        except Exception as e:
            errors.append(f"运行时崩溃: {e}")

        if errors:
            return {'msg': f"❌ [{self.proj_name}] IWellSmry 失败:\n    " + "\n    ".join(errors)}
        return {'msg': f"✅ [{self.proj_name}] IWellSmry 测试通过"}


def interface_test_generator(eclipse_dir, zarr_parent_dir, proj_name, test_smry_props):
    yield TestModelDataTask(eclipse_dir, zarr_parent_dir, proj_name)
    yield TestWellDataTask(eclipse_dir, zarr_parent_dir, proj_name)
    yield TestWellSmryTask(eclipse_dir, zarr_parent_dir, proj_name, test_smry_props)


if __name__ == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    
    print("="*80)
    print("🚀 启动 Zarr 严格契约平替极限并发测试...")
    
    projects = [
        {"dir": r"E:\tocug\1c\water_196311-199307", "name": "BYEPD93"},
        {"dir": r"E:\tocug\4c\2z\sim", "name": "1_E100"},
        {"dir": r"E:\tocug\4c\2z\sim", "name": "1-2_E100"},
        {"dir": r"E:\tocug\4c\4-6\x4-6sm20230518", "name": "X4-6X_E100"},
    ]
    
    OUTPUT_DIR = Path("D:/jbgs/jbgs-model-training/dataset")
    SMRY_WELL_PROPERTIES = ["WOPR", "WWPR", "WGPR", "WBHP", "WWCT"]

    start_time = time.time()
    executor = BoundedExecutor(max_workers=12, executor_cls=cf.ProcessPoolExecutor)
    tasks = []
    
    for p in projects:
        proj_name = p["name"]
        eclipse_dir = Path(p["dir"])
        zarr_parent_dir = OUTPUT_DIR / eclipse_dir.name
        tasks.extend(list(interface_test_generator(eclipse_dir, zarr_parent_dir, proj_name, SMRY_WELL_PROPERTIES)))

    if tasks:
        results = executor.execute_stream(task_source=tasks, total_est=len(tasks), desc="Interface Tests")
        for res in results:
            if '⚠️' in res.get('msg', '') or '❌' in res.get('msg', ''):
                print(f"  {res['msg']}")
            else:
                print(f"  {res['msg']}")

    print(f"\n⏱️ 全局测试总耗时: {time.time() - start_time:.2f} 秒")
    print("="*80)