# ===== AI-Friendly Python Version of Notebook =====
# Source Notebook: /Users/chloezh/Projects/jupyter_to_py_project/input_jupyter/数据清洗.ipynb

# ----- Cell 1 (code) -----
# ------------- 标准库 -------------
import os
import glob
import re
import json
from datetime import datetime
# ------------- 数据分析核心库 -------------
import pandas as pd
import numpy as np
# ------------- 科学计算/统计分析库 -------------
from scipy.optimize import minimize
from scipy.stats import ttest_1samp, binomtest, ttest_ind
import statsmodels.api as sm
# ------------- 可视化库 -------------
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import PercentFormatter
import seaborn as sns

# ----- Cell 2 (code) -----
# 设置程序中文字体
def _set_mpl_chinese_font(force_rebuild_cache=False):
    
    # 清理 matplotlib 字体缓存
    if force_rebuild_cache:
        cache_dir = matplotlib.get_cachedir()
        for fp in glob.glob(os.path.join(cache_dir, "fontlist-v*.json")):
            try:
                os.remove(fp)
            except Exception:
                pass
                
    # 先从已注册字体里找
    preferred_names = ["Microsoft YaHei", "SimSun"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred_names:
        if name in available:
            chosen_font_name = name
            break
    
    
    # 如果已注册字体中找不到，从Windows本地路径找
    if not chosen_font_name:
        win_font_candidates = [
            r"C:\Users\Galax\AppData\Local\Microsoft\Windows\Fonts\方正小标宋简.TTF"
            r"C:\Users\Galax\AppData\Local\Microsoft\Windows\Fonts\大标宋简.ttf",  
        ]
    
        chosen_font_name = ""
    
        # 3) 如果文件存在，就 addfont，让 matplotlib 一定“认识它”
        for fpath in win_font_candidates:
            if os.path.exists(fpath):
                try:
                    fm.fontManager.addfont(fpath)
                    # 取字体名（关键：rcParams 用字体名，不是文件名）
                    prop = fm.FontProperties(fname=fpath)
                    chosen_font_name = prop.get_name()
                    break
                except Exception:
                    continue

    # 5) 强制设置 matplotlib 使用中文字体
    if chosen_font_name:
        plt.rcParams["font.sans-serif"] = [chosen_font_name]
        plt.rcParams["font.family"] = "sans-serif"
    else:
        print("⚠️ 未找到可用中文字体。请安装：微软雅黑/黑体/思源黑体(Noto Sans CJK)。")

    # 6) 负号正常显示
    plt.rcParams["axes.unicode_minus"] = False

    return chosen_font_name

font_used = _set_mpl_chinese_font(force_rebuild_cache=True)
print(font_used)

# ----- Cell 3 (markdown) -----
# # 模型输入
# 输入数据应为日频，模型采用对数收益率；月末调仓，从月末的后一个交易日开始计算收益率；
# 
# 考虑到各资产存在可交易日期不同的情况，部分资产缺少收益率的日期中，将收益率中的NA填为0。

# ----- Cell 4 (code) -----
# 资产数据文件
macro_docu = "macro_docu_0209.xlsx"

# ----- Cell 5 (code) -----
macro_df = pd.read_excel(macro_docu)

#确保日期格式为datetime
if not pd.api.types.is_datetime64_any_dtype(macro_df["release_date"]):
    macro_df["release_date"] = pd.to_datetime(macro_df["release_date"], errors="raise")

# 打印可选的 TARGET_COL（附带发布频率）+ EXPLAINED_ASSET

# 直接在print内生成并展示宏观变量的DataFrame，不创建新对象
print("可选的宏观变量为：")
print(macro_df[["factor_name", "frequency"]].drop_duplicates().sort_values(["factor_name", "frequency"]))

# ----- Cell 6 (markdown) -----
# ## 两指标相加

# ----- Cell 7 (code) -----
# 要合成的两个因子名
corp_name = "AccInc_CorpMLT"
hh_name   = "AccInc_HouseholdMLT"
new_name  = "AccInc_MLT_YoY"

# 用于对齐与合并的键（除 factor_name、value 外都参与对齐）
key_cols = [
    "factor_group", "country", "frequency",
    "value_period_start", "value_period_end",
    "release_date", "release_time", "unit"
]

# 0) 只取两条要合成的序列
sub = macro_df.loc[macro_df["factor_name"].isin([corp_name, hh_name])].copy()

# 1) 临时 pivot 宽表：一行一个“期间+元信息”，两列分别是 corp/hh 的 value
wide = (
    sub.pivot_table(
        index=key_cols,
        columns="factor_name",
        values="value",
        aggfunc="first"  )
    .reset_index()
)

# 2) 生成合并后的“绝对值”序列：value = corp + hh
#    如果某个月其中一项缺失：这里用 min_count=1 让两者都缺失时为 NaN；否则按可用项相加
wide["value_sum"] = wide[[corp_name, hh_name]].sum(axis=1, min_count=1)

# 3) 把“绝对值”序列变回 long，并强制“除 value 外信息与 corp 相同”
#    你要求：除 value 外所有信息都和 corp 相同 —— 最严格做法：用 corp 的元信息作为模板
corp_meta = (
    macro_df.loc[macro_df["factor_name"].eq(corp_name), key_cols]
    .drop_duplicates()
)

# 将 wide 里的 key_cols 与 corp_meta 做内连接，确保只保留 corp 存在的那些时期/元信息
# （这样 release_date/time 等也会与 corp 完全一致）
sum_df = (
    wide[key_cols + ["value_sum"]]
    .merge(corp_meta, on=key_cols, how="inner")
)

sum_df = sum_df.rename(columns={"value_sum": "value"})
sum_df["factor_name"] = new_name

# 4) 计算同比（按 value_period_start 排序，12 期同比）
#    - 这里默认 Monthly 且每月一条；如中间缺月，会导致同比缺失（这通常是对的）
sum_df = sum_df.sort_values(["country", "factor_group", "value_period_start"])

# ✅ 新增：同比列（按你的需求“根据 value_period_start 做同比”）
sum_df["value_yoy"] = (
    sum_df.groupby(["country", "factor_group", "frequency"])["value"]
          .apply(lambda s: s / s.shift(12) - 1.0)
          .reset_index(level=[0,1,2], drop=True)
)

# 如果你希望 YoY 仍放在 value 里（而不是新列），则把 value 替换为 yoy
sum_yoy_df = sum_df.copy()
sum_yoy_df["value"] = sum_yoy_df["value_yoy"]
sum_yoy_df = sum_yoy_df.drop(columns=["value_yoy"])

# 6) 拼回原 macro_df
macro_df = pd.concat([macro_df, sum_yoy_df], ignore_index=True)

# ----- Cell 8 (code) -----
# 定义要排除的因子名列表
exclude_factors = [
    "AccInc_CorpMLT",
    "AccInc_HouseholdMLT",
    "ShehuiRongziGuimoCunliang_value"
]

# 使用 isin() + 取反 (~)
macro_df = macro_df[~macro_df["factor_name"].isin(exclude_factors)]
print(macro_df[["factor_name", "frequency"]].drop_duplicates().sort_values(["factor_name", "frequency"]))

# ----- Cell 9 (markdown) -----
# # 平滑

# ----- Cell 10 (markdown) -----
# ## MA平滑

# ----- Cell 11 (code) -----
factor_pairs = [
    ("ShehuiRongziGuimoCunliang_YoY", "ShehuiRongziGuimoCunliang_YoY_MA6"),
    ("CPI_YoY", "CPI_YoY_MA6"),
    ("PMI", "PMI_MA6"),
]

window = 6
min_periods = 6  # 前面至少有6个月数值，不满不纳入计算

# 收集所有 new_df，最后一次性 concat（避免在循环里反复改 macro_df）
new_dfs = []

for src_factor, new_factor in factor_pairs:

    # =========================
    # 0) 取出目标因子数据（long format）
    # =========================
    sub = macro_df.loc[macro_df["factor_name"].eq(src_factor)].copy()

    if sub.empty:
        print(f"[WARN] src_factor not found or empty: {src_factor}")
        continue

    # （可选但推荐）确保日期列可排序
    sub["value_period_start"] = pd.to_datetime(sub["value_period_start"])
    sub["value_period_end"]   = pd.to_datetime(sub["value_period_end"])
    sub["release_date"]       = pd.to_datetime(sub["release_date"])

    # =========================
    # 1) 排序 + 计算 MA(6)（one-sided）
    # =========================
    sub = sub.sort_values(
        ["country", "factor_name", "frequency",
         "value_period_start", "value_period_end", "release_date"]
    )

    sub["value_ma6"] = (
        sub.groupby(["country", "factor_name", "frequency"], group_keys=False)["value"]
           .apply(lambda s: s.rolling(window=window, min_periods=min_periods).mean())
    )

    # =========================
    # 2) 生成新因子（除 value 外信息保持一致）
    # =========================
    new_df = sub.copy()
    new_df["factor_name"] = new_factor
    new_df["value"] = new_df["value_ma6"]
    new_df = new_df.drop(columns=["value_ma6"])

    # 你当前选择：不保留前 5 期 NaN
    new_df = new_df.dropna(subset=["value"])

    new_dfs.append(new_df)

    # =========================
    # 3) quick check（每组各自看一眼）
    # =========================
    print(f"\n[CHECK] {src_factor} -> {new_factor}")
    print(
        new_df[["factor_name", "country", "frequency", "value_period_start", "value"]]
        .sort_values(["country", "value_period_start"])
        .head(15)
    )

# =========================
# 4) 最后一次性拼回原 macro_df
# =========================
if len(new_dfs) > 0:
    macro_df = pd.concat([macro_df] + new_dfs, ignore_index=True)
else:
    print("[INFO] No new factors generated.")


# ----- Cell 12 (markdown) -----
# ## 季调

# ----- Cell 13 (code) -----
# 输出
macro_df.to_csv("macro_df_v1.csv", index=False, encoding="utf-8-sig")

# ----- Cell 14 (code) -----
