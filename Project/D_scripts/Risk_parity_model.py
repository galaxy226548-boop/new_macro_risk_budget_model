# ===== AI-Friendly Python Version of Notebook =====
# Source Notebook: /Users/chloezh/Projects/jupyter_to_py_project/input_jupyter/风险平价模型_含基准模型 - buy and hold - 含交易费用.ipynb

# ----- Cell 1 (code) -----
import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use("Agg")
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import json
from datetime import datetime

# ----- Cell 2 (markdown) -----
# # 模型输入
# 输入数据应为日频，模型采用对数收益率；月末调仓，从月末的后一个交易日开始计算收益率；
# 考虑到各资产存在可交易日期不同的情况，部分资产缺少收益率的日期中，将收益率中的NA填为0。

# ----- Cell 3 (code) -----
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_NAME = "risk_parity_model1_unlevered"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "C_output", "Risk_parity")
FIGURE_DIR = os.path.join(OUTPUT_DIR, f"{RUN_NAME}_{RUN_TS}_figures")

# 资产数据文件（路径相对于本脚本所在目录向上一级，再进入 A_data/）
target_docu = os.path.join(PROJECT_DIR, "A_data", "ALL_ASSETS_with_W008.xlsx")

# 使用到的数据的时间范围
start_date = "2010-03-05"
end_date = "2026-01-31"
start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date)
if pd.isna(start_ts) or pd.isna(end_ts):
    raise ValueError(f"日期参数无法解析：start_date={start_date}, end_date={end_date}")
if start_ts > end_ts:
    raise ValueError(f"start_date 不能晚于 end_date：{start_date} > {end_date}")

# 风险平价模型参数设置
LOOKBACK_MONTHS = 6          # 用过去6个月日收益估计协方差
ANNUAL_DAYS = 252
RF_ANNUAL = 0.015              # 年化无风险利率（先设0；你之后可换成曲线/序列）
LONG_ONLY = True
STRATEGY_START = "2022-12-31" #策略构建与回测时间起点，此处和国泰君安研报保持一致，以方便策略结果的比较。原则上其实06年11月开始有数据，07年5月就可以回测了
strategy_start_ts = pd.to_datetime(STRATEGY_START)
if pd.isna(strategy_start_ts):
    raise ValueError(f"STRATEGY_START 无法解析：{STRATEGY_START}")

# 手续费设置
FEE_RATE = 1e-4  # 单边手续费率：买入/卖出成交额的万分之一（=1bp）

# 资产选择设置：
# - "interactive"：终端运行时弹出选择；无交互环境自动全选
# - "all"：直接全选
# - ["HS300", "CBA02001", "NHCI"]：指定资产列表，写错时报错
SELECTED_ASSETS = "interactive"

def _save_current_figure(filename: str):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("✅ Figure saved:", path)
    return path

# ----- Cell 4 (markdown) -----
# # 数据准备（日频对数收益率，缺失值以0填充）

# ----- Cell 5 (code) -----
df = pd.read_excel(target_docu)
df["Trddt"] = pd.to_datetime(df["Trddt"])
df["Clsidx"] = pd.to_numeric(df["Clsidx"], errors="coerce")
df = df.dropna(subset=["Indexcd", "Trddt", "Clsidx"])

# 看数据的时间范围
df["Trddt"].min(), df["Trddt"].max()

# ----- Cell 6 (code) -----
# 限制数据范围
df = df[(df["Trddt"] >= start_date) & (df["Trddt"] <= end_date)]
if df.empty:
    raise ValueError(f"日期过滤后没有数据：start_date={start_date}, end_date={end_date}")

# 排序
df = df.sort_values(["Indexcd", "Trddt"]).reset_index(drop=True)

# 计算日对数收益率
df["log_ret"] = df.groupby("Indexcd")["Clsidx"].transform(
    lambda x: np.log(x) - np.log(x.shift(1))
)

df = df.dropna(subset=["log_ret"]) #这里还没有用日期为列，所以应该只去掉了各资产数据的第一天，无法往前减数字所以没有log return的情况
df.info()

# ----- Cell 7 (code) -----
ret_sub = (df
              .pivot(index="Trddt", columns="Indexcd", values="log_ret")
              .sort_index()
             )

ret_sub.head()

# ----- 资产选择交互（命令行运行时弹出选择提示）-----
def _select_assets_interactive(available: list) -> list:
    print("\n" + "=" * 52)
    print("数据中共有以下资产，请选择要纳入计算的资产：")
    for i, a in enumerate(available, 1):
        tag = "  ← W008·外层免手续费" if "W008" in str(a).upper() else ""
        print(f"  [{i:2d}] {a}{tag}")
    print("=" * 52)
    print("输入方式（多选用空格或逗号分隔）：")
    print("  · 编号：如  1 3 5  或  1,3,5")
    print("  · 资产名：如  HS300 CBA02001")
    print("  · 直接回车 / 输入 all → 选择全部")
    try:
        user_input = input("请输入选择 >>> ").strip()
    except EOFError:
        print("⚠️  当前运行环境不可交互，已自动选择全部资产")
        return list(available)

    if not user_input or user_input.lower() == "all":
        print(f"✅ 已选择全部 {len(available)} 个资产")
        return list(available)

    selected, seen = [], set()
    for part in user_input.replace(",", " ").split():
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(available):
                a = available[idx]
                if a not in seen:
                    selected.append(a)
                    seen.add(a)
            else:
                print(f"⚠️  编号 {part} 超出范围（1~{len(available)}），已忽略")
        else:
            if part in available:
                if part not in seen:
                    selected.append(part)
                    seen.add(part)
            else:
                print(f"⚠️  资产名 '{part}' 不存在，已忽略")

    if not selected:
        print("⚠️  未有效选择任何资产，已自动选择全部")
        return list(available)

    print(f"✅ 已选择 {len(selected)} 个资产：{selected}")
    return selected

def _select_assets(available: list, selection) -> list:
    if selection == "interactive":
        if sys.stdin is not None and sys.stdin.isatty():
            return _select_assets_interactive(available)
        print(f"ℹ️  当前运行环境不可交互，已自动选择全部 {len(available)} 个资产")
        return list(available)
    if selection is None or selection == "all":
        return list(available)
    if isinstance(selection, str):
        selection = selection.replace(",", " ").split()
    selected = list(selection)
    missing = [a for a in selected if a not in available]
    if missing:
        raise ValueError(f"SELECTED_ASSETS 中存在数据里没有的资产：{missing}；可用资产={available}")
    if not selected:
        raise ValueError("SELECTED_ASSETS 不能为空")
    return selected

_selected = _select_assets(ret_sub.columns.tolist(), SELECTED_ASSETS)
ret_sub_raw = ret_sub[_selected].copy()
ret_sub = ret_sub_raw.fillna(0.0)

# ----- Cell 8 (code) -----
# 计算各资产之间的协方差和相关系数，由于数据为日频收益率，协方差数值会偏小
# 回测使用 fillna(0.0) 后的收益矩阵；相关系数展示保留原始 pairwise non-NaN 口径，避免缺失日期填 0 扭曲相关性。
cov_matrix = ret_sub.cov()
corr_matrix = ret_sub_raw.corr()

# --- 打印查看 ---
print("=== 协方差矩阵 (Covariance) ===")
print(cov_matrix.head())
print("\n=== 相关系数矩阵 (Correlation) ===")
print(corr_matrix.head())

# ----- Cell 9 (code) -----
ret_sub.columns.tolist()

# ----- Cell 10 (markdown) -----
# # 风险平价模型回测参数设置

# ----- Cell 11 (code) -----
assets = ret_sub.columns.tolist()
n = len(assets)
if n == 0:
    raise ValueError("没有可用于回测的资产")
if strategy_start_ts > ret_sub.index.max():
    raise ValueError(f"STRATEGY_START 晚于数据末日：{STRATEGY_START} > {ret_sub.index.max().date()}")

# ===== 调仓日：每个月最后一个交易日 =====
rebalance_dates = ret_sub.groupby(pd.Grouper(freq="ME")).apply(lambda x: x.index.max())
rebalance_dates = rebalance_dates[rebalance_dates >= STRATEGY_START]

# 防止调仓日是有数据的最后一天引起报错
rebalance_dates = rebalance_dates[rebalance_dates < ret_sub.index.max()]
if rebalance_dates.empty:
    raise ValueError(
        f"没有有效调仓日，请检查 STRATEGY_START={STRATEGY_START}、end_date={end_date} 和输入数据范围"
    )

rebalance_dates[:]

# ----- Cell 12 (markdown) -----
# # 风险平价求解器

# ----- Cell 13 (code) -----
def risk_parity_weights_from_window(ret_window: pd.DataFrame, w0: np.ndarray | None = None) -> np.ndarray:
    """
    输入：ret_window（窗口内日对数收益，T×N）
    输出：风险平价权重 w（N,），约束 sum(w)=1, 0<=w<=1

    P0-2增强：
    - 自适应 Diagonal Loading：Sigma += I * eps（eps 与协方差尺度挂钩）
    - 健康检查：Sigma/优化结果异常时分层兜底
      1) w_prev（即传入的 w0，经清洗归一）
      2) 等权
    """
    # =========================
    # 0) 维度与基础检查
    # =========================
    n_local = int(ret_window.shape[1])
    if n_local <= 0:
        return np.array([])

    # =========================
    # 1) 初始权重（兜底优先：w_prev）
    # =========================
    def _clean_w(w: np.ndarray | None) -> np.ndarray:
        if w is None:
            w = np.ones(n_local, dtype="float64") / n_local
        else:
            w = np.asarray(w, dtype="float64").copy()
            if w.shape[0] != n_local:
                # 尺寸不匹配：回退等权（更安全，不做隐式截断/补零）
                w = np.ones(n_local, dtype="float64") / n_local
        w = np.clip(w, 0.0, 1.0)  # long-only 下的保险
        s = float(np.nansum(w))
        if (not np.isfinite(s)) or s <= 0:
            w = np.ones(n_local, dtype="float64") / n_local
        else:
            w = w / s
        return w

    w_prev = _clean_w(w0)  # ✅ 第一兜底：上一期权重（或外部给的初值）
    w_equal = np.ones(n_local, dtype="float64") / n_local  # ✅ 第二兜底：等权

    # =========================
    # 2) 协方差矩阵（pairwise non-NaN） + 自适应对角加载
    # =========================
    Sigma = ret_window.cov().values
    Sigma = np.asarray(Sigma, dtype="float64")

    # 基础健康检查：形状/有限性
    if Sigma.shape != (n_local, n_local) or (not np.isfinite(Sigma).all()):
        return w_prev

    # 对角线检查：若出现非正或非有限，先回退（或让加载纠正也可，但P0优先稳）
    diag = np.diag(Sigma)
    if (not np.isfinite(diag).all()) or np.any(diag <= 0):
        return w_prev

    # ✅ 自适应 diagonal loading（尺度与窗口波动水平挂钩）
    # 用平均方差作为尺度：trace/N
    avg_var = float(np.trace(Sigma) / n_local)
    if (not np.isfinite(avg_var)) or avg_var <= 0:
        return w_prev

    # 你可以把 alpha 当作超参数；这里给一个偏保守的默认
    ALPHA = 1e-6
    eps = ALPHA * avg_var
    Sigma = Sigma + np.eye(n_local, dtype="float64") * eps

    # =========================
    # 3) 目标函数：TRC 尽量相等
    # =========================
    def obj_w(w):
        w = np.asarray(w, dtype="float64")
        # 数值稳定：避免 0 导致除法不稳
        w = np.clip(w, 1e-12, 1.0)

        R2 = float(w.T @ Sigma @ w)
        if (not np.isfinite(R2)) or R2 <= 0:
            return 1e12
        R = np.sqrt(R2)

        Sw = Sigma @ w
        if not np.isfinite(Sw).all():
            return 1e12

        MRC = Sw / R
        TRC = w * MRC
        trc_mean = float(np.mean(TRC))

        # 用 n_local 而不是全局 n（支持子集资产）
        return 2.0 * n_local * float(np.sum((TRC - trc_mean) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n_local)] if LONG_ONLY else [(-1.0, 1.0) for _ in range(n_local)]

    # =========================
    # 4) 求解 + 分层兜底
    # =========================
    try:
        res = minimize(
            obj_w,
            x0=w_prev,  # ✅ 用上一期权重作为初值（更稳）
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-12, "maxiter": 2000, "disp": False}
        )
    except Exception:
        return w_prev

    # 优化器失败 / 非收敛 / 返回 NaN → 兜底到 w_prev
    if (res is None) or (not getattr(res, "success", False)) or (not np.isfinite(res.x).all()):
        return w_prev

    w = np.asarray(res.x, dtype="float64")
    w = np.clip(w, 0.0, 1.0)

    s = float(w.sum())
    if (not np.isfinite(s)) or s <= 0:
        return w_prev

    w = w / s

    # 最终再做一次健康检查（防止极端数值）
    if not np.isfinite(w).all():
        return w_prev

    return w

# ----- Cell 14 (markdown) -----
# # 回测主循环

# ----- Cell 15 (code) -----
# W008 类因子资产：其内部净值已扣过内部成本，外层回测不再重复计费
FEE_EXEMPT_ASSETS = {a for a in assets if "W008" in str(a).upper()}
fee_mask = np.array([a not in FEE_EXEMPT_ASSETS for a in assets], dtype=bool)
if FEE_EXEMPT_ASSETS:
    print(f"ℹ️  以下资产免收外层手续费：{FEE_EXEMPT_ASSETS}")

weights_by_reb = []
port_log_ret = pd.Series(index=ret_sub.index, dtype="float64")
idx_dates = ret_sub.index

# ✅ 用于风险平价优化器的初始权重：首期等权，之后沿用上一期目标权重
w_prev = np.ones(n, dtype="float64") / n

# ✅ 用于"真实成交额/手续费"的：调仓前漂移权重（初始=零向量，从空仓全额建仓）
w_prev_end = np.zeros(n, dtype="float64")

# ✅ 记录每次调仓的真实换手（单边）与总成交额（双边）都可选
turnover_oneway_list = []     # 0.5*sum(|Δw|)，更常用于年化换手
turnover_gross_list = []      # sum(|Δw|)，用于手续费：fee=rate*gross
all_daily_eff_w_segs = []     # 每段每日漂移有效权重（T×N）

for i, t_reb in enumerate(rebalance_dates):
    # ① 窗口：过去6个月（自然月）到 t_reb（含）
    t_start = t_reb - pd.DateOffset(months=LOOKBACK_MONTHS)
    ret_window = ret_sub.loc[(ret_sub.index >= t_start) & (ret_sub.index <= t_reb), assets]

    # 防止过去六个月交易日太少（但是前面对起点切片过了，应该没有问题，主要起预防作用）
    if ret_window.shape[0] < 60:
        continue

    # ② 在 t_reb 收盘后计算权重
    w_t = risk_parity_weights_from_window(ret_window, w0=w_prev)

    weights_by_reb.append(pd.Series(w_t, index=assets, name=t_reb))
    w_prev = w_t

    # ③ 计算持有区间：从 t_reb 的下一个交易日开始，到下一个调仓日（含）或数据结束
    idx_dates = ret_sub.index

    # 找 t_reb 在交易日序列中的位置
    pos = idx_dates.get_indexer([t_reb])[0]
    if pos == -1:
        continue

    # 生效日 = 下一个交易日
    if pos + 1 >= len(idx_dates):
        break
    t_effective = idx_dates[pos + 1]

    # 持有截止日：下一个月末调仓日（t_next_reb）当日收盘（因为新权重从其下一日生效）
    if i + 1 < len(rebalance_dates):
        t_next_reb = rebalance_dates.iloc[i + 1]
        t_hold_end = min(t_next_reb, idx_dates.max())
    else:
        t_hold_end = idx_dates.max()

    hold_slice = ret_sub.loc[(idx_dates >= t_effective) & (idx_dates <= t_hold_end), assets]
    if hold_slice.shape[0] == 0:
        continue

    # =========================
    # ✅ 交易成本：基于"调仓前漂移权重 w_prev_end"与新目标权重 w_t 的真实成交额
    # =========================
    w_t = np.asarray(w_t, dtype="float64").reshape(-1)
    w_t = np.clip(w_t, 0.0, None) if LONG_ONLY else w_t
    s = w_t.sum()
    w_t = (w_t / s) if s > 0 else (np.ones(n) / n)

    abs_diff = np.abs(w_t - w_prev_end)
    gross_turnover = float(abs_diff.sum())                            # 买+卖成交额占比（含免费资产）
    oneway_turnover = 0.5 * gross_turnover                            # 单边换手
    fee_t = FEE_RATE * float(abs_diff[fee_mask].sum())                # 只对计费资产收手续费

    turnover_gross_list.append(pd.Series({"gross_turnover": gross_turnover, "fee": fee_t}, name=t_reb))
    turnover_oneway_list.append(pd.Series({"oneway_turnover": oneway_turnover}, name=t_reb))

    # =========================
    # ✅ buy & hold：段内资金曲线推演（权重自然漂移）
    # =========================
    asset_rel = np.exp(hold_slice.cumsum())                            # 段起点=1
    port_rel = asset_rel.values @ w_t                                  # 段内组合相对净值

    # 每日漂移有效权重：各资产份额随净值自然漂移后归一化
    daily_eff_w = asset_rel.values * w_t
    daily_eff_w = daily_eff_w / daily_eff_w.sum(axis=1, keepdims=True)
    daily_eff_w_df_seg = pd.DataFrame(daily_eff_w, index=hold_slice.index, columns=assets)
    all_daily_eff_w_segs.append(daily_eff_w_df_seg)

    port_rel_s = pd.Series(port_rel, index=hold_slice.index)
    seg_log_ret = np.log(port_rel_s / port_rel_s.shift(1))
    seg_log_ret.iloc[0] = np.log(port_rel_s.iloc[0] / 1.0)

    # ✅ 把"调仓手续费"体现在新持仓段的第一天（等价于调仓后资金变为(1-fee)）
    #    注意：fee_t 必须 < 1；极端情况下做个保险
    fee_t = min(max(fee_t, 0.0), 0.99)
    seg_log_ret.iloc[0] += np.log(1.0 - fee_t)

    port_log_ret.loc[hold_slice.index] = seg_log_ret.values

    # =========================
    # ✅ 更新"下一次调仓前漂移权重"：用段末各资产相对净值把权重漂移到 t_hold_end 收盘
    # =========================
    last_rel = asset_rel.iloc[-1].values                               # shape (n,)
    w_prev_end = (w_t * last_rel)
    w_prev_end = w_prev_end / w_prev_end.sum() if w_prev_end.sum() > 0 else (np.ones(n) / n)

# 汇总权重表
weights_df = pd.DataFrame(weights_by_reb)
weights_df.index.name = "RebalanceDate"
if weights_df.empty:
    raise ValueError("主模型没有生成任何调仓权重，请检查回测日期、资产选择和 lookback 窗口")

# 汇总每日漂移有效权重（主模型）
daily_weights_df = pd.concat(all_daily_eff_w_segs).sort_index() if all_daily_eff_w_segs else pd.DataFrame(columns=assets)
daily_weights_df.index.name = "Trddt"

# ✅ 真实换手/手续费（基于漂移权重）
turnover_df = pd.DataFrame(turnover_gross_list)
turnover_df.index.name = "RebalanceDate"

turnover_oneway_df = pd.DataFrame(turnover_oneway_list)
turnover_oneway_df.index.name = "RebalanceDate"

weights_df.head(), weights_df.tail()

# ----- Cell 16 (markdown) -----
# # 净值曲线 + 绩效指标（年化收益/波动/回撤/夏普/卡玛）

# ----- Cell 17 (code) -----
# =========================================================
# 0) 对齐收益序列：按策略起点 + 去掉前期窗口空值
# =========================================================
port_log_ret = port_log_ret.loc[port_log_ret.index >= STRATEGY_START]
port_log_ret = port_log_ret.dropna()

# =========================================================
# 1) 换手率：每次调仓的"单边换手"
#    turnover_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}|
# =========================================================
if "turnover_oneway_df" in globals() and isinstance(turnover_oneway_df, pd.DataFrame) and "oneway_turnover" in turnover_oneway_df.columns:
    turnover = turnover_oneway_df["oneway_turnover"].copy()
else:
    # 兜底：忽略漂移，仅目标权重差（会低估真实成交额）
    turnover = weights_df.diff().abs().sum(axis=1) / 2

# ✅ 确保索引是 datetime（不在赋值前引用 turnover）
turnover.index = pd.to_datetime(turnover.index)

# （可选）如果你希望换手只统计策略期内：
# turnover = turnover.loc[turnover.index >= STRATEGY_START]


# =========================================================
# 2) 统一指标计算函数：既支持年度，也支持全样本
# =========================================================
def _calc_one_period_metrics(series: pd.Series,
                             turnover_series: pd.Series,
                             label: str = "",
                             annual_days: int = ANNUAL_DAYS,
                             rf_annual: float = RF_ANNUAL):
    """
    series: 对数收益率（日频）Series
    turnover_series: 调仓日换手率（单边）Series，Index=调仓日
    label: 行标签（年份 or All History）
    """
    if series is None or len(series) < 5:
        return None

    start_date = series.index.min()
    end_date = series.index.max()

    # --- NAV（用于回撤）---
    nav_local = np.exp(series.cumsum())
    nav_local = nav_local / nav_local.iloc[0]

    # --- 年化收益/波动 ---
    mean_daily = series.mean()
    ann_return = np.exp(mean_daily * annual_days) - 1
    ann_vol = series.std(ddof=1) * np.sqrt(annual_days)

    # --- 最大回撤 ---
    running_max = nav_local.cummax()
    drawdown = nav_local / running_max - 1.0
    max_drawdown = drawdown.min()

    # --- 夏普/卡玛 ---
    sharpe = (ann_return - rf_annual) / ann_vol if ann_vol > 0 else np.nan
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    # --- 年化换手率（用"时间跨度/年"来年化）---
    # 取该期间内的调仓记录（换手发生在调仓点）
    period_turnover = turnover_series.loc[start_date:end_date]
    total_turnover = float(period_turnover.sum())

    trading_days = len(series)
    years_actual = trading_days / annual_days
    ann_turnover = total_turnover / years_actual if years_actual > 0 else total_turnover

    return {
        "Year": label,
        "StartDate": start_date,
        "EndDate": end_date,
        "Obs": len(series),
        "AnnReturn": ann_return,
        "AnnVolatility": ann_vol,
        "MaxDrawdown": max_drawdown,
        "AnnTurnover": ann_turnover,
        "Sharpe": sharpe,
        "Calmar": calmar,
    }

# =========================================================
# 3) 按年滚动 + 全样本（All History）
# =========================================================
metrics_list = []

# 年度
for year, grp in port_log_ret.groupby(port_log_ret.index.year):
    m = _calc_one_period_metrics(grp, turnover, label=str(year))
    if m is not None:
        metrics_list.append(m)

# 全样本
m_all = _calc_one_period_metrics(port_log_ret, turnover, label="All History")
if m_all is not None:
    metrics_list.append(m_all)

if not metrics_list:
    raise ValueError("主模型没有足够的收益数据用于计算绩效指标")
metrics_raw = pd.DataFrame(metrics_list).set_index("Year")

# =========================================================
# 4) 输出两份：原始数值版（metrics_raw）+ 格式化展示版（metrics）
# =========================================================
metrics = metrics_raw.copy()

# 百分比列
pct_cols = ["AnnReturn", "AnnVolatility", "MaxDrawdown", "AnnTurnover"]
for col in pct_cols:
    if col in metrics.columns:
        metrics[col] = metrics[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else np.nan)

# 数值列
num_cols = ["Sharpe", "Calmar"]
for col in num_cols:
    if col in metrics.columns:
        metrics[col] = metrics[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else np.nan)

# =========================================================
# 5) 为后续绘图代码准备：全样本 nav（基于已清洗的 port_log_ret）
# =========================================================
rp_nav = np.exp(port_log_ret.cumsum())
rp_nav = rp_nav / rp_nav.iloc[0]

# ----- Cell 18 (code) -----
# 输出结果
metrics

# ----- Cell 19 (code) -----
plt.figure(figsize=(10, 4))
plt.plot(rp_nav.index, rp_nav.values)
plt.title("Risk Parity Strategy Cumulative NAV")
plt.xlabel("Date")
plt.ylabel("NAV")
plt.grid(True)
_save_current_figure(f"{RUN_NAME}_{RUN_TS}_nav_risk_parity.png")

# ----- 主风险平价模型：权重与风险贡献图 -----
def _risk_contribution_from_window(ret_window: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    用与风险平价优化器一致的协方差口径计算总风险贡献占比。
    返回每个资产的 TRC / sum(TRC)，异常时返回 NaN。
    """
    asset_list = list(weights.index)
    n_local = len(asset_list)
    out = pd.Series(np.nan, index=asset_list, dtype="float64")
    if n_local == 0 or ret_window.shape[0] < 2:
        return out

    w = weights.astype("float64").reindex(asset_list).values
    if (not np.isfinite(w).all()) or np.nansum(np.abs(w)) <= 0:
        return out

    Sigma = ret_window.loc[:, asset_list].cov().values
    Sigma = np.asarray(Sigma, dtype="float64")
    if Sigma.shape != (n_local, n_local) or (not np.isfinite(Sigma).all()):
        return out

    diag = np.diag(Sigma)
    if (not np.isfinite(diag).all()) or np.any(diag <= 0):
        return out

    avg_var = float(np.trace(Sigma) / n_local)
    if (not np.isfinite(avg_var)) or avg_var <= 0:
        return out

    Sigma = Sigma + np.eye(n_local, dtype="float64") * (1e-6 * avg_var)
    port_var = float(w.T @ Sigma @ w)
    if (not np.isfinite(port_var)) or port_var <= 0:
        return out

    port_risk = np.sqrt(port_var)
    marginal_rc = (Sigma @ w) / port_risk
    total_rc = w * marginal_rc
    rc_sum = float(np.nansum(total_rc))
    if (not np.isfinite(rc_sum)) or abs(rc_sum) <= 1e-12:
        return out

    return pd.Series(total_rc / rc_sum, index=asset_list, dtype="float64")


def _calc_risk_contribution_table(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    asset_list: list,
    lookback_months: int,
) -> pd.DataFrame:
    rows = []
    for t_reb, w_row in weights.loc[:, asset_list].iterrows():
        t_start = pd.Timestamp(t_reb) - pd.DateOffset(months=lookback_months)
        ret_window = returns.loc[(returns.index >= t_start) & (returns.index <= t_reb), asset_list]
        rc = _risk_contribution_from_window(ret_window, w_row)
        rc.name = t_reb
        rows.append(rc)

    risk_contribution = pd.DataFrame(rows, columns=asset_list)
    risk_contribution.index.name = "RebalanceDate"
    return risk_contribution


def _annual_100pct_table(df_in: pd.DataFrame) -> pd.DataFrame:
    annual = df_in.copy()
    annual.index = pd.to_datetime(annual.index)
    annual = annual.groupby(annual.index.year).mean()
    annual = annual.reindex(columns=assets)
    row_sum = annual.sum(axis=1).replace(0, np.nan)
    return annual.div(row_sum, axis=0)


risk_contribution_df = _calc_risk_contribution_table(weights_df, ret_sub, assets, LOOKBACK_MONTHS)

plt.figure(figsize=(12, 5))
for asset in assets:
    plt.plot(daily_weights_df.index, daily_weights_df[asset], label=asset, linewidth=1.4)
plt.title("Risk Parity Daily Asset Allocation Weights")
plt.xlabel("Date")
plt.ylabel("Weight")
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.grid(True, alpha=0.3)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
_save_current_figure(f"{RUN_NAME}_{RUN_TS}_weights_line_risk_parity.png")

annual_weights_100pct = _annual_100pct_table(daily_weights_df)
ax = annual_weights_100pct.plot(kind="bar", stacked=True, figsize=(10, 5), width=0.78)
ax.set_title("Risk Parity Annual Average Asset Allocation Weights")
ax.set_xlabel("Year")
ax.set_ylabel("Weight")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.set_ylim(0, 1)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
_save_current_figure(f"{RUN_NAME}_{RUN_TS}_weights_annual_stacked_risk_parity.png")

annual_risk_contribution_100pct = _annual_100pct_table(risk_contribution_df)
ax = annual_risk_contribution_100pct.plot(kind="bar", stacked=True, figsize=(10, 5), width=0.78)
ax.set_title("Risk Parity Annual Average Asset Risk Contribution")
ax.set_xlabel("Year")
ax.set_ylabel("Risk Contribution")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.set_ylim(0, 1)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
_save_current_figure(f"{RUN_NAME}_{RUN_TS}_risk_contribution_annual_stacked_risk_parity.png")

# ----- Cell 20 (code) -----
# =========================================================
# ✅ Benchmarks: Inverse-Vol / Equal-Weight / Equity:CBA:NHCI(6:3:1)
#    - 资产集合、回测区间、调仓规则、执行逻辑与 Risk Parity 保持一致
#    - 指标统计区间：STRATEGY_START ~ end_date（不是 start_date ~ end_date）
# =========================================================

def _run_backtest_with_weight_func(weight_func, label: str):
    weights_by_reb_local = []
    port_log_ret_local = pd.Series(index=ret_sub.index, dtype="float64")
    idx_dates = ret_sub.index

    # ✅ 真实交易成本：调仓前漂移权重（初始=零向量，从空仓全额建仓）
    w_prev_end = np.zeros(n, dtype="float64")
    turnover_gross_list_local = []
    turnover_oneway_list_local = []
    all_daily_eff_w_segs_local = []

    for i, t_reb in enumerate(rebalance_dates):
        # ① lookback window（与 Risk Parity 一致）
        t_start = t_reb - pd.DateOffset(months=LOOKBACK_MONTHS)
        ret_window = ret_sub.loc[(idx_dates >= t_start) & (idx_dates <= t_reb), assets]

        if ret_window.shape[0] < 60:
            continue

        # ② 在 t_reb 收盘后计算权重
        w_t = weight_func(ret_window)

        # 保险：非负 + 归一化
        w_t = np.asarray(w_t, dtype="float64").reshape(-1)
        if LONG_ONLY:
            w_t = np.clip(w_t, 0.0, None)
        s = w_t.sum()
        w_t = (w_t / s) if s > 0 else np.ones(n) / n

        weights_by_reb_local.append(pd.Series(w_t, index=assets, name=t_reb))

        # ③ 生效日 = 下一个交易日；持有到下一次调仓日（含）
        pos = idx_dates.get_indexer([t_reb])[0]
        if pos == -1:
            continue
        if pos + 1 >= len(idx_dates):
            break
        t_effective = idx_dates[pos + 1]

        if i + 1 < len(rebalance_dates):
            t_next_reb = rebalance_dates.iloc[i + 1]
            t_hold_end = min(t_next_reb, idx_dates.max())
        else:
            t_hold_end = idx_dates.max()

        hold_slice = ret_sub.loc[(idx_dates >= t_effective) & (idx_dates <= t_hold_end), assets]

        if hold_slice.shape[0] == 0:
            continue

        # ✅ 交易成本：用漂移权重 vs 新权重（W008 类资产免收外层手续费）
        abs_diff = np.abs(w_t - w_prev_end)
        gross_turnover = float(abs_diff.sum())
        oneway_turnover = 0.5 * gross_turnover
        fee_t = FEE_RATE * float(abs_diff[fee_mask].sum())

        turnover_gross_list_local.append(pd.Series({"gross_turnover": gross_turnover, "fee": fee_t}, name=t_reb))
        turnover_oneway_list_local.append(pd.Series({"oneway_turnover": oneway_turnover}, name=t_reb))

        # ✅ buy & hold 段内推演
        asset_rel = np.exp(hold_slice.cumsum())
        port_rel = asset_rel.values @ w_t

        # 每日漂移有效权重：各资产份额随净值自然漂移后归一化
        daily_eff_w_local = asset_rel.values * w_t
        daily_eff_w_local = daily_eff_w_local / daily_eff_w_local.sum(axis=1, keepdims=True)
        daily_eff_w_df_seg_local = pd.DataFrame(daily_eff_w_local, index=hold_slice.index, columns=assets)
        all_daily_eff_w_segs_local.append(daily_eff_w_df_seg_local)

        port_rel_s = pd.Series(port_rel, index=hold_slice.index)

        seg_log_ret = np.log(port_rel_s / port_rel_s.shift(1))
        seg_log_ret.iloc[0] = np.log(port_rel_s.iloc[0] / 1.0)

        fee_t = min(max(fee_t, 0.0), 0.99)
        seg_log_ret.iloc[0] += np.log(1.0 - fee_t)

        # ✅ 修复：必须写回 port_log_ret_local（不是 port_log_ret）
        port_log_ret_local.loc[hold_slice.index] = seg_log_ret.values

        # ✅ 更新漂移权重到段末
        last_rel = asset_rel.iloc[-1].values
        w_prev_end = (w_t * last_rel)
        w_prev_end = w_prev_end / w_prev_end.sum() if w_prev_end.sum() > 0 else (np.ones(n) / n)

    weights_df_local = pd.DataFrame(weights_by_reb_local)
    weights_df_local.index.name = "RebalanceDate"

    turnover_df_local = pd.DataFrame(turnover_gross_list_local)
    turnover_df_local.index.name = "RebalanceDate"

    turnover_oneway_df_local = pd.DataFrame(turnover_oneway_list_local)
    turnover_oneway_df_local.index.name = "RebalanceDate"

    daily_weights_df_local = pd.concat(all_daily_eff_w_segs_local).sort_index() if all_daily_eff_w_segs_local else pd.DataFrame(columns=assets)
    daily_weights_df_local.index.name = "Trddt"

    return weights_df_local, port_log_ret_local, turnover_df_local, turnover_oneway_df_local, daily_weights_df_local

# -------------------------
# 1) Inverse-Volatility（波动率倒数）
#    w_i ∝ 1 / σ_i（σ 用过去 LOOKBACK_MONTHS 的日收益标准差估计）
# -------------------------
def _w_inv_vol(ret_window: pd.DataFrame, eps: float = 1e-12):
    vol = ret_window.std(ddof=1).values  # 日频波动（对数收益）
    inv = 1.0 / np.maximum(vol, eps)
    return inv / inv.sum()

inv_weights, inv_port_log_ret, inv_turnover_df, inv_turnover_oneway_df, inv_daily_weights_df = _run_backtest_with_weight_func(_w_inv_vol, "InvVol")

# -------------------------
# 2) Equal-Weight（均分）
# -------------------------
def _w_equal(ret_window: pd.DataFrame):
    return np.ones(n) / n

eq_weights,  eq_port_log_ret,  eq_turnover_df,  eq_turnover_oneway_df,  eq_daily_weights_df  = _run_backtest_with_weight_func(_w_equal, "Equal")

# -------------------------
# 3) Equity:CBA:NHCI = 6:3:1（股债商 631）
#    股：HS300 / W008_signal_binary 动态选择
#        - 两者都有：优先 HS300
#        - 只有 W008_signal_binary：使用 W008_signal_binary
#        - 只有 HS300：使用 HS300
#    债：CBA02001 / CBA00201.CS 动态选择
#        - 两者都有：优先 CBA02001
#        - 只有其中一个：使用那个资产
#    商：NHCI
#    其余资产权重 = 0
# -------------------------
def _select_631_asset(asset_list: list, candidates: list, label: str) -> str:
    for candidate in candidates:
        if candidate in asset_list:
            return candidate
    raise ValueError(
        f"631模型的{label}资产需要包含 {candidates} 中的至少一个，当前 assets={asset_list}"
    )

HB631_EQUITY_ASSET = _select_631_asset(assets, ["HS300", "W008_signal_binary"], "股")
HB631_BOND_ASSET = _select_631_asset(assets, ["CBA02001", "CBA00201.CS"], "债")
print(f"ℹ️  631模型股资产使用：{HB631_EQUITY_ASSET}")
print(f"ℹ️  631模型债资产使用：{HB631_BOND_ASSET}")

def _w_631(ret_window: pd.DataFrame):
    w = np.zeros(n, dtype="float64")
    required_assets = [HB631_EQUITY_ASSET, HB631_BOND_ASSET, "NHCI"]
    missing_assets = [a for a in required_assets if a not in assets]
    if missing_assets:
        raise ValueError(f"631模型需要 assets 里包含 {required_assets}，缺失={missing_assets}，当前 assets={assets}")
    w[assets.index(HB631_EQUITY_ASSET)] = 0.6
    w[assets.index(HB631_BOND_ASSET)] = 0.3
    w[assets.index("NHCI")] = 0.1
    return w

hb_weights,  hb_port_log_ret,  hb_turnover_df,  hb_turnover_oneway_df,  hb_daily_weights_df  = _run_backtest_with_weight_func(_w_631, "HB631")

# =========================================================
# ✅ 对齐收益序列（与 Risk Parity 指标区间一致：STRATEGY_START ~ end_date）
# =========================================================
def _align_series_and_turnover(port_log_ret_local: pd.Series, weights_df_local: pd.DataFrame):
    s = port_log_ret_local.loc[port_log_ret_local.index >= STRATEGY_START].dropna()
    t = weights_df_local.diff().abs().sum(axis=1) / 2.0
    t.index = pd.to_datetime(t.index)
    return s, t

s_invvol, t_invvol = _align_series_and_turnover(inv_port_log_ret, inv_weights)
s_equal,  t_equal  = _align_series_and_turnover(eq_port_log_ret,  eq_weights)
s_631,    t_631    = _align_series_and_turnover(hb_port_log_ret,     hb_weights)

# =========================================================
# ✅ 计算指标（复用上面已定义的 _calc_one_period_metrics）
# =========================================================
def _calc_metrics_table(series: pd.Series, turnover_series: pd.Series):
    metrics_list = []
    for year, grp in series.groupby(series.index.year):
        m = _calc_one_period_metrics(grp, turnover_series, label=str(year))
        if m is not None:
            metrics_list.append(m)
    m_all = _calc_one_period_metrics(series, turnover_series, label="All History")
    if m_all is not None:
        metrics_list.append(m_all)
    if not metrics_list:
        raise ValueError("基准模型没有足够的收益数据用于计算绩效指标")
    return pd.DataFrame(metrics_list).set_index("Year")

metrics_raw_invvol = _calc_metrics_table(s_invvol, t_invvol)
metrics_raw_equal  = _calc_metrics_table(s_equal,  t_equal)
metrics_raw_631    = _calc_metrics_table(s_631,    t_631)

# ----- Cell 21 (code) -----
pct_cols = ["AnnReturn", "AnnVolatility", "MaxDrawdown", "AnnTurnover"]
for col in pct_cols:
    if col in metrics_raw_invvol.columns:
        metrics_raw_invvol[col] = metrics_raw_invvol[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else np.nan)

# 数值列
num_cols = ["Sharpe", "Calmar"]
for col in num_cols:
    if col in metrics_raw_invvol.columns:
        metrics_raw_invvol[col] = metrics_raw_invvol[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else np.nan)

metrics_raw_invvol

# ----- Cell 22 (code) -----
invvol_nav = np.exp(s_invvol.cumsum())
invvol_nav = invvol_nav / invvol_nav.iloc[0]
plt.figure(figsize=(10, 4))
plt.plot(invvol_nav.index, invvol_nav.values)
plt.title("Inverse Volatility Strategy Cumulative NAV")
plt.xlabel("Date")
plt.ylabel("NAV")
plt.grid(True)
_save_current_figure(f"{RUN_NAME}_{RUN_TS}_nav_inverse_volatility.png")

# ----- Cell 23 (code) -----
pct_cols = ["AnnReturn", "AnnVolatility", "MaxDrawdown", "AnnTurnover"]
for col in pct_cols:
    if col in metrics_raw_equal.columns:
        metrics_raw_equal[col] = metrics_raw_equal[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else np.nan)

# 数值列
num_cols = ["Sharpe", "Calmar"]
for col in num_cols:
    if col in metrics_raw_equal.columns:
        metrics_raw_equal[col] = metrics_raw_equal[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else np.nan)

metrics_raw_equal

# ----- Cell 24 (code) -----
equal_nav = np.exp(s_equal.cumsum())
equal_nav = equal_nav / equal_nav.iloc[0]
plt.figure(figsize=(10, 4))
plt.plot(equal_nav.index, equal_nav.values)
plt.title("Equal Weighted Strategy Cumulative NAV")
plt.xlabel("Date")
plt.ylabel("NAV")
plt.grid(True)
_save_current_figure(f"{RUN_NAME}_{RUN_TS}_nav_equal_weight.png")

# ----- Cell 25 (code) -----
pct_cols = ["AnnReturn", "AnnVolatility", "MaxDrawdown", "AnnTurnover"]
for col in pct_cols:
    if col in metrics_raw_631.columns:
        metrics_raw_631[col] = metrics_raw_631[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else np.nan)

# 数值列
num_cols = ["Sharpe", "Calmar"]
for col in num_cols:
    if col in metrics_raw_631.columns:
        metrics_raw_631[col] = metrics_raw_631[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else np.nan)

metrics_raw_631

# ----- Cell 26 (code) -----
hb631_nav = np.exp(s_631.cumsum())
hb631_nav = hb631_nav / hb631_nav.iloc[0]
plt.figure(figsize=(10, 4))
plt.plot(hb631_nav.index, hb631_nav.values)
plt.title(f"0.6{HB631_EQUITY_ASSET} + 0.3{HB631_BOND_ASSET} + 0.1NHCI Strategy Cumulative NAV")
plt.xlabel("Date")
plt.ylabel("NAV")
plt.grid(True)
_save_current_figure(f"{RUN_NAME}_{RUN_TS}_nav_hb631.png")

plt.figure(figsize=(10, 4))
plt.plot(rp_nav.index, rp_nav.values, label="Risk Parity")
plt.plot(invvol_nav.index, invvol_nav.values, label="Inverse Volatility")
plt.plot(equal_nav.index, equal_nav.values, label="Equal Weight")
plt.plot(hb631_nav.index, hb631_nav.values, label=f"{HB631_EQUITY_ASSET}/{HB631_BOND_ASSET}/NHCI 6:3:1")
plt.title("Strategy NAV Comparison")
plt.xlabel("Date")
plt.ylabel("NAV")
plt.grid(True)
plt.legend()
_save_current_figure(f"{RUN_NAME}_{RUN_TS}_nav_comparison.png")

# ----- Cell 27 (code) -----
print("""最后5次调仓权重分配情况:\t""")
print(weights_df.tail(5))

print("""\n最后5次调仓权重分配和""")
print(weights_df.sum(axis=1).tail(5))
     
print("""\n权重表中出现过的最小仓位""")
print(weights_df.min().min())

# ----- Cell 28 (markdown) -----
# # 输入和输出存档代码

# ----- Cell 29 (code) -----
# =========================
# 模型 1（无杠杆 Risk Parity）专用输出：只输出该模型现有的东西
# 不输出任何 leverage / target_vol / levered NAV / levered returns
# =========================
def _params_to_sheet_df(params: dict) -> pd.DataFrame:
    """把任意 dict（可嵌套）展开成 Key / Value 两列，写进 Excel 更好读。"""
    rows = []

    def _walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, (list, tuple)):
            rows.append({"Key": prefix, "Value": json.dumps(obj, ensure_ascii=False)})
        else:
            rows.append({"Key": prefix, "Value": "" if obj is None else str(obj)})

    _walk("", params or {})
    return pd.DataFrame(rows).sort_values("Key").reset_index(drop=True)


def _dtindex_to_date(obj):
    """
    DataFrame / Series 的 DatetimeIndex -> datetime.date（不带时间）
    让 Excel 里 RebalanceDate / Trddt 只显示 YYYY-MM-DD
    """
    if obj is None:
        return None
    if isinstance(obj, (pd.Series, pd.DataFrame)) and isinstance(obj.index, pd.DatetimeIndex):
        obj = obj.copy()
        obj.index = pd.Index(obj.index.date, name=obj.index.name)
    return obj


def save_risk_parity_model1_two_files(
    output_dir: str,
    run_name: str,
    # ===== 必含（模型1一定有）=====
    params: dict,
    weights_df: pd.DataFrame,
    metrics_raw: pd.DataFrame,
    # ===== 可选（模型1里你现在也有，但允许你不传）=====
    ret_sub: pd.DataFrame | None = None,
    cov_matrix: pd.DataFrame | None = None,
    corr_matrix: pd.DataFrame | None = None,
    turnover: pd.Series | None = None,
    port_log_ret: pd.Series | None = None,
    nav: pd.Series | None = None,
    risk_contribution: pd.DataFrame | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    ts = RUN_TS
    base = f"{run_name}_{ts}"

    # ---------- 1) params.json ----------
    params_path = os.path.join(output_dir, f"{base}_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2, default=str)

    # ---------- 2) 统一日期粒度（去掉时分秒） ----------
    weights_df   = _dtindex_to_date(weights_df)     # RebalanceDate
    turnover     = _dtindex_to_date(turnover)       # RebalanceDate
    ret_sub      = _dtindex_to_date(ret_sub)        # Trddt
    port_log_ret = _dtindex_to_date(port_log_ret)   # Trddt
    nav          = _dtindex_to_date(nav)            # Trddt
    risk_contribution = _dtindex_to_date(risk_contribution)  # RebalanceDate

    # 协方差/相关一般 index/columns 是资产名，不需要日期处理，但传进来就不动也没关系
    # metrics_raw 的 StartDate/EndDate 若是 Timestamp，也转成 date（更干净）
    if isinstance(metrics_raw, pd.DataFrame):
        metrics_raw = metrics_raw.copy()
        for c in ["StartDate", "EndDate"]:
            if c in metrics_raw.columns:
                metrics_raw[c] = pd.to_datetime(metrics_raw[c], errors="coerce").dt.date

    # ---------- 3) results.xlsx（只写模型1相关的 sheet） ----------
    xlsx_path = os.path.join(output_dir, f"{base}_results.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # 必含
        _params_to_sheet_df(params).to_excel(writer, sheet_name="params", index=False)
        weights_df.to_excel(writer, sheet_name="weights", index=True)
        metrics_raw.to_excel(writer, sheet_name="metrics_raw", index=True)

        # 可选（存在就写）
        if ret_sub is not None:
            ret_sub.to_excel(writer, sheet_name="ret_sub_logret", index=True)
        if cov_matrix is not None:
            cov_matrix.to_excel(writer, sheet_name="cov_matrix", index=True)
        if corr_matrix is not None:
            corr_matrix.to_excel(writer, sheet_name="corr_matrix", index=True)
        if turnover is not None:
            turnover.rename("turnover").to_frame().to_excel(writer, sheet_name="turnover", index=True)
        if port_log_ret is not None:
            port_log_ret.rename("port_log_ret").to_frame().to_excel(writer, sheet_name="port_log_ret", index=True)
        if nav is not None:
            nav.rename("nav").to_frame().to_excel(writer, sheet_name="nav", index=True)
        if risk_contribution is not None:
            risk_contribution.to_excel(writer, sheet_name="risk_contribution", index=True)

    print("✅ Model 1 saved (exactly 2 files, no leverage outputs):")
    print(" -", xlsx_path)
    print(" -", params_path)
    return xlsx_path, params_path


# =========================
# 模型 1 的 params_snapshot（注意：这里不包含任何 leverage 参数）
# =========================
params_snapshot_model1 = {
    "data_file": os.path.basename(target_docu),
    "date_filter": {"start_date": start_date, "end_date": end_date},
    "strategy_start": STRATEGY_START,
    "lookback_months": LOOKBACK_MONTHS,
    "annual_days": ANNUAL_DAYS,
    "rf_annual": RF_ANNUAL,
    "long_only": LONG_ONLY,
    "selected_assets": assets,
    "output_dir": OUTPUT_DIR,
    "figure_dir": FIGURE_DIR,
    "min_window_days_guard": 60,
    "optimizer": {"method": "SLSQP", "ftol": 1e-12, "maxiter": 2000},
    "missing_value_rule": "ret_sub_raw selected assets are filled with 0.0 for backtest alignment; corr_matrix uses pairwise non-NaN raw returns",
    "risk_contribution_rule": f"Risk contribution is calculated at each rebalance date from the past {LOOKBACK_MONTHS} months of return covariance, then normalized by total risk contribution",
    "benchmark": f"{HB631_EQUITY_ASSET}:{HB631_BOND_ASSET}:NHCI = 6:3:1",
}

# =========================
# 调用
# =========================
save_risk_parity_model1_two_files(
    output_dir=OUTPUT_DIR,
    run_name=RUN_NAME,
    params=params_snapshot_model1,
    weights_df=weights_df,
    metrics_raw=metrics_raw,
    ret_sub=ret_sub,
    cov_matrix=cov_matrix,
    corr_matrix=corr_matrix,
    turnover=turnover,
    port_log_ret=port_log_ret,
    nav=rp_nav,
    risk_contribution=risk_contribution_df,
)
