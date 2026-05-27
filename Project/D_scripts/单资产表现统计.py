# ===== AI-Friendly Python Version of Notebook =====
# Source Notebook: /Users/chloezh/Projects/jupyter_to_py_project/input_jupyter/单资产表现统计.ipynb

# ----- Cell 1 (code) -----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib.pyplot as plt

# ----- Cell 2 (code) -----
df = pd.read_excel("ALL_ASSETS.xlsx")
asset_cols = df["Indexcd"].dropna().astype(str).unique().tolist()
asset_cols

# ----- Cell 3 (code) -----
df["Trddt"] = pd.to_datetime(df["Trddt"])
df["Clsidx"] = pd.to_numeric(df["Clsidx"], errors="coerce")

#START_DATE = "2023-1-3"
#END_DATE   = "2026-02-26"
#df = df[(df["Trddt"] >= START_DATE) & (df["Trddt"] <= END_DATE)]

out_root = Path("Asset performance")
out_root.mkdir(parents=True, exist_ok=True)
performance = []

df = df.dropna()
df = df.sort_values(["Trddt"]).reset_index(drop=True)
df.head(20)

# ----- Cell 4 (markdown) -----
# # A. 常数无风险利率

# ----- Cell 5 (code) -----
ANNUAL_DAYS = 252

# ===== 用户输入（当 df 里没有 Rf 列时才会用到）=====
RF_ANNUAL_INPUT = 0.015   # 例如 1.5% 年化无风险利率；

# ===== 如果 df 中没有 Rf 列，则用年化 rf 自动生成“日度对数 rf”=====
if "Rf" not in df.columns:
    # 年化 rf → 日度对数 rf
    rf_daily_log = np.log(1 + RF_ANNUAL_INPUT) / ANNUAL_DAYS
    
    df["Rf"] = rf_daily_log

# ----- Cell 6 (markdown) -----
# # B. 无风险利率数据列

# ----- Cell 7 (code) -----
# 读入 rf（示例：rf_df 里有 Trddt, Rf）
# rf_df = pd.read_excel("RF.xlsx")

rf_df["Trddt"] = pd.to_datetime(rf_df["Trddt"])
rf_df["Rf"] = pd.to_numeric(rf_df["Rf"], errors="coerce")

# 按日期合并到资产 df
df = df.merge(rf_df[["Trddt", "Rf"]], on="Trddt", how="left")

# 删除 rf 缺失的日期（或你也可以选择 forward fill）
df = df.dropna(subset=["Rf"])

df.head()

ANNUAL_DAYS = 252

# ----- Cell 8 (markdown) -----
# # 继续原程序

# ----- Cell 9 (code) -----
for asset in asset_cols:
    df_asset =df.loc[df["Indexcd"]  == asset].copy()
    df_asset["log_ret"] = np.log(df_asset["Clsidx"]) - np.log(df_asset["Clsidx"].shift(1))
    df_asset = df_asset.dropna(subset=["log_ret"])
    #净值、回撤
    df_asset["NAV"] = np.exp(df_asset["log_ret"].cumsum())
    df_asset["NAV_max"] = df_asset["NAV"].cummax()
    df_asset["Drawdown"] = df_asset["NAV"] / df_asset["NAV_max"] - 1.0

    n = len(df_asset)
    start_date = df_asset["Trddt"].iloc[0]
    end_date = df_asset["Trddt"].iloc[-1]

    # ===== 累计收益率（几何累计）=====
    # 如果 log_ret 是 ln(1+r)，那累计收益 = exp(sum(log_ret)) - 1
    total_log_return = df_asset["log_ret"].sum()
    cum_return = np.exp(total_log_return) - 1
    ann_return = np.exp(total_log_return * ANNUAL_DAYS / n) - 1
    ann_vol = df_asset["log_ret"].std(ddof=1) * np.sqrt(ANNUAL_DAYS)
    # 简单波动率
    #simple_ret = np.exp(df["log_ret"]) - 1
    #ann_vol = simple_ret.std(ddof=1) * np.sqrt(ANNUAL_DAYS)

    # ===== 最大回撤 =====
    max_drawdown = df_asset["Drawdown"].min()  # 负数

    # ===== 年化无风险收益（来自时间序列）=====
    ann_rf = RF_ANNUAL_INPUT

    # ===== 夏普比率（基于几何年化收益）=====
    sharpe = (ann_return - ann_rf) / ann_vol if ann_vol != 0 else np.nan

    # ===== 卡玛比率 =====
    # Calmar = 年化收益 / |最大回撤|
    calmar = ann_return / abs(max_drawdown) if (max_drawdown is not None and max_drawdown < 0) else np.nan

    metrics = pd.DataFrame([{
        "Asset": asset,
        "StartDate": start_date,
        "EndDate": end_date,
        "Obs": n,
        "CumReturn": cum_return,
        "AnnReturn": ann_return,
        "AnnRf": ann_rf,
        "AnnVolatility": ann_vol,
        "MaxDrawdown": max_drawdown,
        "Sharpe": sharpe,
        "Calmar": calmar
    }])

    metrics_fmt = metrics.copy()

    metrics_fmt["StartDate"] = metrics_fmt["StartDate"].dt.strftime("%Y-%m-%d")
    metrics_fmt["EndDate"]   = metrics_fmt["EndDate"].dt.strftime("%Y-%m-%d")

    metrics_fmt["CumReturn"] = metrics_fmt["CumReturn"].map(lambda x: f"{x:.2%}")
    metrics_fmt["AnnReturn"]      = metrics_fmt["AnnReturn"].map(lambda x: f"{x:.2%}")
    metrics_fmt["AnnRf"]          = metrics_fmt["AnnRf"].map(lambda x: f"{x:.2%}")
    metrics_fmt["AnnVolatility"]  = metrics_fmt["AnnVolatility"].map(lambda x: f"{x:.2%}")
    metrics_fmt["MaxDrawdown"]    = metrics_fmt["MaxDrawdown"].map(lambda x: f"{x:.2%}")
    metrics_fmt["Sharpe"]         = metrics_fmt["Sharpe"].map(lambda x: f"{x:.2f}")

    performance.append(metrics_fmt) #把metrics加在performance表格后面

    #保存每个资产三张图
    asset_dir = out_root / f"{asset} performance"
    asset_dir.mkdir(parents=True, exist_ok=True)

    # 图1：NAV
    fig = plt.figure(figsize=(10, 4))
    plt.plot(df_asset["Trddt"], df_asset["NAV"], linewidth=2)
    plt.title(f"Net Asset Value (Log-return based) - {asset}")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(asset_dir / "01_NAV.png", dpi=200)
    plt.close(fig)

    # 图2：Drawdown
    fig = plt.figure(figsize=(10, 3))
    plt.plot(df_asset["Trddt"], df_asset["Drawdown"], linewidth=1.5)
    plt.title(f"Drawdown - {asset}")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(asset_dir / "02_Drawdown.png", dpi=200)
    plt.close(fig)

    # 图3：NAV + 最大回撤点
    dd_idx = df_asset["Drawdown"].idxmin()
    dd_date = df_asset.loc[dd_idx, "Trddt"]
    dd_nav = df_asset.loc[dd_idx, "NAV"]

    fig = plt.figure(figsize=(10, 4))
    plt.plot(df_asset["Trddt"], df_asset["NAV"], linewidth=2)
    plt.scatter(dd_date, dd_nav, zorder=5)
    plt.title(f"NAV with Maximum Drawdown Point - {asset}")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(asset_dir / "03_NAV_with_MaxDD.png", dpi=200)
    plt.close(fig)

# ----- Cell 10 (code) -----
# =========================
# 汇总输出：performance_all
# =========================
performance_all = pd.concat(performance, ignore_index=True) if performance else pd.DataFrame()

# 可选：导出总表到 performance/metrics_all.csv
if not performance_all.empty:
    performance_all.to_csv(out_root / "metrics_all.csv", index=False, encoding="utf-8-sig")

performance_all


# ----- Cell 11 (markdown) -----
# # 资产协方差

# ----- Cell 12 (code) -----
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df["log_ret"] = df.groupby("Indexcd")["Clsidx"].transform(
    lambda x: np.log(x) - np.log(x.shift(1))
)


df = df.dropna(subset=["log_ret"])

ret_sub = (df
              .pivot(index="Trddt", columns="Indexcd", values="log_ret")
              .sort_index()
             )

ret_sub.head()

cov_matrix = ret_sub.cov()

ANNUAL_DAYS = 252
cov_matrix_annual = cov_matrix * ANNUAL_DAYS


plt.figure(figsize=(8, 6))

sns.heatmap(
    cov_matrix_annual,
    annot=True,
    fmt=".4f",
    cmap="YlOrRd",
    linewidths=0.5
)

plt.title("Annualized Covariance Matrix")
plt.tight_layout()
plt.show()

# ----- Cell 13 (code) -----
START_DATE = "2023-01-03"
END_DATE   = "2026-02-26"

df2 = df.copy()
df2["Trddt"] = pd.to_datetime(df2["Trddt"])
df2 = df2[(df2["Trddt"] >= START_DATE) & (df2["Trddt"] <= END_DATE)].copy()

df2["log_ret"] = df2.groupby("Indexcd")["Clsidx"].transform(lambda x: np.log(x).diff())
df2 = df2.dropna(subset=["log_ret"])

ret_sub2 = (df2.pivot(index="Trddt", columns="Indexcd", values="log_ret")
              .sort_index())

cov_matrix2 = ret_sub2.cov()

ANNUAL_DAYS = 252
cov_matrix_annual2 = cov_matrix2 * ANNUAL_DAYS

plt.figure(figsize=(8, 6))

sns.heatmap(
    cov_matrix_annual2,
    annot=True,
    fmt=".4f",
    cmap="YlOrRd",
    linewidths=0.5
)

plt.title("Annualized Covariance Matrix")
plt.tight_layout()
plt.show()

# ----- Cell 14 (code) -----
plt.figure(figsize=(8, 6))

sns.heatmap(
    cov_matrix_annual2,
    annot=True,
    fmt=".4f",
    cmap="YlOrRd",
    linewidths=0.5
)

plt.title("Annualized Covariance Matrix")
plt.tight_layout()
plt.show()

# ----- Cell 15 (code) -----
