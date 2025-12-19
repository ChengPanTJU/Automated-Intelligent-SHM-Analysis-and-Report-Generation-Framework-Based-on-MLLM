"""
交通荷载分析

Created on 2024

@author: Gong Fengzong
"""
import json
import os
import hashlib
from typing import Union, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import FileConfig, TrafConfig, OutputConfig, traf_config
from matplotlib.colors import LogNorm
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["mathtext.fontset"] = "dejavusans"   # 关键：colorbar 的 10^{...} 用这个字体



# =========================
# 配置索引读取工具
# =========================
def _idx_list(conf, attr: str) -> List[int]:
    v = getattr(conf, attr, [])
    if v is None:
        return []
    try:
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return []
    except Exception:
        pass

    if isinstance(v, (list, tuple, np.ndarray, pd.Index)):
        out: List[int] = []
        for x in v:
            if x is None:
                continue
            try:
                if isinstance(x, (float, np.floating)) and np.isnan(x):
                    continue
            except Exception:
                pass
            out.append(int(x))
        return out

    return [int(v)]


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# =========================
# 绘图工具
# =========================
def _cm_to_inch(x_cm: float) -> float:
    return x_cm * 0.393701


def set_plot_style() -> None:
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12


def save_close(fig, fig_path: str, dpi: int = 300) -> None:
    _ensure_parent_dir(fig_path)
    fig.tight_layout()
    fig.savefig(fig_path, format='png', dpi=dpi)
    plt.close(fig)


def _cycle_color(i: int) -> str:
    colors = plt.rcParams.get('axes.prop_cycle', None)
    if colors is not None:
        clist = colors.by_key().get('color', [])
        if clist:
            return clist[i % len(clist)]
    return f"C{i}"


def plot_placeholder(fig_path: str, title: str, msg: str) -> None:
    fig = plt.figure(figsize=(_cm_to_inch(14), _cm_to_inch(8)))
    plt.axis("off")
    plt.title(title, fontsize=12)
    plt.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
    save_close(fig, fig_path)


def plot_hourly_single_line(summary: pd.DataFrame,
                            xcol: str,
                            ycol: str,
                            ylabel: str,
                            title: str,
                            fig_path: str) -> None:
    if summary.empty or (ycol not in summary.columns):
        return

    fig = plt.figure(figsize=(_cm_to_inch(14), _cm_to_inch(8)))
    plt.plot(summary[xcol], summary[ycol], linewidth=0.25)

    plt.xlabel('时间 (月-日）', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=12)

    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=10)
    plt.xticks(rotation=30)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    year = pd.to_datetime(summary[xcol].iloc[-1]).year
    plt.annotate(f'{year}', xy=(1, 0), xycoords='axes fraction', fontsize=12,
                 xytext=(0, -30), textcoords='offset points', ha='right', va='top')

    save_close(fig, fig_path)


def _hist_density_bar(series: pd.Series, bins: int, range_: Union[None, Tuple[float, float]] = None):
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return None, None, None, None
    hist_values, bin_edges = np.histogram(s, bins=bins, density=True, range=range_)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
    return bin_centers, hist_values, width, bin_edges


# -------------------------
# 两张 PDF：双 y 轴 + y 轴颜色与柱色一致
# -------------------------
def plot_pdf_bar_compare_dual_y(series_all: pd.Series,
                                series_sub: pd.Series,
                                bins: int,
                                xlabel: str,
                                title: str,
                                fig_path: str,
                                label_all: str = "全部",
                                label_sub: str = "子集") -> None:
    bc_a, hv_a, w_a, _ = _hist_density_bar(series_all, bins=bins)
    if bc_a is None:
        return

    bc_b, hv_b, w_b, _ = _hist_density_bar(series_sub, bins=bins)

    c1 = _cycle_color(0)
    c2 = _cycle_color(1)

    fig, ax1 = plt.subplots(figsize=(_cm_to_inch(14), _cm_to_inch(8)))
    ax1.bar(bc_a, hv_a, width=w_a, alpha=0.55, edgecolor='black', linewidth=1.0,
            label=label_all, color=c1)
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('概率密度', fontsize=12, color=c1)
    ax1.tick_params(axis='y', labelsize=10, colors=c1)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.set_title(title, fontsize=12)

    ax2 = None
    if bc_b is not None:
        ax2 = ax1.twinx()
        ax2.bar(bc_b, hv_b, width=w_b, alpha=0.55, edgecolor='black', linewidth=1.0,
                label=label_sub, color=c2)
        ax2.set_ylabel('概率密度', fontsize=12, color=c2)
        ax2.tick_params(axis='y', labelsize=10, colors=c2)

    h1, l1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=10)
    else:
        ax1.legend(h1, l1, fontsize=10)

    save_close(fig, fig_path)


def plot_speed_pdf_with_limit_dual_y(speed_series: pd.Series,
                                     is_overspeed: pd.Series,
                                     speed_limit: Union[None, float],
                                     fig_path: str) -> None:
    s_all = pd.to_numeric(speed_series, errors='coerce').dropna()
    if s_all.empty:
        return

    mask = is_overspeed.fillna(False).astype(bool)
    s_over = pd.to_numeric(speed_series[mask], errors='coerce').dropna()

    vmax = float(s_all.max())
    range_ = (0.0, vmax) if np.isfinite(vmax) and vmax > 0 else None

    bc_a, hv_a, w_a, _ = _hist_density_bar(s_all, bins=30, range_=range_)
    if bc_a is None:
        return

    c1 = _cycle_color(0)
    c2 = _cycle_color(1)

    fig, ax1 = plt.subplots(figsize=(_cm_to_inch(14), _cm_to_inch(8)))
    ax1.bar(bc_a, hv_a, width=w_a, alpha=0.55, edgecolor='black', linewidth=1.0,
            label="全部", color=c1)

    ax1.set_xlabel('车速 (km/h)', fontsize=12)
    ax1.set_ylabel('概率密度', fontsize=12, color=c1)
    ax1.tick_params(axis='y', labelsize=10, colors=c1)
    ax1.tick_params(axis='x', labelsize=10)
    if range_ is not None:
        ax1.set_xlim(range_)

    ax2 = None
    if not s_over.empty:
        bc_b, hv_b, w_b, _ = _hist_density_bar(s_over, bins=30, range_=range_)
        if bc_b is not None:
            ax2 = ax1.twinx()
            ax2.bar(bc_b, hv_b, width=w_b, alpha=0.55, edgecolor='black', linewidth=1.0,
                    label="超速", color=c2)
            ax2.set_ylabel('概率密度', fontsize=12, color=c2)
            ax2.tick_params(axis='y', labelsize=10, colors=c2)

    if speed_limit is not None:
        ax1.axvline(float(speed_limit), linestyle="--", linewidth=1.0, label=f"限速 {speed_limit:g} km/h")

    ax1.set_title("车速概率密度分布", fontsize=12)

    h1, l1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=10)
    else:
        ax1.legend(h1, l1, fontsize=10)

    save_close(fig, fig_path)


def plot_lane_weight_pdfs(data: pd.DataFrame,
                          lane_col: str,
                          weight_col: str,
                          fig_path: str,
                          bins: int = 50) -> None:
    lanes = sorted([x for x in data[lane_col].dropna().unique()])
    n = len(lanes)
    if n == 0:
        return

    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    width_cm = 14 if ncols == 2 else 7
    height_cm = max(10.0, 3.5 * nrows + 2.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(_cm_to_inch(width_cm), _cm_to_inch(height_cm)), sharex=False)
    axes = np.array(axes).reshape(-1)
    for ax, lane in zip(axes[:n], lanes):
        s = pd.to_numeric(data.loc[data[lane_col] == lane, weight_col], errors='coerce').dropna()
        if s.empty:
            ax.set_title(f"车道 {lane}（无数据）", fontsize=12)
            continue

        hv, be = np.histogram(s, bins=bins, density=True)
        bc = (be[:-1] + be[1:]) / 2
        ax.bar(bc, hv, width=(be[1] - be[0]), alpha=0.6, edgecolor='black', linewidth=1.0)

        ax.set_xlabel('车重 (t)', fontsize=12)
        ax.set_ylabel('概率密度', fontsize=12)
        ax.set_title(f"车道 {lane} 车重概率密度分布", fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10)
    # 关闭未使用的子图（当车道数为奇数时会出现空位）
    for ax in axes[n:]:
        ax.axis('off')

    save_close(fig, fig_path)


def plot_lane_single_count(lane_summary: pd.DataFrame,
                           lane_col: str,
                           count_col: str,
                           title: str,
                           fig_path: str) -> None:
    if lane_summary.empty or count_col not in lane_summary.columns:
        return

    fig, ax = plt.subplots(figsize=(_cm_to_inch(14), _cm_to_inch(8)))
    ax.bar(lane_summary[lane_col], lane_summary[count_col], alpha=0.7, edgecolor='black')

    for i, v in enumerate(lane_summary[count_col].to_numpy()):
        ax.text(lane_summary.iloc[i][lane_col], v + 5, str(int(v)), ha='center', fontsize=10)

    plt.xlabel('车道号', fontsize=12)
    plt.ylabel('车辆数量 (辆)', fontsize=12)
    plt.title(title, fontsize=12)

    try:
        vmin = int(np.nanmin(lane_summary[lane_col].to_numpy()))
        vmax = int(np.nanmax(lane_summary[lane_col].to_numpy()))
        plt.xticks(range(vmin, vmax + 1))
    except Exception:
        pass

    plt.tick_params(axis='y', labelsize=10)
    plt.tick_params(axis='x', labelsize=10)

    save_close(fig, fig_path)


# -------------------------
# diurnal 绘图工具
# -------------------------
def plot_diurnal_line(df: pd.DataFrame,
                      xcol: str,
                      ycol: str,
                      ylabel: str,
                      title: str,
                      fig_path: str) -> None:
    if df.empty or ycol not in df.columns:
        return
    fig = plt.figure(figsize=(_cm_to_inch(14), _cm_to_inch(8)))
    plt.plot(df[xcol], df[ycol], linewidth=0.8)
    plt.xlabel("小时 (0-23)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=12)
    plt.xticks(range(0, 24, 1), fontsize=9)
    plt.tick_params(axis='y', labelsize=10)
    save_close(fig, fig_path)


def plot_diurnal_two_lines(df: pd.DataFrame,
                           xcol: str,
                           y1: str,
                           y2: str,
                           label1: str,
                           label2: str,
                           ylabel: str,
                           title: str,
                           fig_path: str) -> None:
    if df.empty or (y1 not in df.columns) or (y2 not in df.columns):
        return
    fig = plt.figure(figsize=(_cm_to_inch(14), _cm_to_inch(8)))
    plt.plot(df[xcol], df[y1], linewidth=0.8, label=label1)
    plt.plot(df[xcol], df[y2], linewidth=0.8, label=label2)
    plt.xlabel("小时 (0-23)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=12)
    plt.xticks(range(0, 24, 1), fontsize=9)
    plt.tick_params(axis='y', labelsize=10)
    plt.legend(fontsize=10)
    save_close(fig, fig_path)


def plot_diurnal_box_by_hour(series: pd.Series,
                             hour_series: pd.Series,
                             title: str,
                             ylabel: str,
                             fig_path: str) -> None:
    s = pd.to_numeric(series, errors="coerce")
    h = pd.to_numeric(hour_series, errors="coerce")
    fig = plt.figure(figsize=(_cm_to_inch(14), _cm_to_inch(8)))
    data = []
    for hh in range(24):
        vals = s[(h == hh)].dropna().to_numpy()
        if vals.size == 0:
            vals = np.array([np.nan])
        data.append(vals)
    plt.boxplot(data, positions=list(range(24)), showfliers=False)
    plt.xlabel("小时 (0-23)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=12)
    plt.xticks(range(0, 24, 1), fontsize=9)
    plt.tick_params(axis='y', labelsize=10)
    save_close(fig, fig_path)


def plot_heatmap_hour_bin(pivot: pd.DataFrame,
                          title: str,
                          xlabel: str,
                          ylabel: str,
                          fig_path: str,
                          tick_decimals: int = 1) -> None:
    """
    修改点：
    - x 轴 bin_center 只保留 1 位小数
    - 不旋转（rotation=0）
    """
    if pivot.empty:
        return
    fig = plt.figure(figsize=(_cm_to_inch(14), _cm_to_inch(8)))
    mat = pivot.to_numpy()
    mat = np.nan_to_num(mat, nan=0.0)
    mat_plot = mat.copy()
    pos = mat_plot[mat_plot > 0]
    if pos.size > 0:
        vmin = pos.min()
        mat_plot[mat_plot <= 0] = vmin
        norm = LogNorm(vmin=vmin, vmax=mat_plot.max())
    else:
        norm = None  # 全零就没必要 log

    plt.imshow(mat_plot, norm=norm, aspect="auto", origin="lower")
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    ncol = pivot.shape[1]
    if ncol > 0:
        xt = np.linspace(0, ncol - 1, num=min(10, ncol), dtype=int)
        labels = [f"{float(pivot.columns[i]):.{tick_decimals}f}" for i in xt]
        plt.xticks(xt, labels, fontsize=9, rotation=0)

    plt.yticks(range(0, 48, 2), [str(i) for i in range(24)], fontsize=9)
    cb = plt.colorbar()
    import matplotlib.ticker as mticker

    cb.formatter = mticker.FuncFormatter(lambda x, pos: f"{x:.0e}")  # 例如 1e-3（这里的 '-' 是 ASCII）
    cb.update_ticks()
    save_close(fig, fig_path)


# =========================
# 超速/超载标志解析
# =========================
def _parse_flag_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(False)

    if s.dtype == bool:
        return s.fillna(False)

    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().any():
        return sn.fillna(0).gt(0)

    ss = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y", "是", "超速", "超重", "超载", "over", "overspeed", "overweight"}
    return ss.isin(true_set)


# =========================
# 读取优化：只读必要列 + 读入即时间过滤
# =========================
def _excel_col_letters(n0: int) -> str:
    n = n0 + 1
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _make_excel_usecols(indices: List[int]) -> str:
    cols = [_excel_col_letters(i) for i in sorted(set(indices))]
    return ",".join(cols)


def _compute_required_indices(traf_conf) -> List[int]:
    traffic_time = getattr(traf_conf, "traffic_time", None)
    if traffic_time is None:
        raise ValueError("TrafConfig.traffic_time 未配置。")

    total_weight = _idx_list(traf_conf, "total_weight")
    lane_num = _idx_list(traf_conf, "lane_num")
    speed = _idx_list(traf_conf, "speed")
    axle_weight = _idx_list(traf_conf, "axle_weight")
    over_weight_index = _idx_list(traf_conf, "over_weight_index")
    over_speed_index = _idx_list(traf_conf, "over_speed_index")

    left_axle_weight = _idx_list(traf_conf, "left_axle_weight")
    right_axle_weight = _idx_list(traf_conf, "right_axle_weight")

    if not total_weight:
        raise ValueError("TrafConfig.total_weight 未配置或为空（如 [6]）。")

    idxs = [traffic_time, total_weight[0]]
    if lane_num:
        idxs.append(lane_num[0])
    if speed:
        idxs.append(speed[0])

    idxs += axle_weight
    idxs += over_weight_index
    idxs += over_speed_index
    idxs += left_axle_weight
    idxs += right_axle_weight

    idxs = [i for i in idxs if isinstance(i, int) and i >= 0]
    return sorted(set(idxs))


def _read_csv_header(fp: str) -> Tuple[List[str], str]:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            header_df = pd.read_csv(fp, encoding=enc, low_memory=False, nrows=0)
            return list(header_df.columns), enc
        except Exception:
            continue
    header_df = pd.read_csv(fp, low_memory=False, nrows=0)
    return list(header_df.columns), "auto"


def load_traffic_files_filtered(traffic_path: str, traf_conf, file_conf) -> pd.DataFrame:
    time_format = getattr(traf_conf, "time_format", None)
    traffic_time = getattr(traf_conf, "traffic_time", None)
    if not time_format:
        raise ValueError("TrafConfig.time_format 未配置或为空。")
    if traffic_time is None:
        raise ValueError("TrafConfig.traffic_time 未配置。")

    total_weight_idx = _idx_list(traf_conf, "total_weight")
    if not total_weight_idx:
        raise ValueError("TrafConfig.total_weight 未配置或为空（如 [6]）。")

    must_idx = sorted(set([traffic_time, total_weight_idx[0]]))

    start_time = pd.to_datetime(file_conf.start_time, format="%Y-%m-%d %H", errors="coerce")
    end_time = pd.to_datetime(file_conf.end_time, format="%Y-%m-%d %H", errors="coerce")
    if pd.isna(start_time) or pd.isna(end_time):
        raise ValueError("FileConfig.start_time / end_time 无法解析，请确认格式 '%Y-%m-%d %H'。")

    use_idx = _compute_required_indices(traf_conf)

    files = [f for f in os.listdir(traffic_path) if f.endswith(('.xlsx', '.xls', '.csv'))]
    if not files:
        raise FileNotFoundError(f"在目录中未找到数据文件：{traffic_path}")

    kept: List[pd.DataFrame] = []

    for file in files:
        fp = os.path.join(traffic_path, file)

        if file.lower().endswith(".csv"):
            colnames, enc = _read_csv_header(fp)
            ncol = len(colnames)

            valid_idx = [i for i in use_idx if i < ncol]
            if (not valid_idx) or (not all(i in valid_idx for i in must_idx)):
                continue

            usecols_names = [colnames[i] for i in valid_idx]
            rename_map = {colnames[i]: f"c{i}" for i in valid_idx}

            row_base = 0
            for chunk in pd.read_csv(fp, encoding=(enc if enc != "auto" else None),
                                     low_memory=False, usecols=usecols_names, chunksize=200000):
                chunk = chunk.copy()
                chunk.rename(columns=rename_map, inplace=True)

                chunk["source_file"] = file
                chunk["source_row"] = np.arange(len(chunk), dtype=int) + row_base
                row_base += len(chunk)

                tcol = f"c{traffic_time}"
                chunk["pass_time"] = pd.to_datetime(chunk[tcol].astype(str), format=time_format, errors="coerce")
                m = chunk["pass_time"].notna() & (chunk["pass_time"] >= start_time) & (chunk["pass_time"] <= end_time)
                if m.any():
                    kept.append(chunk.loc[m].copy())

        else:
            header_df = pd.read_excel(fp, nrows=0)
            ncol = len(header_df.columns)

            valid_idx = [i for i in use_idx if i < ncol]
            if (not valid_idx) or (not all(i in valid_idx for i in must_idx)):
                continue

            excel_usecols = _make_excel_usecols(valid_idx)
            df = pd.read_excel(fp, usecols=excel_usecols).copy()

            valid_idx_sorted = sorted(valid_idx)
            df.columns = [f"c{i}" for i in valid_idx_sorted]

            df["source_file"] = file
            df["source_row"] = np.arange(len(df), dtype=int)

            tcol = f"c{traffic_time}"
            df["pass_time"] = pd.to_datetime(df[tcol].astype(str), format=time_format, errors="coerce")
            m = df["pass_time"].notna() & (df["pass_time"] >= start_time) & (df["pass_time"] <= end_time)
            if m.any():
                kept.append(df.loc[m].copy())

    if not kept:
        return pd.DataFrame()

    return pd.concat(kept, ignore_index=True)


# =========================
# 标准表 + QC + 超载/超速字段
# =========================
def build_vehicle_pass_and_qc_inrange(raw_inrange: pd.DataFrame,
                                      traf_conf,
                                      qc_cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], List[str]]:
    qc_out_dir = qc_cfg.get('save_dir', '')
    if not qc_out_dir:
        raise ValueError("QC 配置缺少 save_dir：outputconfig['traffic']['qc']['save_dir']")

    qc_files = qc_cfg.get('files', {}) or {}
    f_all = qc_files.get('pass_all_inrange', 'vehicle_pass_all_inrange.parquet')
    f_clean = qc_files.get('pass_clean', 'vehicle_pass_clean.parquet')
    f_anom = qc_files.get('anomalies', 'qc_anomalies.csv')
    f_report = qc_files.get('report', 'qc_report.json')

    os.makedirs(qc_out_dir, exist_ok=True)

    total_weight_idx = _idx_list(traf_conf, "total_weight")
    lane_idx = _idx_list(traf_conf, "lane_num")
    speed_idx = _idx_list(traf_conf, "speed")
    axle_idx = _idx_list(traf_conf, "axle_weight")

    over_weight_index = _idx_list(traf_conf, "over_weight_index")
    over_speed_index = _idx_list(traf_conf, "over_speed_index")
    over_speed_limit = _idx_list(traf_conf, "over_speed_limit")

    if not total_weight_idx:
        raise ValueError("TrafConfig.total_weight 未配置或为空（如 [6]）。")

    weight_divisor = float(getattr(traf_conf, "weight_divisor", 1000.0))

    speed_min = float(getattr(traf_conf, "speed_min_kmh", 0.0))
    speed_max = float(getattr(traf_conf, "speed_max_kmh", 160.0))
    axle_abs_tol_t = float(getattr(traf_conf, "axle_sum_abs_tol_t", 0.30))
    axle_rel_tol = float(getattr(traf_conf, "axle_sum_rel_tol", 0.03))

    df = raw_inrange.copy()

    wcol = f"c{total_weight_idx[0]}"
    if wcol not in df.columns:
        raise ValueError(f"总重列不存在：{wcol}（请检查 TrafConfig.total_weight）。")

    df["gross_t"] = pd.to_numeric(df[wcol], errors="coerce") / weight_divisor

    if lane_idx and f"c{lane_idx[0]}" in df.columns:
        df["lane"] = pd.to_numeric(df[f"c{lane_idx[0]}"], errors="coerce")
    else:
        df["lane"] = np.nan

    if speed_idx and f"c{speed_idx[0]}" in df.columns:
        df["speed_kmh"] = pd.to_numeric(df[f"c{speed_idx[0]}"], errors="coerce")
    else:
        df["speed_kmh"] = np.nan

    axle_cols_std: List[str] = []
    if axle_idx:
        for k, idx in enumerate(axle_idx, start=1):
            c = f"c{idx}"
            if c in df.columns:
                std = f"axle_{k}_t"
                df[std] = pd.to_numeric(df[c], errors="coerce") / weight_divisor
                axle_cols_std.append(std)

    if over_weight_index and f"c{over_weight_index[0]}" in df.columns:
        df["is_overweight"] = _parse_flag_series(df[f"c{over_weight_index[0]}"])
    else:
        df["is_overweight"] = False

    speed_limit_value: Union[None, float] = None
    if over_speed_index and f"c{over_speed_index[0]}" in df.columns:
        df["is_overspeed"] = _parse_flag_series(df[f"c{over_speed_index[0]}"])
        df["overspeed_excess_kmh"] = np.nan
    else:
        speed_limit_value = float(over_speed_limit[0]) if over_speed_limit else None
        if (speed_limit_value is not None) and df["speed_kmh"].notna().any():
            df["is_overspeed"] = df["speed_kmh"].gt(speed_limit_value).fillna(False)
            df["overspeed_excess_kmh"] = (df["speed_kmh"] - speed_limit_value).clip(lower=0)
        else:
            df["is_overspeed"] = False
            df["overspeed_excess_kmh"] = np.nan

    def _eid(sf, sr, pt, ln):
        t = "" if pd.isna(pt) else pd.Timestamp(pt).isoformat()
        s = f"{sf}|{int(sr)}|{t}|{ln}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    df["event_id"] = [
        _eid(sf, sr, pt, ln)
        for sf, sr, pt, ln in zip(df["source_file"], df["source_row"], df["pass_time"], df["lane"])
    ]

    qc_time_parse_fail = df["pass_time"].isna()
    qc_gross_missing = df["gross_t"].isna()
    qc_gross_nonpositive = df["gross_t"].notna() & (df["gross_t"] <= 0)

    if speed_idx:
        qc_speed_missing = df["speed_kmh"].isna()
        qc_speed_out_of_range = df["speed_kmh"].notna() & (
            (df["speed_kmh"] < speed_min) | (df["speed_kmh"] > speed_max)
        )
    else:
        qc_speed_missing = pd.Series(False, index=df.index)
        qc_speed_out_of_range = pd.Series(False, index=df.index)

    if axle_cols_std:
        axle_mat = df[axle_cols_std]
        df["axle_sum_t"] = axle_mat.sum(axis=1, skipna=True)
        df["axle_count_pos"] = (axle_mat.fillna(0) > 0).sum(axis=1)
        df["axle_diff_t"] = df["gross_t"] - df["axle_sum_t"]
        df["axle_rel_diff"] = np.where(df["gross_t"].notna() & (df["gross_t"] > 0),
                                       df["axle_diff_t"] / df["gross_t"],
                                       np.nan)
        tol_t = np.maximum(axle_abs_tol_t, axle_rel_tol * df["gross_t"].fillna(0))
        qc_axle_sum_mismatch = (
            df["gross_t"].notna() & (df["gross_t"] > 0) &
            (df["axle_count_pos"] >= 2) &
            (df["axle_diff_t"].abs() > tol_t)
        )
        qc_axle_info_missing = (
            df["gross_t"].notna() & (df["gross_t"] > 0) &
            (df["axle_count_pos"] < 2)
        )
    else:
        df["axle_sum_t"] = np.nan
        df["axle_count_pos"] = np.nan
        df["axle_diff_t"] = np.nan
        df["axle_rel_diff"] = np.nan
        qc_axle_sum_mismatch = pd.Series(False, index=df.index)
        qc_axle_info_missing = pd.Series(False, index=df.index)

    QC_RULES = [
        "qc_time_parse_fail",
        "qc_gross_missing",
        "qc_gross_nonpositive",
        "qc_speed_missing",
        "qc_speed_out_of_range",
        "qc_axle_sum_mismatch",
        "qc_axle_info_missing",
    ]
    qc_dict = {
        "qc_time_parse_fail": qc_time_parse_fail,
        "qc_gross_missing": qc_gross_missing,
        "qc_gross_nonpositive": qc_gross_nonpositive,
        "qc_speed_missing": qc_speed_missing,
        "qc_speed_out_of_range": qc_speed_out_of_range,
        "qc_axle_sum_mismatch": qc_axle_sum_mismatch,
        "qc_axle_info_missing": qc_axle_info_missing,
    }

    mask = np.zeros(len(df), dtype=np.uint32)
    for i, r in enumerate(QC_RULES):
        mask |= (qc_dict[r].to_numpy(dtype=bool).astype(np.uint32) << i)

    df["qc_mask"] = mask
    df["qc_ok"] = df["qc_mask"].eq(0)

    base_cols = [
        "event_id", "pass_time", "lane", "speed_kmh",
        "gross_t",
        "is_overweight", "is_overspeed", "overspeed_excess_kmh",
        "axle_sum_t", "axle_diff_t", "axle_rel_diff", "axle_count_pos",
        *axle_cols_std,
        "source_file", "source_row",
        "qc_ok", "qc_mask",
    ]
    vehicle_pass_all = df[base_cols].copy()
    vehicle_pass_clean = vehicle_pass_all[vehicle_pass_all["qc_ok"]].copy()

    qc_anomalies = vehicle_pass_all[~vehicle_pass_all["qc_ok"]].copy()
    if not qc_anomalies.empty:
        qc_anomalies["qc_reason"] = qc_anomalies["qc_mask"].apply(lambda x: str(int(x)))

    qc_report = {
        "counts": {
            "in_range_rows": int(len(vehicle_pass_all)),
            "qc_ok_rows": int(vehicle_pass_all["qc_ok"].sum()),
            "qc_bad_rows": int((~vehicle_pass_all["qc_ok"]).sum()),
            "qc_ok_rate": float(vehicle_pass_all["qc_ok"].mean()) if len(vehicle_pass_all) else None,
        },
        "rules_hit_counts": {r: int(qc_dict[r].sum()) for r in QC_RULES},
        "speed_limit_used_kmh": speed_limit_value,
    }

    # ——按 outputconfig 控制的文件名落盘——
    vehicle_pass_all.to_parquet(os.path.join(qc_out_dir, f_all), index=False)
    vehicle_pass_clean.to_parquet(os.path.join(qc_out_dir, f_clean), index=False)
    qc_anomalies.to_csv(os.path.join(qc_out_dir, f_anom), index=False, encoding="utf-8-sig")

    with open(os.path.join(qc_out_dir, f_report), "w", encoding="utf-8") as fw:
        json.dump(qc_report, fw, ensure_ascii=False, indent=2)

    return vehicle_pass_all, vehicle_pass_clean, qc_report, axle_cols_std

    qc_report = {
        "counts": {
            "in_range_rows": int(len(vehicle_pass_all)),
            "qc_ok_rows": int(vehicle_pass_all["qc_ok"].sum()),
            "qc_bad_rows": int((~vehicle_pass_all["qc_ok"]).sum()),
            "qc_ok_rate": float(vehicle_pass_all["qc_ok"].mean()) if len(vehicle_pass_all) else None,
        },
        "rules_hit_counts": {r: int(qc_dict[r].sum()) for r in QC_RULES},
        "speed_limit_used_kmh": speed_limit_value,
    }

    return vehicle_pass_all, vehicle_pass_clean, qc_report, axle_cols_std


# =========================
# diurnal：工程化固定分箱
# =========================
def _safe_quantile(s: pd.Series, q: float) -> float:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return np.nan
    return float(s2.quantile(q))


def _fixed_edges(min_val: float, max_val: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step 必须 > 0")
    # 使得 edges 覆盖 max_val
    edges = np.arange(min_val, max_val + step, step, dtype=float)
    if edges[-1] < max_val:
        edges = np.append(edges, max_val)
    return edges


def _get_engineering_edges(traf_conf) -> Tuple[np.ndarray, np.ndarray]:
    # 默认工程分箱：车重 0-200t 每 5t；车速 0-160km/h 每 10
    w_max = float(getattr(traf_conf, "weight_bin_max_t", 200.0))
    w_step = float(getattr(traf_conf, "weight_bin_width_t", 0.1))
    s_max = float(getattr(traf_conf, "speed_bin_max_kmh", 160.0))
    s_step = float(getattr(traf_conf, "speed_bin_width_kmh", 1))
    weight_edges = _fixed_edges(0.0, w_max, w_step)
    speed_edges = _fixed_edges(0.0, s_max, s_step)
    return weight_edges, speed_edges


def build_diurnal_hourly_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["hour"] = pd.to_datetime(df["pass_time"]).dt.hour
    df["overweight_i"] = df["is_overweight"].fillna(False).astype(int)
    df["overspeed_i"] = df["is_overspeed"].fillna(False).astype(int)
    df["both_i"] = (df["is_overweight"].fillna(False) & df["is_overspeed"].fillna(False)).astype(int)
    df["gross_overweight_t"] = np.where(df["is_overweight"].fillna(False), df["gross_t"], 0.0)

    g = df.groupby("hour", dropna=False)
    out = pd.DataFrame({"hour": list(range(24))}).set_index("hour")

    out["count_total"] = g.size()
    out["gross_sum_t"] = g["gross_t"].sum()
    out["gross_mean_t"] = g["gross_t"].mean()
    out["gross_median_t"] = g["gross_t"].median()
    out["gross_p95_t"] = g["gross_t"].apply(lambda s: _safe_quantile(s, 0.95))
    out["gross_p99_t"] = g["gross_t"].apply(lambda s: _safe_quantile(s, 0.99))

    out["count_overweight"] = g["overweight_i"].sum()
    out["rate_overweight"] = out["count_overweight"] / out["count_total"]
    out["overweight_weight_sum_t"] = g["gross_overweight_t"].sum()

    out["count_overspeed"] = g["overspeed_i"].sum()
    out["rate_overspeed"] = out["count_overspeed"] / out["count_total"]

    out["count_overweight_and_overspeed"] = g["both_i"].sum()
    out["rate_overweight_and_overspeed"] = out["count_overweight_and_overspeed"] / out["count_total"]

    if df["speed_kmh"].notna().any():
        out["speed_mean_kmh"] = g["speed_kmh"].mean()
        out["speed_median_kmh"] = g["speed_kmh"].median()
        out["speed_p85_kmh"] = g["speed_kmh"].apply(lambda s: _safe_quantile(s, 0.85))
        out["speed_p95_kmh"] = g["speed_kmh"].apply(lambda s: _safe_quantile(s, 0.95))
    else:
        out["speed_mean_kmh"] = np.nan
        out["speed_median_kmh"] = np.nan
        out["speed_p85_kmh"] = np.nan
        out["speed_p95_kmh"] = np.nan

    if "overspeed_excess_kmh" in df.columns and df["overspeed_excess_kmh"].notna().any():
        out["overspeed_excess_mean_kmh"] = g.apply(
            lambda x: float(pd.to_numeric(x.loc[x["is_overspeed"].fillna(False), "overspeed_excess_kmh"],
                                          errors="coerce").dropna().mean())
            if x.loc[x["is_overspeed"].fillna(False), "overspeed_excess_kmh"].notna().any() else np.nan,
            include_groups=False
        )
        out["overspeed_excess_p95_kmh"] = g.apply(
            lambda x: float(pd.to_numeric(x.loc[x["is_overspeed"].fillna(False), "overspeed_excess_kmh"],
                                          errors="coerce").dropna().quantile(0.95))
            if x.loc[x["is_overspeed"].fillna(False), "overspeed_excess_kmh"].notna().any() else np.nan,
            include_groups=False
        )
    else:
        out["overspeed_excess_mean_kmh"] = np.nan
        out["overspeed_excess_p95_kmh"] = np.nan

    out = out.reindex(range(24)).reset_index()
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def build_hour_bin_dist_table(values: pd.Series,
                              hour_series: pd.Series,
                              edges: np.ndarray,
                              prefix: str, n_time_bins: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    工程化固定分箱版本：
    - edges 由 TrafConfig 控制（或默认）
    - 输出 long 表 + pivot(density)
    """
    v = pd.to_numeric(values, errors="coerce")
    h = pd.to_numeric(hour_series, errors="coerce")

    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2

    bin_id = pd.cut(v, bins=edges, right=False, include_lowest=True, labels=False)
    tmp = pd.DataFrame({"hour": h, "bin_id": bin_id})
    tmp = tmp.dropna(subset=["hour", "bin_id"])
    tmp["hour"] = tmp["hour"].astype(int)
    tmp["bin_id"] = tmp["bin_id"].astype(int)

    counts = tmp.groupby(["hour", "bin_id"]).size().rename("count").reset_index()

    meta = pd.DataFrame({
        "bin_id": np.arange(len(centers), dtype=int),
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "bin_center": centers,
        "bin_width": widths
    })

    long_table = counts.merge(meta, on="bin_id", how="left")
    hour_total = tmp.groupby("hour").size().rename("hour_total")
    long_table = long_table.merge(hour_total, on="hour", how="left")

    long_table["prob"] = long_table["count"] / long_table["hour_total"]
    long_table["density"] = long_table["prob"] / long_table["bin_width"]

    pivot = long_table.pivot(index="hour", columns="bin_center", values="density").reindex(range(n_time_bins))

    long_table = long_table.rename(columns={
        "count": f"{prefix}_count",
        "prob": f"{prefix}_prob",
        "density": f"{prefix}_density"
    })

    return long_table, pivot


def build_joint_2d_dist_fixed(speed: pd.Series,
                              gross: pd.Series,
                              speed_edges: np.ndarray,
                              gross_edges: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.to_numeric(speed, errors="coerce")
    g = pd.to_numeric(gross, errors="coerce")
    m = s.notna() & g.notna()
    s = s[m]
    g = g[m]
    if s.empty or g.empty:
        return pd.DataFrame(), pd.DataFrame()

    H, xedges, yedges = np.histogram2d(s, g, bins=[speed_edges, gross_edges], density=True)
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2

    rows = []
    for i in range(len(xc)):
        for j in range(len(yc)):
            rows.append([xc[i], yc[j], H[i, j]])
    long = pd.DataFrame(rows, columns=["speed_bin_center", "gross_bin_center", "density"])
    pivot = long.pivot(index="gross_bin_center", columns="speed_bin_center", values="density")
    return long, pivot


# =========================
# 主函数
# =========================
def TRAF_main(numwork=4):
    file_conf = FileConfig()
    traf_conf = TrafConfig()
    out_conf = OutputConfig()
    outputconfig = out_conf.tasks

    set_plot_style()

    traffic_path = getattr(traf_conf, "traffic_path", None) or getattr(traf_config, "traffic_path", None)
    if not traffic_path:
        raise ValueError("traffic_path 未配置。")

    raw_inrange = load_traffic_files_filtered(traffic_path, traf_conf, file_conf)
    if raw_inrange.empty:
        print("时间范围内无数据，分析终止。")
        return

    qc_cfg = outputconfig['traffic']['qc']  # 目录 + 文件名都在这里
    _, data, qc_report, axle_cols_std = build_vehicle_pass_and_qc_inrange(raw_inrange, traf_conf, qc_cfg)

    if data.empty:
        print("QC 后无可用数据，分析终止。")
        return

    # =========================
    # common_analysis
    # =========================
    data["hourly_time"] = pd.to_datetime(data["pass_time"]).dt.floor("h")
    data["overweight_i"] = data["is_overweight"].fillna(False).astype(int)
    data["overspeed_i"] = data["is_overspeed"].fillna(False).astype(int)
    data["gross_overweight_t"] = np.where(data["is_overweight"].fillna(False), data["gross_t"], 0.0)

    summary = data.groupby("hourly_time").agg(
        total_weight_sum=("gross_t", "sum"),
        vehicle_count=("gross_t", "count"),
        overweight_weight_sum=("gross_overweight_t", "sum"),
        overweight_count=("overweight_i", "sum"),
        overspeed_count=("overspeed_i", "sum"),
    ).reset_index()

    if summary.empty:
        print("小时聚合结果为空，分析终止。")
        return

    formatted_date1 = summary['hourly_time'].iloc[0].strftime("%y-%m-%d")
    formatted_date2 = summary['hourly_time'].iloc[-1].strftime("%y-%m-%d")

    save_path = outputconfig['traffic']['common_analysis']['save_path']
    _ensure_parent_dir(save_path)

    df2 = summary[['hourly_time', 'total_weight_sum', 'vehicle_count']].copy()
    df2.columns = ['时间', '数据1', '数据2']

    # 按你的要求：该路径写法不改
    x_time_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'PostProcessing',  "rlt_time.csv")
    x_time = pd.read_csv(x_time_path, header=None)
    x_time.columns = ['时间']

    df1 = x_time.copy()
    df1['时间'] = pd.to_datetime(df1['时间'], errors='coerce').dt.tz_localize(None)
    df2['时间'] = pd.to_datetime(df2['时间'], errors='coerce')

    df1 = df1.dropna(subset=['时间'])
    df2 = df2.dropna(subset=['时间'])

    missing_times = df1[~df1['时间'].isin(df2['时间'])]

    mean_data1 = df2['数据1'].mean()
    mean_data2 = df2['数据2'].mean()
    mean_data1 = 0.0 if pd.isna(mean_data1) else mean_data1
    mean_data2 = 0.0 if pd.isna(mean_data2) else mean_data2

    missing_rows = missing_times.copy()
    missing_rows['数据1'] = mean_data1
    missing_rows['数据2'] = mean_data2

    df2_filled = pd.concat([df2, missing_rows], ignore_index=True).sort_values(by='时间').reset_index(drop=True)
    df2_filled.iloc[:, [1, 2]].to_csv(save_path, index=False, header=None)

    common_figs = outputconfig['traffic']['common_analysis']['figure_path']
    if not isinstance(common_figs, list) or len(common_figs) < 6:
        raise ValueError(
            "outputconfig['traffic']['common_analysis']['figure_path'] 需要至少 6 个路径："
            "[0]=weight_hour, [1]=weight_probability, [2]=count, "
            "[3]=weight_over_weight, [4]=count_over_weight, [5]=count_over_speed。"
        )

    plot_hourly_single_line(
        summary=summary, xcol="hourly_time", ycol="total_weight_sum",
        ylabel="重量 (t)",
        title=f"{formatted_date1}至{formatted_date2}日每小时车流总重量统计",
        fig_path=common_figs[0]
    )

    plot_pdf_bar_compare_dual_y(
        series_all=data["gross_t"],
        series_sub=data.loc[data["is_overweight"].fillna(False), "gross_t"],
        bins=50,
        xlabel="车重 (t)",
        title="车重概率密度分布",
        fig_path=common_figs[1],
        label_all="全部",
        label_sub="超载"
    )

    plot_hourly_single_line(
        summary=summary, xcol="hourly_time", ycol="vehicle_count",
        ylabel="车流量 (辆)",
        title=f"{formatted_date1}至{formatted_date2}日每小时车流量统计",
        fig_path=common_figs[2]
    )

    plot_hourly_single_line(
        summary=summary, xcol="hourly_time", ycol="overweight_weight_sum",
        ylabel="重量 (t)",
        title=f"{formatted_date1}至{formatted_date2}日每小时超载车流总重量统计",
        fig_path=common_figs[3]
    )

    plot_hourly_single_line(
        summary=summary, xcol="hourly_time", ycol="overweight_count",
        ylabel="车流量 (辆)",
        title=f"{formatted_date1}至{formatted_date2}日每小时超载车流量统计",
        fig_path=common_figs[4]
    )

    plot_hourly_single_line(
        summary=summary, xcol="hourly_time", ycol="overspeed_count",
        ylabel="车流量 (辆)",
        title=f"{formatted_date1}至{formatted_date2}日每小时超速车流量统计",
        fig_path=common_figs[5]
    )

    # =========================
    # lane 分析
    # =========================
    if ('lane' in outputconfig['traffic']) and (not data["lane"].dropna().empty):
        lane_figs = outputconfig['traffic']['lane']['figure_path']
        if not isinstance(lane_figs, list) or len(lane_figs) < 4:
            raise ValueError(
                "outputconfig['traffic']['lane']['figure_path'] 需要至少 4 个路径："
                "[0]=lane_prob, [1]=lane_count, [2]=lane_count_over_weight, [3]=lane_count_over_speed。"
            )

        lane_summary = data.groupby("lane").agg(
            vehicle_count=("gross_t", "count"),
            overweight_count=("overweight_i", "sum"),
            overspeed_count=("overspeed_i", "sum"),
        ).reset_index()

        plot_lane_weight_pdfs(data=data, lane_col="lane", weight_col="gross_t", fig_path=lane_figs[0], bins=50)
        plot_lane_single_count(lane_summary, "lane", "vehicle_count", "各车道车辆数量统计", lane_figs[1])
        plot_lane_single_count(lane_summary, "lane", "overweight_count", "各车道超载车辆数量统计", lane_figs[2])
        plot_lane_single_count(lane_summary, "lane", "overspeed_count", "各车道超速车辆数量统计", lane_figs[3])
    else:
        print("lane_num 未配置或 lane 数据缺失，跳过按车道分析。")

    # =========================
    # speed PDF（双 y 轴 + 限速线）
    # =========================
    if ('speed' in outputconfig['traffic']) and (not data["speed_kmh"].dropna().empty):
        fig_speed = outputconfig['traffic']['speed']['figure_path']
        limit_list = _idx_list(traf_conf, "over_speed_limit")
        limit_val = float(limit_list[0]) if limit_list else None

        plot_speed_pdf_with_limit_dual_y(
            speed_series=data["speed_kmh"],
            is_overspeed=data["is_overspeed"],
            speed_limit=limit_val,
            fig_path=fig_speed
        )
    else:
        print("speed 未配置或 speed 数据缺失，跳过车速分析。")

    # =========================
    # axle（保持你原输出逻辑）
    # =========================
    if ('axle' in outputconfig['traffic']) and axle_cols_std:
        axle_mat = data[axle_cols_std].apply(pd.to_numeric, errors="coerce").fillna(0)
        axle_count = (axle_mat > 0).sum(axis=1)

        axle_count_summary = axle_count.value_counts().sort_index()
        if 1 in axle_count_summary.index:
            axle_count_summary = axle_count_summary.drop(1)

        if axle_count_summary.empty:
            print("轴数统计为空，跳过轴重分布绘图。")
        else:
            nplots = len(axle_count_summary.index)
            ncols = 2 if nplots > 1 else 1
            nrows = int(np.ceil(nplots / ncols))
            width_cm = 14 if ncols == 2 else 7
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(_cm_to_inch(width_cm), _cm_to_inch(4 * nrows)))
            axes = np.array(axes).reshape(-1)
            fig_index = 0
            for ac in axle_count_summary.index:
                idxs = axle_count == ac
                total_axle_weight = axle_mat.loc[idxs, axle_cols_std].sum(axis=1)
                total_axle_weight = total_axle_weight[total_axle_weight > 0]
                if total_axle_weight.empty:
                    continue

                hv, be = np.histogram(total_axle_weight, bins=50, density=True)
                bc = (be[:-1] + be[1:]) / 2
                ax = axes[fig_index]
                fig_index += 1
                ax.bar(bc, hv, width=(be[1] - be[0]), alpha=0.6, edgecolor='black', linewidth=1.5)
                ax.set_xlabel('总重量 (t)', fontsize=12)
                ax.set_ylabel('概率密度', fontsize=12)
                ax.set_title(f"{ac} 轴车车重概率密度分布", fontsize=12)
                ax.tick_params(axis='y', labelsize=10)
                ax.tick_params(axis='x', labelsize=10)
            # 关闭未使用的子图（当轴数类别为奇数时会出现空位）
            for ax in axes[fig_index:]:
                ax.axis('off')

            fig_axle = outputconfig['traffic']['axle']['figure_path']
            save_close(fig, fig_axle)
    else:
        print("axle_weight 未配置或无有效轴重列，跳过轴重分析。")

    # =========================
    # diurnal（日内 24h）
    # =========================
    if 'diurnal' not in outputconfig['traffic']:
        raise ValueError("OutputConfig 中未配置 traffic.diurnal 任务。")

    diurnal_cfg = outputconfig['traffic']['diurnal']
    diu_save = diurnal_cfg['save_path']
    diu_figs = diurnal_cfg['figure_path']

    if (not isinstance(diu_save, list)) or len(diu_save) < 6:
        raise ValueError("traffic.diurnal.save_path 需要至少 6 个路径。")
    if (not isinstance(diu_figs, list)) or len(diu_figs) < 11:
        raise ValueError("traffic.diurnal.figure_path 需要至少 11 个路径。")

    data["hour"] = pd.to_datetime(data["pass_time"]).dt.hour
    # 时间分箱（仅用于热力图更密的时间轴）
    bin_minutes = 30  # 想更密就改成 15；想更粗就改成 60
    dt = pd.to_datetime(data["pass_time"])
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    data["time_bin"] = (minute_of_day // bin_minutes).astype(int)

    # 一天有多少个 time_bin
    n_time_bins = int(np.ceil(1440 / bin_minutes))

    # 工程化固定分箱 edges
    weight_edges, speed_edges = _get_engineering_edges(traf_conf)

    # 仅用于绘图：按 95% 分位数裁剪显示范围（不影响CSV统计）
    plot_q_w = 0.9973
    # 默认 step 取 edges 的步长（若异常则回退到 1）
    w_step = float(weight_edges[1] - weight_edges[0]) if len(weight_edges) > 1 else 1.0
    s_step = float(speed_edges[1] - speed_edges[0]) if len(speed_edges) > 1 else 1.0

    w_p95 = _safe_quantile(data["gross_t"], plot_q_w)
    w_max_plot = float(weight_edges[-1])
    if np.isfinite(w_p95) and (w_p95 > 0) and (w_step > 0):
        w_max_plot = float(min(weight_edges[-1], np.ceil(w_p95 / w_step) * w_step))

    plot_q_s = 0.9999
    s_p95 = _safe_quantile(data["speed_kmh"], plot_q_s) if ("speed_kmh" in data.columns) else np.nan
    s_max_plot = float(speed_edges[-1])
    if np.isfinite(s_p95) and (s_p95 > 0) and (s_step > 0):
        s_max_plot = float(min(speed_edges[-1], np.ceil(s_p95 / s_step) * s_step))

    diurnal_hourly = build_diurnal_hourly_features(data)
    _ensure_parent_dir(diu_save[0])
    diurnal_hourly.to_csv(diu_save[0], index=False, encoding="utf-8-sig")

    # weight dist + heatmap（x 轴 1 位小数，无旋转）
    weight_dist_long, weight_pivot = build_hour_bin_dist_table(
        values=data["gross_t"],
        hour_series=data["time_bin"],
        edges=weight_edges,
        prefix="gross",
        n_time_bins = n_time_bins
    )
    _ensure_parent_dir(diu_save[1])
    weight_dist_long.to_csv(diu_save[1], index=False, encoding="utf-8-sig")

    # 仅绘图：裁剪到 95% 分位数范围内（例如约 80t 内）
    weight_pivot_plot = weight_pivot
    try:
        weight_pivot_plot = weight_pivot.loc[:, weight_pivot.columns <= w_max_plot]
        if weight_pivot_plot.empty:
            weight_pivot_plot = weight_pivot
    except Exception:
        weight_pivot_plot = weight_pivot

    plot_heatmap_hour_bin(
        pivot=weight_pivot_plot,
        title="日内分时段车重分布热力图（density）",
        xlabel="车重 bin_center (t)",
        ylabel="小时",
        fig_path=diu_figs[8],
        tick_decimals=1
    )


    # speed dist + heatmap（同样工程化 bins）
    if data["speed_kmh"].notna().any():
        speed_dist_long, speed_pivot = build_hour_bin_dist_table(
            values=data["speed_kmh"],
            hour_series=data["time_bin"],
            n_time_bins=n_time_bins,
            edges=speed_edges,
            prefix="speed"
        )
        _ensure_parent_dir(diu_save[2])
        speed_dist_long.to_csv(diu_save[2], index=False, encoding="utf-8-sig")

        # 仅绘图：裁剪到 95% 分位数范围内（例如约 100km/h 内）
        speed_pivot_plot = speed_pivot
        try:
            speed_pivot_plot = speed_pivot.loc[:, speed_pivot.columns <= s_max_plot]
            if speed_pivot_plot.empty:
                speed_pivot_plot = speed_pivot
        except Exception:
            speed_pivot_plot = speed_pivot

        plot_heatmap_hour_bin(
            pivot=speed_pivot_plot,
            title="日内分时段车速分布热力图（density）",
            xlabel="车速 bin_center (km/h)",
            ylabel="小时",
            fig_path=diu_figs[9],
            tick_decimals=1
        )

    else:
        _ensure_parent_dir(diu_save[2])
        pd.DataFrame(columns=["hour", "bin_id", "bin_left", "bin_right", "bin_center",
                              "bin_width", "hour_total", "speed_count", "speed_prob", "speed_density"]) \
            .to_csv(diu_save[2], index=False, encoding="utf-8-sig")
        plot_placeholder(diu_figs[9], "日内分时段车速分布热力图（density）", "无车速数据或未配置 speed 列")

    # lane×hour 表
    if data["lane"].notna().any():
        tmp = data.copy()
        tmp["overweight_i"] = tmp["is_overweight"].fillna(False).astype(int)
        tmp["overspeed_i"] = tmp["is_overspeed"].fillna(False).astype(int)
        g = tmp.groupby(["lane", "hour"])
        lane_hour = g.agg(
            count_total=("event_id", "count"),
            gross_sum_t=("gross_t", "sum"),
            gross_p95_t=("gross_t", lambda s: _safe_quantile(s, 0.95)),
            overweight_count=("overweight_i", "sum"),
            overspeed_count=("overspeed_i", "sum"),
        ).reset_index()
        lane_hour["rate_overweight"] = lane_hour["overweight_count"] / lane_hour["count_total"]
        lane_hour["rate_overspeed"] = lane_hour["overspeed_count"] / lane_hour["count_total"]
        _ensure_parent_dir(diu_save[3])
        lane_hour.to_csv(diu_save[3], index=False, encoding="utf-8-sig")
    else:
        _ensure_parent_dir(diu_save[3])
        pd.DataFrame(columns=["lane", "hour", "count_total", "gross_sum_t", "gross_p95_t",
                              "overweight_count", "overspeed_count", "rate_overweight", "rate_overspeed"]) \
            .to_csv(diu_save[3], index=False, encoding="utf-8-sig")

    # Top-N 极端事件
    top_n = int(getattr(traf_conf, "top_events_n", 200))
    cols = ["event_id", "pass_time", "lane", "gross_t", "speed_kmh",
            "is_overweight", "is_overspeed", "overspeed_excess_kmh",
            "axle_count_pos", "axle_sum_t", "source_file", "source_row"]
    keep_cols = [c for c in cols if c in data.columns]
    top_events = data.sort_values("gross_t", ascending=False).head(top_n)[keep_cols].copy()
    _ensure_parent_dir(diu_save[4])
    top_events.to_csv(diu_save[4], index=False, encoding="utf-8-sig")

    # speed×weight 联合分布：工程化 bins + 轴标签 1 位小数无旋转
    if data["speed_kmh"].notna().any():
        joint_long, joint_pivot = build_joint_2d_dist_fixed(
            speed=data["speed_kmh"],
            gross=data["gross_t"],
            speed_edges=speed_edges,
            gross_edges=weight_edges
        )
        _ensure_parent_dir(diu_save[5])
        joint_long.to_csv(diu_save[5], index=False, encoding="utf-8-sig")

        if not joint_pivot.empty:
            # 仅绘图：按 95% 分位数裁剪显示范围（不影响联合分布CSV）
            joint_pivot_plot = joint_pivot
            try:
                joint_pivot_plot = joint_pivot_plot.loc[joint_pivot_plot.index <= w_max_plot, :]
                joint_pivot_plot = joint_pivot_plot.loc[:, joint_pivot_plot.columns <= s_max_plot]
                if joint_pivot_plot.empty:
                    joint_pivot_plot = joint_pivot
            except Exception:
                joint_pivot_plot = joint_pivot

            fig = plt.figure(figsize=(_cm_to_inch(14), _cm_to_inch(8)))
            mat = np.nan_to_num(joint_pivot_plot.to_numpy(), nan=0.0)
            mat_plot = mat.copy()
            pos = mat_plot[mat_plot > 0]
            if pos.size > 0:
                vmin = pos.min()
                mat_plot[mat_plot <= 0] = vmin
                norm = LogNorm(vmin=vmin, vmax=mat_plot.max())
            else:
                norm = None

            plt.imshow(mat_plot, norm=norm, aspect="auto", origin="lower")
            plt.title("车速-车重联合分布热力图（density）", fontsize=12)
            plt.xlabel("车速 bin_center (km/h)", fontsize=12)
            plt.ylabel("车重 bin_center (t)", fontsize=12)

            ncol = joint_pivot_plot.shape[1]
            nrow = joint_pivot_plot.shape[0]
            if ncol > 0:
                xt = np.linspace(0, ncol - 1, num=min(10, ncol), dtype=int)
                plt.xticks(xt, [f"{float(joint_pivot_plot.columns[i]):.1f}" for i in xt], fontsize=9, rotation=0)
            if nrow > 0:
                yt = np.linspace(0, nrow - 1, num=min(10, nrow), dtype=int)
                plt.yticks(yt, [f"{float(joint_pivot_plot.index[i]):.1f}" for i in yt], fontsize=9)
            cb=plt.colorbar()
            import matplotlib.ticker as mticker

            cb.formatter = mticker.FuncFormatter(lambda x, pos: f"{x:.0e}")  # 例如 1e-3（这里的 '-' 是 ASCII）
            cb.update_ticks()

            save_close(fig, diu_figs[10])
        else:
            plot_placeholder(diu_figs[10], "车速-车重联合分布热力图（density）", "联合分布为空")
    else:
        _ensure_parent_dir(diu_save[5])
        pd.DataFrame(columns=["speed_bin_center", "gross_bin_center", "density"]).to_csv(diu_save[5], index=False, encoding="utf-8-sig")
        plot_placeholder(diu_figs[10], "车速-车重联合分布热力图（density）", "无车速数据或未配置 speed 列")

    # diurnal 线图/箱线（保持你之前的逻辑）
    plot_diurnal_line(diurnal_hourly, "hour", "count_total", "车流量 (辆)", "日内 24h 车流量", diu_figs[0])
    plot_diurnal_line(diurnal_hourly, "hour", "gross_sum_t", "总重量 (t)", "日内 24h 总重量", diu_figs[1])
    plot_diurnal_line(diurnal_hourly, "hour", "rate_overweight", "超载率", "日内 24h 超载率", diu_figs[2])

    if data["speed_kmh"].notna().any():
        plot_diurnal_line(diurnal_hourly, "hour", "rate_overspeed", "超速率", "日内 24h 超速率", diu_figs[3])
        plot_diurnal_two_lines(diurnal_hourly, "hour", "speed_p85_kmh", "speed_p95_kmh",
                               "P85", "P95", "车速分位数 (km/h)", "日内 24h 车速尾部（P85/P95）", diu_figs[5])
        plot_diurnal_box_by_hour(data["speed_kmh"], data["hour"], "日内 24h 车速箱线图", "车速 (km/h)", diu_figs[7])
    else:
        plot_placeholder(diu_figs[3], "日内 24h 超速率", "无车速数据或未配置 speed 列")
        plot_placeholder(diu_figs[5], "日内 24h 车速尾部（P85/P95）", "无车速数据或未配置 speed 列")
        plot_placeholder(diu_figs[7], "日内 24h 车速箱线图", "无车速数据或未配置 speed 列")

    plot_diurnal_two_lines(diurnal_hourly, "hour", "gross_p95_t", "gross_p99_t",
                           "P95", "P99", "车重分位数 (t)", "日内 24h 车重尾部（P95/P99）", diu_figs[4])
    plot_diurnal_box_by_hour(data["gross_t"], data["hour"], "日内 24h 车重箱线图", "车重 (t)", diu_figs[6])

    print('交通荷载分析完成!')


if __name__ == "__main__":
    TRAF_main()
