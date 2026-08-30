# 文件: utils/plot_results.py
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator  # 用于强制 X 轴显示整数

def set_academic_style():
    """设置学术论文风格的绘图格式"""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.markersize": 7,
        "figure.dpi": 300,  # 高分辨率
    })

# ==========================================================
# 学术化调色盘 (引入层级和高对比度)
# ==========================================================
STRATEGY_STYLES = {
    'Random_Taylor': {'color': '#377eb8', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.5, 'zorder': 8, 'label': 'Ours-Taylor'},
    'Random_FIM': {'color': '#4daf4a', 'marker': 's', 'linestyle': '-', 'linewidth': 2.5, 'zorder': 8, 'label': 'Ours-FIM'},
    'Random_FIM_NoDepth': {'color': '#8dd3c7', 'marker': 's', 'linestyle': '--', 'linewidth': 1.8, 'zorder': 5, 'label': 'Ours-FIM w/o Depth'},
    'Random_Taylor_NoDepth': {'color': '#80b1d3', 'marker': 'o', 'linestyle': '--', 'linewidth': 1.8, 'zorder': 5, 'label': 'Ours-Taylor w/o Depth'},
    'Random_Scores': {'color': '#b3de69', 'marker': 'p', 'linestyle': '--', 'linewidth': 1.8, 'zorder': 4, 'label': 'Random Scores'},
    'FedAvg_LoRA_homogeneous': {'color': '#e41a1c', 'marker': '*', 'linestyle': '-.', 'linewidth': 2.0, 'zorder': 5, 'label': 'FedAvg-LoRA'},
    'Uniform_allocation': {'color': '#17becf', 'marker': '.', 'linestyle': '-.', 'linewidth': 1.8, 'zorder': 4, 'label': 'Uniform'},
    'Fed-HeLLo': {'color': '#ff7f00', 'marker': '^', 'linestyle': '--', 'linewidth': 2.0, 'zorder': 6, 'label': 'Fed-HeLLo (Layer-wise)'},
    'LEGEND': {'color': '#984ea3', 'marker': 'v', 'linestyle': '--', 'linewidth': 2.0, 'zorder': 6, 'label': 'LEGEND (Hetero-Rank)'},
    'HETLoRA': {'color': '#6a3d9a', 'marker': 'P', 'linestyle': '--', 'linewidth': 2.0, 'zorder': 6, 'label': 'HETLoRA-style (Hetero-Rank)'},
    'FAH_QLoRA': {'color': '#b15928', 'marker': 'X', 'linestyle': '--', 'linewidth': 2.0, 'zorder': 6, 'label': 'FAH-QLoRA-style'},
    'Physical_FIM': {'color': '#fb8072', 'marker': '^', 'linestyle': ':', 'linewidth': 1.8, 'zorder': 4, 'label': 'Physical-FIM'},
    'Physical_Taylor': {'color': '#bebada', 'marker': '^', 'linestyle': ':', 'linewidth': 1.8, 'zorder': 4, 'label': 'Physical-Taylor'},
    'Greedy_Value': {'color': '#bc80bd', 'marker': 'P', 'linestyle': ':', 'linewidth': 1.8, 'zorder': 4, 'label': 'Greedy-Value'},
    'Greedy_ValueCost': {'color': '#ccebc5', 'marker': 'P', 'linestyle': ':', 'linewidth': 1.8, 'zorder': 4, 'label': 'Greedy-Value/Cost'},
    'Greedy_LatencyAware': {'color': '#ffed6f', 'marker': 'P', 'linestyle': ':', 'linewidth': 1.8, 'zorder': 4, 'label': 'Greedy-Latency'},
    'FedRA': {'color': '#a65628', 'marker': 'D', 'linestyle': ':', 'linewidth': 1.5, 'zorder': 4, 'label': 'FedRA'},
    'MS': {'color': '#f781bf', 'marker': 'x', 'linestyle': ':', 'linewidth': 1.5, 'zorder': 4, 'label': 'MS'},
    'MH': {'color': '#999999', 'marker': '+', 'linestyle': ':', 'linewidth': 1.5, 'zorder': 4, 'label': 'MH'},
    'EL': {'color': '#17becf', 'marker': '.', 'linestyle': '-.', 'linewidth': 1.5, 'zorder': 3, 'label': 'EL'},
    'Random': {'color': '#bcbd22', 'marker': '.', 'linestyle': '-.', 'linewidth': 1.5, 'zorder': 3, 'label': 'Random'}
}

def normalize_metric_columns(df):
    """Normalize historical CSV column spelling/whitespace differences."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {}
    for col in df.columns:
        normalized = col.strip().lower().replace("-", "_")
        if normalized in {"rouge_l", "rougel", "rouge_l_score"}:
            rename_map[col] = "Rouge_l"
        elif normalized == "accuracy":
            rename_map[col] = "Accuracy"
        elif normalized == "loss":
            rename_map[col] = "Loss"
        elif normalized in {"raw_lm_loss", "lm_loss", "raw_loss"}:
            rename_map[col] = "Raw_LM_Loss"

    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def recover_rouge_from_tensorboard(dataset_name, run_name, strategy_name, df):
    """Best-effort recovery for CSV files produced before Rouge_l was exported."""
    if "Rouge_l" in df.columns and df["Rouge_l"].fillna(0).sum() > 0:
        return df

    tb_dir = os.path.join(f"eval_tensorboard_{dataset_name}", run_name, strategy_name)
    if not os.path.isdir(tb_dir):
        return df

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return df

    try:
        accumulator = EventAccumulator(tb_dir)
        accumulator.Reload()
        if "Main/Rouge-L" not in accumulator.Tags().get("scalars", []):
            return df

        rouge_by_round = {
            int(event.step): float(event.value)
            for event in accumulator.Scalars("Main/Rouge-L")
        }
        if not rouge_by_round or "Round" not in df.columns:
            return df

        recovered = df["Round"].map(lambda step: rouge_by_round.get(int(step), None))
        if recovered.notna().any():
            df = df.copy()
            df["Rouge_l"] = recovered.ffill()
            print(f"[提示] 已从 TensorBoard 为 {strategy_name} 恢复 Rouge_l。")
    except Exception as exc:
        print(f"[警告] 从 TensorBoard 恢复 {strategy_name} 的 Rouge_l 失败: {exc}")

    return df

def recover_rouge_from_logs(run_name, strategy_name, df):
    """Recover old SAMSum Rouge values from Slurm stdout when CSV missed Rouge_l."""
    if "Rouge_l" in df.columns and df["Rouge_l"].fillna(0).sum() > 0:
        return df
    if "Round" not in df.columns:
        return df

    pattern = re.compile(
        rf"\[{re.escape(strategy_name)}\]\s+Round\s+(\d+):.*?rouge-l=([0-9.]+)"
    )
    recovered = {}

    for log_path in glob.glob(os.path.join("logs", "*.out")):
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
                content = handle.read()
        except OSError:
            continue

        if run_name not in content:
            continue

        for round_id, rouge_l in pattern.findall(content):
            recovered[int(round_id)] = float(rouge_l)

    if not recovered:
        return df

    rouge_series = df["Round"].map(lambda step: recovered.get(int(step), None))
    if rouge_series.notna().any():
        df = df.copy()
        df["Rouge_l"] = rouge_series.ffill()
        print(f"[提示] 已从 Slurm 日志为 {strategy_name} 恢复 Rouge_l。")

    return df

def load_latest_results(dataset_name="cifar100", run_name=None):
    base_dir = f"evaluation_results/{dataset_name}"
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"找不到数据目录: {base_dir}")

    if run_name:
        latest_dir = os.path.join(base_dir, run_name)
        if not os.path.isdir(latest_dir):
            raise FileNotFoundError(f"找不到指定评估目录: {latest_dir}")
    else:
        subdirs =[os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not subdirs:
            raise FileNotFoundError(f"目录 {base_dir} 下没有评估数据。")
        latest_dir = max(subdirs, key=os.path.getmtime)

    print(f"--- [绘图模块] 正在读取最新数据目录: {latest_dir} ---")
    run_name = os.path.basename(latest_dir)

    data_dict = {}
    csv_files = glob.glob(os.path.join(latest_dir, "eval_*.csv"))

    for file in csv_files:
        strategy_name = os.path.basename(file).replace("eval_", "").replace(".csv", "")
        df = pd.read_csv(file)
        df = normalize_metric_columns(df)
        df = recover_rouge_from_tensorboard(dataset_name, run_name, strategy_name, df)
        df = recover_rouge_from_logs(run_name, strategy_name, df)
        if dataset_name == "samsum" and "Rouge_l" in df.columns:
            loss_values = pd.to_numeric(df["Loss"], errors="coerce") if "Loss" in df.columns else pd.Series(dtype=float)
            has_finite_loss = np.isfinite(loss_values.dropna()).any()
            if not has_finite_loss:
                df["Loss"] = 1.0 - pd.to_numeric(df["Rouge_l"], errors="coerce").clip(lower=0.0, upper=1.0)
                print(f"[提示] 已为 {strategy_name} 使用 1 - Rouge_l 生成 SAMSum Loss proxy。")
        data_dict[strategy_name] = df

    return data_dict, latest_dir

def plot_combined_metrics(data_dict, x_col, metric1_col, metric2_col, xlabel, ylabel1, ylabel2, main_title, save_path):
    """
    [核心重构] 将 Loss 和 Accuracy 画在同一张图的左右两侧 (1x2 Subplots)
    并提取全局共享图例放在最右侧。
    """
    # 创建 1行2列 的画板，宽度拉长以容纳两个图和外侧图例
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 用于收集全局图例的句柄和标签
    handles, labels = [],[]

    for strategy, df in data_dict.items():
        if x_col not in df.columns: continue

        style = STRATEGY_STYLES.get(strategy, {'color': 'black', 'marker': 'o', 'linestyle': '-', 'linewidth': 1.5, 'zorder': 1, 'label': strategy})
        markevery = max(1, len(df) // 10)

        # --- 绘制左图 (Loss) ---
        if metric1_col in df.columns:
            line, = ax1.plot(df[x_col], df[metric1_col],
                     color=style['color'], marker=style['marker'],
                     linestyle=style['linestyle'], linewidth=style['linewidth'],
                     zorder=style['zorder'], markevery=markevery)

            # 收集图例 (只收集一次)
            if style['label'] not in labels:
                handles.append(line)
                labels.append(style['label'])

        # --- 绘制右图 (Accuracy / Rouge) ---
        if metric2_col in df.columns:
            ax2.plot(df[x_col], df[metric2_col],
                     color=style['color'], marker=style['marker'],
                     linestyle=style['linestyle'], linewidth=style['linewidth'],
                     zorder=style['zorder'], markevery=markevery)

    # --- 左图格式化 ---
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    ax1.set_title(f"{ylabel1} vs. {xlabel.split('(')[0].strip()}", pad=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    if x_col == 'Round':
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- 右图格式化 ---
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel2)
    ax2.set_title(f"{ylabel2} vs. {xlabel.split('(')[0].strip()}", pad=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    if x_col == 'Round':
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- 整体大标题 ---
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=1.05)

    # --- 添加全局共享图例 (放在整张图的最右侧外边) ---
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.18, 0.5), frameon=True, shadow=True)

    # 自动调整子图间距
    plt.tight_layout()

    # 保存图片 (bbox_inches='tight' 确保外侧图例不被裁掉)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  -> 已保存合并图表: {save_path}")

def plot_latency_bar_chart(data_dict, title, save_path):
    """延迟柱状图保持独立"""
    plt.figure(figsize=(9, 5))
    strategies, avg_latencies, colors = [], [],[]

    for strategy, df in data_dict.items():
        if 'T_Actual' in df.columns:
            style = STRATEGY_STYLES.get(strategy, {})
            strategies.append(style.get('label', strategy))
            avg_latencies.append(df['T_Actual'].mean())
            colors.append(style.get('color', 'grey'))

    if not strategies: return

    bars = plt.bar(strategies, avg_latencies, color=colors, alpha=0.8, edgecolor='black', zorder=3)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel('Strategies')
    plt.ylabel('Average Latency per Round (s)')
    plt.title(title, pad=15)
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  -> 已保存柱状图: {save_path}")

def generate_all_plots(dataset_name, distribution="Unknown", run_name=None):
    set_academic_style()
    try:
        data_dict, save_dir = load_latest_results(dataset_name, run_name=run_name)
    except Exception as e:
        print(f"绘图失败: {e}")
        return

    print("---[绘图模块] 开始生成合并双子图 ---")

    all_columns = set()
    for df in data_dict.values():
        all_columns.update(df.columns)

    # 动态识别第二个指标是 Accuracy 还是 Rouge_l
    has_accuracy = (
        'Accuracy' in all_columns
        and any('Accuracy' in df.columns and df['Accuracy'].fillna(0).sum() > 0 for df in data_dict.values())
    )
    has_rouge = (
        'Rouge_l' in all_columns
        and any('Rouge_l' in df.columns and df['Rouge_l'].fillna(0).sum() > 0 for df in data_dict.values())
    )

    metric2_col = None
    ylabel2 = ""
    if has_accuracy:
        metric2_col = 'Accuracy'
        ylabel2 = 'Validation Accuracy'
    elif has_rouge:
        metric2_col = 'Rouge_l'
        ylabel2 = 'Rouge-L Score'

    metric1_col = 'Loss'
    ylabel1 = 'Performance Proxy (1 - Rouge-L)' if dataset_name == "samsum" else 'Validation Loss'

    task_info = f"[Task: {dataset_name.upper()} | Data Dist: {distribution}]"

    # 如果同时存在 Loss 和 (Accuracy/Rouge)，则画双子图
    if metric2_col:
        # 1. 轮次合并图
        plot_combined_metrics(data_dict, x_col='Round', metric1_col=metric1_col, metric2_col=metric2_col,
                              xlabel='Communication Round', ylabel1=ylabel1, ylabel2=ylabel2,
                              main_title=f'Performance vs. Round\n{task_info}',
                              save_path=os.path.join(save_dir, "plot_combined_vs_round.png"))

        # 2. 累计时间合并图
        plot_combined_metrics(data_dict, x_col='Cumulative_Time', metric1_col=metric1_col, metric2_col=metric2_col,
                              xlabel='Cumulative Wall-clock Time (s)', ylabel1=ylabel1, ylabel2=ylabel2,
                              main_title=f'Performance vs. Cumulative Time\n{task_info}',
                              save_path=os.path.join(save_dir, "plot_combined_vs_time.png"))

        # 3. 累计预算合并图
        plot_combined_metrics(data_dict, x_col='Cumulative_Budget', metric1_col=metric1_col, metric2_col=metric2_col,
                              xlabel='Cumulative Budget Consumed', ylabel1=ylabel1, ylabel2=ylabel2,
                              main_title=f'Performance vs. Cumulative Budget\n{task_info}',
                              save_path=os.path.join(save_dir, "plot_combined_vs_budget.png"))
    else:
        print("[警告] 未检测到 Accuracy 或 Rouge_l 数据，无法生成双子图。")

    # 4. 延迟柱状图 (依然保持单图)
    plot_latency_bar_chart(data_dict,
                           title=f'Avg. Comm. Round Latency\n{task_info}',
                           save_path=os.path.join(save_dir, "plot_avg_latency.png"))

    print(f"--- [绘图模块] 所有的超宽合并高清图表生成完毕，保存在: {save_dir} ---")

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cifar100"
    dist = sys.argv[2] if len(sys.argv) > 2 else "Unknown"
    run_name = sys.argv[3] if len(sys.argv) > 3 else None
    generate_all_plots(dataset, dist, run_name=run_name)
