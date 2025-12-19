import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

from sklearn.preprocessing import MinMaxScaler

from config import OutputConfig, Ass1Config

# =========================
# 可选导入 torch：没有 torch 也要能跑
# =========================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False

# =========================
# 读取 fileconfig 的 GPU 开关
# =========================
try:
    import fileconfig
except Exception:
    fileconfig = None

def get_use_gpu_from_fileconfig(default=True) -> bool:
    """
    从 fileconfig 中获取是否允许使用 GPU 的开关。
    支持字段名：USE_GPU / use_gpu / USE_CUDA
    """
    if fileconfig is None:
        return default
    for key in ("USE_GPU", "use_gpu", "USE_CUDA"):
        if hasattr(fileconfig, key):
            return bool(getattr(fileconfig, key))
    return default

USE_GPU_SWITCH = get_use_gpu_from_fileconfig(default=True)

if TORCH_AVAILABLE:
    device = torch.device("cuda" if (USE_GPU_SWITCH and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

# =========================
# plot global
# =========================
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# config load
# =========================
ass1config = Ass1Config()
OutConfig = OutputConfig()
outputconfig = OutConfig.tasks

# =========================
# meta dicts
# =========================
file_infor = {
    'displacement': 'dis',
    'temperature': 'tmp',
    'strain': 'str',
    'vibration': 'acc',
    'cable_vib': 'vic',
    'wind_speed': 'wsd',
    'wind_direction': 'wdr',
    'inclination': 'ian',
    'settlement': 'set',
    'GPS': 'gps',
    'cable_force': 'cbf',
    'traffic': 'traf',
}
task_infor = {
    'mean': '均值',
    'rms': '均方根',
    'OMA1': '频率',
    'common_analysis': ''
}
label_infor = {
    'displacement': ['位移', 'mm'],
    'temperature': ['温度', '℃'],
    'strain': ['应变', '微应变'],
    'vibration': ['主梁加速度', 'mg'],
    'cable_vib': ['拉索加速度', 'mg'],
    'wind_speed': ['风速', 'm/s'],
    'wind_direction': ['风向', '°'],
    'inclination': ['倾角', '°'],
    'settlement': ['沉降', 'mm'],
    'GPS': ['GPS', 'mm'],
    'cable_force': ['索力', 'kN'],
    'traffic': ['交通荷载', ' '],
}

# =========================
# paths / utils
# =========================
upper_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(upper_dir, 'PostProcessing', 'assessment1', 'rlt_figure'), exist_ok=True)
os.makedirs(os.path.join(upper_dir, 'PostProcessing', 'assessment1', 'ass1_model'), exist_ok=True)

def ensure_dir_for_file(file_path: str):
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

def get_data_path(folder_type: str, type_code: str, task_name: str) -> str:
    """
    upper/PostProcessing/<folder_type>/rlt_table/rlt_<type_code>_<task_name>.csv
    """
    upper_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_name = f"rlt_{type_code}_{task_name}.csv"
    path = os.path.join(upper_path, 'PostProcessing', folder_type, "rlt_table")
    return os.path.join(path, file_name)

def load_time_series() -> pd.Series:
    x_time_path = os.path.join(upper_dir, 'PostProcessing', "rlt_time.csv")
    x_time = pd.read_csv(x_time_path, header=None)
    x_time[0] = pd.to_datetime(x_time[0], errors='coerce')
    return x_time[0]

def create_sequences(data: np.ndarray, seq_length: int):
    """
    data: shape (N, 1)
    return:
      X: (N-seq, seq, 1)
      y: (N-seq, 1)
    """
    Xs, ys = [], []
    for i in range(len(data) - seq_length):
        Xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.asarray(Xs), np.asarray(ys)

def plot_train_test(
    fig_path: str,
    x_time: pd.Series,
    y_all: np.ndarray,          # 与 x_time 等长
    train_split_idx: int,       # 训练段结束
    y_pred_test: np.ndarray,    # 测试段预测
    y_lower_test: np.ndarray,
    y_upper_test: np.ndarray,
    title: str,
    ylabel: str,
    method: str
):
    ensure_dir_for_file(fig_path)

    plt.rcParams['font.size'] = 12
    fig_width = 14 * 0.393701
    fig_height = 8 * 0.393701
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.plot(x_time.iloc[:train_split_idx], y_all[:train_split_idx],
            label="训练数据", linewidth=0.5, alpha=0.5)
    ax.plot(x_time.iloc[train_split_idx:], y_all[train_split_idx:],
            label="真实值", linewidth=0.5, alpha=0.5)
    ax.plot(x_time.iloc[train_split_idx:], y_pred_test,
            label="预测值", linestyle='dashed', linewidth=0.5, alpha=0.5)

    ax.fill_between(x_time.iloc[train_split_idx:], y_lower_test, y_upper_test,
                    alpha=0.3, label="置信区间")

    num_ticks = 9
    tick_locs = np.linspace(0, len(x_time) - 1, num_ticks, dtype=int)
    ax.set_xticks(x_time.iloc[tick_locs])
    ax.set_xticklabels([pd.to_datetime(x_time.iloc[i]).strftime('%Y-%m-%d') for i in tick_locs],
                       rotation=20, fontsize=10, fontfamily='Times New Roman')

    ax.set_xlabel("时间 (年-月-日）")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}(方法:{method})")

    fig.subplots_adjust(bottom=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.12),
              bbox_transform=fig.transFigure, ncol=4, fontsize=10)

    plt.savefig(fig_path, format='png', dpi=300)
    plt.close(fig)

# =========================
# LSTM (仅 torch 可用时启用)
# =========================
def torch_load_compat(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # 不再随机初始化 h0/c0，避免每次 forward 输出漂移
            out, _ = self.lstm(x)
            return self.linear(out[:, -1, :])

    def train_lstm_and_save_univariate(
        y_data: np.ndarray,        # shape (N,)
        seq_length: int,
        train_size_raw: int,
        model_path: str,
        lr: float = 0.01,
        max_epochs: int = 300,
    ):
        ensure_dir_for_file(model_path)

        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y_data.reshape(-1, 1))

        X, y = create_sequences(y_scaled, seq_length)
        if len(X) < 20:
            raise ValueError("序列太短，无法训练 LSTM。")

        train_size_seq = min(train_size_raw, len(X))
        X_train, y_train = X[:train_size_seq], y[:train_size_seq]

        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)

        model = LSTMModel().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        min_loss = float('inf')
        best_state = None

        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_state = model.state_dict()

        torch.save({
            "model_state": best_state,
            "min_loss": float(min_loss),
            "seq_length": int(seq_length),
            "scaler": scaler,
        }, model_path)

    def infer_lstm_univariate_compat(
        y_data: np.ndarray,        # shape (N,)
        model_path: str,
        default_seq_length: int = 24
    ):
        """
        兼容旧模型：
        - 旧版：{'model': state_dict, 'min_loss': ...}
        - 新版：{'model_state': ..., 'seq_length':..., 'scaler':...}
        """
        ckpt = torch_load_compat(model_path, map_location=device)

        seq_length = int(ckpt.get("seq_length", default_seq_length))

        state_dict = ckpt.get("model_state", None)
        if state_dict is None:
            state_dict = ckpt.get("model", None)
        if state_dict is None:
            raise KeyError("checkpoint 中未找到 model_state 或 model")

        min_loss = float(ckpt.get("min_loss", 0.0))

        scaler = ckpt.get("scaler", None)
        if scaler is None:
            # 旧模型没有 scaler：按旧逻辑，用当前数据 fit
            scaler = MinMaxScaler().fit(y_data.reshape(-1, 1))

        y_scaled = scaler.transform(y_data.reshape(-1, 1))
        X, y = create_sequences(y_scaled, seq_length)

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        model = LSTMModel().to(device)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            y_pred_scaled = model(X_t).detach().cpu().numpy().reshape(-1, 1)

        raw_margin = (np.sqrt(min_loss) * 3) if min_loss > 0 else 0.0
        y_upper_scaled = y_pred_scaled + raw_margin
        y_lower_scaled = y_pred_scaled - raw_margin

        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_true = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        y_upper = scaler.inverse_transform(y_upper_scaled).flatten()
        y_lower = scaler.inverse_transform(y_lower_scaled).flatten()

        return seq_length, y_true, y_pred, y_lower, y_upper, min_loss

# =========================
# ARIMA/AR train & infer
# =========================
def train_arima_and_save(train: np.ndarray, model_path: str, order=(24, 0, 1)):
    ensure_dir_for_file(model_path)
    model = sm.tsa.ARIMA(train, order=order).fit()
    mse_loss = float(np.mean(model.resid ** 2))
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "loss": mse_loss, "order": order}, f)

def infer_arima_test_only(model_path: str, test: np.ndarray):
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)
    model = loaded["model"]
    mse = float(loaded["loss"])

    # 这里保留你原来的“滚动一步预测 + extend”，保证行为一致
    preds = []
    current_model = model
    for true_value in test:
        pred = current_model.forecast(steps=1)
        preds.append(pred[0])
        current_model = current_model.extend([true_value])

    preds = np.asarray(preds)
    raw_margin = np.sqrt(mse) * 3
    y_upper = preds + raw_margin
    y_lower = preds - raw_margin
    return preds, y_lower, y_upper, mse

# =========================
# Main
# =========================
def ASS1_main(num_workers: int = 4):
    if TORCH_AVAILABLE:
        print(f"[INFO] TORCH_AVAILABLE=True, USE_GPU_SWITCH={USE_GPU_SWITCH}, torch.cuda.is_available()={torch.cuda.is_available()}, device={device}")
    else:
        print(f"[INFO] TORCH_AVAILABLE=False, 未安装 PyTorch；dl 将回退为 ARIMA/AR。device=cpu")

    x_time = load_time_series()

    for item, infor in ass1config.tasks_channels.items():
        # infor: [task_name, [channels], method_flag]
        data_path = get_data_path(item, file_infor[item], infor[0])
        full_data = pd.read_csv(data_path, header=None)

        if len(full_data) < 400:
            print(f"[WARN] item={item} 数据太少(len={len(full_data)})，跳过该 item。")
            continue

        # 原始 train_size 规则
        train_size_raw = max(400, int(len(full_data) * 0.5))

        # 是否 dl（配置）以及实际是否 dl
        is_dl_cfg = (infor[2] == 'dl')
        is_dl = is_dl_cfg and TORCH_AVAILABLE
        if is_dl_cfg and (not TORCH_AVAILABLE):
            print(f"[WARN] item={item} 配置为 dl(LSTM)，但未安装 PyTorch；已自动回退为 ARIMA/AR。")

        for channel_index, channel in enumerate(infor[1]):
            model_save_path = outputconfig['assessment1'][item]['save_path'][channel_index]
            fig_path = outputconfig['assessment1'][item]['figure_path'][channel_index]

            series = full_data.iloc[:, channel - 1]
            if series.isna().all():
                print(f"[WARN] item={item}, channel={channel} 全为 NaN，跳过。")
                continue

            y_data = series.fillna(float(np.mean(series))).to_numpy()

            title = f"{label_infor[item][0]}{channel}号通道{task_infor[infor[0]]}评估图"
            ylabel = f"{label_infor[item][0]}{task_infor[infor[0]]}数据/{label_infor[item][1]}"

            try:
                # ========== 训练（无模型时） ==========
                if not os.path.exists(model_save_path):
                    print(f"[INFO] item={item}, channel={channel} 未检测到模型：{os.path.basename(model_save_path)}，将训练。")

                    if is_dl:
                        train_lstm_and_save_univariate(
                            y_data=y_data,
                            seq_length=24,
                            train_size_raw=train_size_raw,
                            model_path=model_save_path,
                            lr=0.01,
                            max_epochs=300
                        )
                    else:
                        # ARIMA/AR
                        split = min(train_size_raw, len(y_data))
                        train = y_data[:split]
                        train_arima_and_save(train, model_save_path, order=(24, 0, 1))

                # ========== 推理 + 画图（含训练段） ==========
                if is_dl:
                    # LSTM：序列对齐
                    seq_length, y_true_seq, y_pred_seq, y_lower_seq, y_upper_seq, _ = infer_lstm_univariate_compat(
                        y_data=y_data,
                        model_path=model_save_path,
                        default_seq_length=24
                    )

                    # y_true_seq/y_pred_seq 对应原始索引 [seq_length, ..., N-1]
                    x_plot = x_time.iloc[seq_length:]
                    y_all_plot = y_data[seq_length:]

                    # 训练分割点映射到序列空间
                    train_split_idx = max(0, min(len(y_all_plot), train_size_raw - seq_length))

                    # 只画测试段预测/区间
                    y_pred_test = y_pred_seq[train_split_idx:]
                    y_lower_test = y_lower_seq[train_split_idx:]
                    y_upper_test = y_upper_seq[train_split_idx:]

                    plot_train_test(
                        fig_path=fig_path,
                        x_time=x_plot,
                        y_all=y_all_plot,
                        train_split_idx=train_split_idx,
                        y_pred_test=y_pred_test,
                        y_lower_test=y_lower_test,
                        y_upper_test=y_upper_test,
                        title=title,
                        ylabel=ylabel,
                        method="LSTM"
                    )
                else:
                    split = min(train_size_raw, len(y_data))
                    test = y_data[split:]
                    if len(test) == 0:
                        print(f"[WARN] item={item}, channel={channel} 测试段为空，跳过绘图。")
                        continue

                    preds, y_lower, y_upper, _ = infer_arima_test_only(model_save_path, test)

                    plot_train_test(
                        fig_path=fig_path,
                        x_time=x_time,
                        y_all=y_data,
                        train_split_idx=split,
                        y_pred_test=preds,
                        y_lower_test=y_lower,
                        y_upper_test=y_upper,
                        title=title,
                        ylabel=ylabel,
                        method="AR"
                    )

            except Exception as e:
                print(f"[ERROR] item={item}, channel={channel} 处理失败：{e}")
                continue

if __name__ == "__main__":
    ASS1_main()
