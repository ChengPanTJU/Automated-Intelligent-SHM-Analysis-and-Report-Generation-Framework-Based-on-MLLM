import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import OutputConfig, Ass2Config

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
# warnings / plot global
# =========================
warnings.simplefilter("ignore", ConvergenceWarning)

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# config load
# =========================
ass2config = Ass2Config()
OutConfig = OutputConfig()
outputconfig = OutConfig.tasks

# =========================
# dirs
# =========================
upper_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(upper_dir, 'PostProcessing', 'assessment2', 'rlt_figure'), exist_ok=True)

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

def fillna_by_col_mean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(lambda col: col.fillna(col.mean()), axis=0)
    df = df.dropna(axis=1, how='all')
    return df

def create_sequences(X: np.ndarray, y: np.ndarray, window_size: int):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    Xs, ys = [], []
    n = len(X)
    for i in range(n - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size])
    return np.asarray(Xs), np.asarray(ys)

def torch_load_compat(path: str, map_location):
    """
    兼容不同 pytorch 版本对 torch.load 参数的差异。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

# =========================
# Plot: 带“训练数据”曲线
# =========================
def plot_assessment_train_test(
    fig_path: str,
    x_time: pd.Series,
    y_all: np.ndarray,          # 与 x_time 等长的真实序列
    y_pred_test: np.ndarray,    # 测试段预测（长度 = len(x_time) - train_split_idx）
    y_lower_test: np.ndarray,
    y_upper_test: np.ndarray,
    title: str,
    ylabel: str,
    method: str,
    train_split_idx: int,
):
    ensure_dir_for_file(fig_path)

    plt.rcParams['font.size'] = 12
    fig_width = 14 * 0.393701
    fig_height = 8 * 0.393701
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 训练段
    ax.plot(x_time.iloc[:train_split_idx], y_all[:train_split_idx],
            label="训练数据", linewidth=0.5, alpha=0.5,color='blue')

    # 测试段真实值
    ax.plot(x_time.iloc[train_split_idx:], y_all[train_split_idx:],
            label="真实值", linewidth=0.5, alpha=0.5,color='green')

    # 测试段预测值
    ax.plot(x_time.iloc[train_split_idx:], y_pred_test,
            label="预测值", linestyle='dashed', linewidth=0.5, alpha=0.5,color='red')

    # 测试段置信区间
    ax.fill_between(x_time.iloc[train_split_idx:], y_lower_test, y_upper_test,
                    alpha=0.5, label="置信区间",color='gray')

    # x ticks: 9 个
    num_ticks = 9
    tick_locs = np.linspace(0, len(x_time) - 1, num_ticks, dtype=int)
    ax.set_xticks(x_time.iloc[tick_locs])
    ax.set_xticklabels([pd.to_datetime(x_time.iloc[i]).strftime('%Y-%m-%d') for i in tick_locs],
                       rotation=20, fontsize=10, fontfamily='Times New Roman')

    ax.set_xlabel("时间 (年-月-日)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}(方法:{method})")

    fig.subplots_adjust(bottom=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.12),
              bbox_transform=fig.transFigure, ncol=4, fontsize=10)

    plt.savefig(fig_path, format='png', dpi=300)
    plt.close(fig)

# =========================
# LSTM：只有 TORCH_AVAILABLE 才定义/使用
# =========================
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 2, output_size: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # 默认 0 初始化隐状态，稳定
            out, _ = self.lstm(x)
            return self.linear(out[:, -1, :])

    def train_lstm_and_save(
        X_data: np.ndarray,
        y_data: np.ndarray,
        seq_length: int,
        train_size: int,
        model_path: str,
        hidden_dim: int = 50,
        num_layers: int = 2,
        lr: float = 0.01,
        max_epochs: int = 2000,
        patience: int = 100,
    ):
        ensure_dir_for_file(model_path)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(X_data)
        y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1))

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
        if len(X_seq) < 10:
            raise ValueError("序列数据太短，不足以训练 LSTM")

        train_size = min(train_size, len(X_seq))
        X_train = torch.tensor(X_seq[:train_size], dtype=torch.float32, device=device)
        y_train = torch.tensor(y_seq[:train_size], dtype=torch.float32, device=device)

        input_dim = X_data.shape[1]
        model = LSTMModel(input_dim, hidden_dim, num_layers, output_size=1).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=60)

        min_loss = float('inf')
        patience_counter = 0
        best_epoch = -1

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            pred = model(X_train)
            loss = criterion(pred, y_train)

            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "min_loss": float(min_loss),
                    "best_epoch": int(best_epoch),
                    "seq_length": int(seq_length),
                    "input_dim": int(input_dim),
                    "hidden_dim": int(hidden_dim),
                    "num_layers": int(num_layers),
                    "scaler_X": scaler_X,
                    "scaler_y": scaler_y,
                }, model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    def infer_lstm_from_saved(
        X_data: np.ndarray,
        y_data: np.ndarray,
        model_path: str,
        default_seq_length: int = 24,
    ):
        """
        兼容旧 checkpoint：
        - 旧版可能只有 model/min_loss/epoch（无 seq_length/scaler）
        - seq_length 缺失 -> default_seq_length
        - scaler 缺失 -> 用当前数据 fit（与旧版逻辑一致）
        """
        ckpt = torch_load_compat(model_path, map_location=device)

        seq_length = int(ckpt.get("seq_length", default_seq_length))

        state_dict = ckpt.get("model_state", None)
        if state_dict is None:
            state_dict = ckpt.get("model", None)
        if state_dict is None:
            raise KeyError("checkpoint 中未找到 model_state 或 model")

        min_loss = float(ckpt.get("min_loss", 0.0))
        input_dim = int(ckpt.get("input_dim", X_data.shape[1]))
        hidden_dim = int(ckpt.get("hidden_dim", 50))
        num_layers = int(ckpt.get("num_layers", 2))

        scaler_X = ckpt.get("scaler_X", None)
        scaler_y = ckpt.get("scaler_y", None)
        if scaler_X is None or scaler_y is None:
            scaler_X = MinMaxScaler().fit(X_data)
            scaler_y = MinMaxScaler().fit(y_data.reshape(-1, 1))

        X_scaled = scaler_X.transform(X_data)
        y_scaled = scaler_y.transform(y_data.reshape(-1, 1))

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

        X_t = torch.tensor(X_seq, dtype=torch.float32, device=device)
        model = LSTMModel(input_dim, hidden_dim, num_layers, output_size=1).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            y_pred_scaled = model(X_t).detach().cpu().numpy().reshape(-1, 1)

        raw_margin = np.sqrt(min_loss) * 3 if min_loss > 0 else 0.0
        y_upper_scaled = y_pred_scaled + raw_margin
        y_lower_scaled = y_pred_scaled - raw_margin

        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = scaler_y.inverse_transform(y_seq.reshape(-1, 1)).flatten()
        y_upper = scaler_y.inverse_transform(y_upper_scaled).flatten()
        y_lower = scaler_y.inverse_transform(y_lower_scaled).flatten()

        return seq_length, y_true, y_pred, y_lower, y_upper

# =========================
# SARIMAX train / infer
# =========================
def train_sarimax_and_save(X_train: np.ndarray, y_train: np.ndarray, model_path: str, order=(20, 1, 5)):
    ensure_dir_for_file(model_path)
    model = SARIMAX(y_train, exog=X_train, order=order)
    model_fit = model.fit(disp=False)
    y_train_pred = model_fit.fittedvalues
    mse_loss = mean_squared_error(y_train, y_train_pred)

    with open(model_path, "wb") as f:
        pickle.dump({"model": model_fit, "loss": mse_loss, "order": order}, f)

def infer_sarimax_from_saved_test_only(X_test: np.ndarray, y_test: np.ndarray, model_path: str):
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)
    model_fit = loaded["model"]
    mse = loaded["loss"]

    y_pred_test = np.asarray(model_fit.forecast(steps=len(y_test), exog=X_test))
    raw_margin = np.sqrt(mse) * 3
    y_upper_test = y_pred_test + raw_margin
    y_lower_test = y_pred_test - raw_margin
    return y_pred_test, y_lower_test, y_upper_test

# =========================
# Meta dicts
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
# Main
# =========================
def ASS2_main(num_workers: int = 1):
    if TORCH_AVAILABLE:
        print(f"[INFO] TORCH_AVAILABLE=True, USE_GPU_SWITCH={USE_GPU_SWITCH}, torch.cuda.is_available()={torch.cuda.is_available()}, device={device}")
    else:
        print(f"[INFO] TORCH_AVAILABLE=False, 未安装 PyTorch；dl 将回退为 ARIMAX。device=cpu")

    x_time = load_time_series()

    for item, data_infor in ass2config.tasks_channels.items():
        data_list = list(data_infor.keys())

        # target
        target_data_path = get_data_path(data_list[0], file_infor[data_list[0]], data_infor[data_list[0]][0])

        # sources -> X
        source_data_path_list = [
            get_data_path(data_list[i], file_infor[data_list[i]], data_infor[data_list[i]][0])
            for i in range(1, len(data_list))
        ]
        source_data_channels_list = [
            np.array(data_infor[data_list[i]][1]) - 1
            for i in range(1, len(data_list))
        ]

        df_list = []
        for p, ch_idx in zip(source_data_path_list, source_data_channels_list):
            source_data = pd.read_csv(p, header=None)
            source_data = source_data.iloc[:, ch_idx]
            df_list.append(source_data)

        X_df = pd.concat(df_list, axis=1)
        X_df = fillna_by_col_mean(X_df)
        X_data = X_df.to_numpy()

        full_data = pd.read_csv(target_data_path, header=None)

        if len(full_data) < 400:
            print(f"[WARN] item={item} 数据太少(len={len(full_data)})，跳过该 item。")
            continue

        train_size_raw = max(400, int(len(full_data) * 0.5))

        # 配置是否 dl
        is_dl_cfg = (data_infor[data_list[0]][2] == 'dl')
        # 实际是否 dl（未装 torch 时回退）
        is_dl = is_dl_cfg and TORCH_AVAILABLE
        if is_dl_cfg and (not TORCH_AVAILABLE):
            print(f"[WARN] item={item} 配置为 dl(LSTM)，但未安装 PyTorch；已自动回退为 ARIMAX。")

        for channel_index in range(len(data_infor[data_list[0]][1])):
            channel = data_infor[data_list[0]][1][channel_index]
            single_channel_data = full_data.iloc[:, channel - 1]

            if single_channel_data.isna().all():
                print(f"[WARN] item={item} 被拟合列(第{channel}列) 全为 NaN，跳过。")
                continue

            y_data = single_channel_data.fillna(float(np.mean(single_channel_data))).to_numpy()

            model_save_path = outputconfig['assessment2'][item]['save_path'][channel_index]
            fig_name = outputconfig['assessment2'][item]['figure_path'][channel_index]

            try:
                # ========== 训练（无模型时） ==========
                if not os.path.exists(model_save_path):
                    print(f"[INFO] item={item}, channel={channel} 未检测到模型：{os.path.basename(model_save_path)}，将训练。")

                    if is_dl:
                        train_lstm_and_save(
                            X_data=X_data,
                            y_data=y_data,
                            seq_length=24,
                            train_size=train_size_raw,
                            model_path=model_save_path
                        )
                    else:
                        # ARIMAX
                        split = min(train_size_raw, len(y_data))
                        X_train, y_train = X_data[:split], y_data[:split]
                        train_sarimax_and_save(X_train, y_train, model_save_path, order=(20, 1, 5))

                # ========== 推理 + 出图（带训练段） ==========
                title = f"{label_infor[data_list[0]][0]}{channel}号通道{task_infor[data_infor[data_list[0]][0]]}评估图"
                ylabel = f"{label_infor[data_list[0]][0]}{task_infor[data_infor[data_list[0]][0]]}数据/{label_infor[data_list[0]][1]}"

                if is_dl:
                    # LSTM 输出从 seq_length 对齐
                    seq_length, y_true_seq, y_pred_seq, y_lower_seq, y_upper_seq = infer_lstm_from_saved(
                        X_data=X_data,
                        y_data=y_data,
                        model_path=model_save_path,
                        default_seq_length=24
                    )

                    # 序列空间对齐
                    x_plot = x_time.iloc[seq_length:]
                    y_all_plot = y_data[seq_length:]

                    # 训练分割点映射到序列空间
                    train_split_idx = max(0, min(len(y_all_plot), train_size_raw - seq_length))

                    # 只画测试段预测/区间（恢复“训练数据”曲线）
                    y_pred_test = y_pred_seq[train_split_idx:]
                    y_lower_test = y_lower_seq[train_split_idx:]
                    y_upper_test = y_upper_seq[train_split_idx:]

                    plot_assessment_train_test(
                        fig_path=fig_name,
                        x_time=x_plot,
                        y_all=y_all_plot,
                        y_pred_test=y_pred_test,
                        y_lower_test=y_lower_test,
                        y_upper_test=y_upper_test,
                        title=title,
                        ylabel=ylabel,
                        method="LSTM",
                        train_split_idx=train_split_idx
                    )

                else:
                    # ARIMAX：按 train/test 切分，只预测测试段
                    split = min(train_size_raw, len(y_data))
                    X_test, y_test = X_data[split:], y_data[split:]

                    # 若 split 已到末尾，测试集为空：直接跳过绘图
                    if len(y_test) == 0:
                        print(f"[WARN] item={item}, channel={channel} 测试段为空，跳过绘图。")
                        continue

                    y_pred_test, y_lower_test, y_upper_test = infer_sarimax_from_saved_test_only(
                        X_test=X_test,
                        y_test=y_test,
                        model_path=model_save_path
                    )

                    plot_assessment_train_test(
                        fig_path=fig_name,
                        x_time=x_time,
                        y_all=y_data,
                        y_pred_test=y_pred_test,
                        y_lower_test=y_lower_test,
                        y_upper_test=y_upper_test,
                        title=title,
                        ylabel=ylabel,
                        method="ARIMAX",
                        train_split_idx=split
                    )

            except Exception as e:
                print(f"[ERROR] item={item}, channel={channel} 处理失败：{e}")
                continue

if __name__ == "__main__":
    ASS2_main()
