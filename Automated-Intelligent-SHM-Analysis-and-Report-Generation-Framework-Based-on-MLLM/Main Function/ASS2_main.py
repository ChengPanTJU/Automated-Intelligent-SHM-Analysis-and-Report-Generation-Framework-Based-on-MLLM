import os
import pickle
import warnings

from sklearn.metrics import mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.simplefilter("ignore", ConvergenceWarning)
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from config import OutputConfig,Ass2Config
ass2config=Ass2Config()
OutConfig = OutputConfig()
outputconfig = OutConfig.tasks
# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置全局字体为宋体，字号为12
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def get_data_path(type, task_name):
    """
    根据任务名称生成保存结果的文件路径和文件名。
    假设结果文件名为：<任务名称>_results.csv
    """
    # 生成文件保存路径
    current_path=os.getcwd()
    file_name = f"rlt_{type}_{task_name}.csv"
    dir2 = f"rlt_table"
    path = os.path.join(current_path,'Post_processing', dir2)
    save_path = os.path.join(path, file_name)

    return save_path

def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size-1])
    return np.array(Xs), np.array(ys)

# 定义 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, win_out=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.win_out = win_out
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        input_seq = input_seq.to(device)  # 确保 input 在同一设备
        h_0 = torch.randn(self.num_directions * self.num_layers, input_seq.size(0), self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, input_seq.size(0), self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))  # output
        out = self.linear(output[:, -self.win_out, :])   # (batch_size, sequence_length, output_size)
        return out

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
    'OMA1':'频率',
    'common_analysis': ''
}
label_infor={
    'displacement': ['位移','mm'],
    'temperature': ['温度','℃'],
    'strain': ['应变','微应变'],
    'vibration': ['主梁加速度','mg'],
    'cable_vib': ['拉索加速度','mg'],
    'wind_speed': ['风速','m/s'],
    'wind_direction': ['风向','°'],
    'inclination': ['倾角','°'],
    'settlement': ['沉降','mm'],
    'GPS': ['GPS','mm'],
    'cable_force': ['索力','kN'],
    'traffic': ['交通荷载',' '],
}
for item,data_infor in ass2config.tasks_channels.items():
    data_list = list(data_infor.keys())
    target_data_path = get_data_path(file_infor[data_list[0]], data_infor[data_list[0]][0])
    source_data_path_list =[get_data_path(file_infor[data_list[i]], data_infor[data_list[i]][0]) for i in range(1,len(data_list))]
    source_data_channels_list =[np.array(data_infor[data_list[i]][1])-1 for i in range(1,len(data_list))]
    df_list = []
    for i in range(len(source_data_path_list)):
        source_data_path=source_data_path_list[i]
        source_data=pd.read_csv(source_data_path, header=None)
        source_data =source_data.iloc[:,source_data_channels_list[i]]
        df_list.append(source_data)
    # **横向拼接**
    X_data = pd.concat(df_list, axis=1)
    X_data = X_data.apply(lambda col: col.fillna(col.mean()), axis=0)
    X_data = X_data.dropna(axis=1, how='all')
    X_data =np.array(X_data)
    full_data = pd.read_csv(target_data_path, header=None)
    for channel_index in range(len(data_infor[data_list[0]][1])):
        channel = data_infor[data_list[0]][1][channel_index]
        single_channel_data = full_data.iloc[:, channel-1]
        if single_channel_data.isna().all().all():
            print(f"被拟合列(第{channel}列）全为nan，跳过拟合")
        else:
            #single_channel_data = single_channel_data.fillna(np.mean(single_channel_data))
            import AnalysisFunction.analysis_methods as AM
            single_channel_data, _ = AM.rmoutliers_gesd(single_channel_data)
            model_save_path = outputconfig['assessment2'][item]['save_path'][channel_index]
            if not os.path.exists(model_save_path):  #########如果没模型，就训练
                print(f'未检测到模型路径：{os.path.basename(model_save_path)},将进行训练')
                seq_length = 24  ##############周期
                train_size = max(400, int(len(full_data) * 0.5))  ############至少用300条训练
                if len(full_data) < 400:
                    print('数据太少，不支持模型训练')
                    break
                if data_infor[data_list[0]][2] == 'dl':
                    x_time_path = os.path.join(os.getcwd(), 'Post_processing', "rlt_table", f"rlt_time.csv")
                    x_time = pd.read_csv(x_time_path, header=None)
                    x_time[0] = pd.to_datetime(x_time[0], errors='coerce')
                    tick_data = x_time.copy()
                    year = x_time.iloc[-1, 0].year  # 获取时间序列的最后一年
                    x_time[0] = x_time[0].dt.strftime("%Y-%m-%d %H")
                    y_data = np.array(single_channel_data.tolist())
                    y_data[:train_size], _ = AM.rmoutliers_gesd(y_data[:train_size])
                    y_data[train_size:], _ = AM.rmoutliers_gesd(y_data[train_size:])
                    scaler_X = MinMaxScaler()
                    scaler_y = MinMaxScaler()

                    X_scaled = scaler_X.fit_transform(X_data)  # 归一化输入数据
                    y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1))  # 归一化目标数据
                    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

                    # 划分训练集和测试集
                    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
                    X_test, y_test = X_seq[train_size:], y_seq[train_size:]

                    # 转换为 PyTorch 张量
                    X_train = torch.tensor(X_train, dtype=torch.float32)
                    y_train = torch.tensor(y_train, dtype=torch.float32)
                    X_test = torch.tensor(X_test, dtype=torch.float32)
                    y_test = torch.tensor(y_test, dtype=torch.float32)
                    X_train = X_train.to(device)
                    y_train = y_train.to(device)
                    X_test = X_test.to(device)
                    # 训练模型
                    input_dim = X_data.shape[1] # 输入特征维度
                    hidden_dim = 50  # LSTM 隐藏层维度
                    num_layers = 2  # LSTM 层数
                    output_dim = 1  # 预测目标变量
                    weight_decay=1e-5
                    model = LSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)
                    criterion = nn.MSELoss()
                    #############第一次训练
                    optimizer = optim.Adam(model.parameters(), lr=0.005)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,verbose=True)
                    patience = 15 # 早停 patience
                    min_loss = float('inf')  # 记录最小 loss
                    patience_counter = 0  # 记录连续未改进的次数
                    # 确保保存路径存在
                    save_dir = 'Post_processing/ass2_model'
                    os.makedirs(save_dir, exist_ok=True)
                    pth_name = os.path.basename(model_save_path)
                    file_path = os.path.join(save_dir, pth_name)
                    from torch.utils.data import Dataset, DataLoader
                    class SequenceDataset(Dataset):
                        def __init__(self, X, y):
                            self.X = X
                            self.y = y

                        def __len__(self):
                            return len(self.X)

                        def __getitem__(self, idx):
                            return self.X[idx], self.y[idx]
                    batch_size = 100
                    train_dataset = SequenceDataset(X_train, y_train)
                    test_dataset = SequenceDataset(X_test, y_test)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                    num_epochs = 500
                    train_losses = []
                    for epoch in range(num_epochs):
                        model.train()
                        running_loss = 0.0
                        for X_batch, y_batch in train_loader:
                            optimizer.zero_grad()
                            outputs = model(X_batch)
                            # 找出不是 NaN 的位置
                            mask = ~torch.isnan(y_batch)
                            # 仅对非 NaN 的位置计算 loss
                            if mask.sum() == 0:
                                continue  # 如果整批数据都是 NaN，则跳过
                            loss = criterion(outputs[mask], y_batch[mask])
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item() * mask.sum().item()  # 注意是 mask.sum()

                        # 计算平均 loss（防止除以全是 NaN 的总数）
                        total_valid = sum((~torch.isnan(y_batch)).sum().item() for _, y_batch in train_loader)
                        epoch_loss = running_loss / max(total_valid, 1)  # 防止除以 0
                        train_losses.append(epoch_loss)
                        scheduler.step(epoch_loss)
                        if epoch_loss < min_loss:
                            min_loss = epoch_loss
                            patience_counter = 0
                            #torch.save(model.state_dict(), file_path)
                        else:
                            patience_counter += 1
                        if patience_counter >= patience:
                            break

                    model.eval()

                    y_preds = []
                    y_trues = []

                    with torch.no_grad():
                        for X_batch, y_batch in test_loader:
                            preds = model(X_batch).cpu().numpy()
                            y_true = y_batch.cpu().numpy()

                            y_preds.append(preds)
                            y_trues.append(y_true)

                    # 合并结果
                    y_pred = np.vstack(y_preds)
                    y_true = np.vstack(y_trues)

                    raw_margin = np.sqrt(min_loss)
                    # 计算上下限
                    y_upper = y_pred + raw_magin*2.576
                    y_lower = y_pred - raw_margin*2.576
                    # 反归一化
                    y_pred = scaler_y.inverse_transform(y_pred)
                    y_test = scaler_y.inverse_transform(y_true)
                    y_upper = scaler_y.inverse_transform(y_upper)
                    y_lower = scaler_y.inverse_transform(y_lower)

                    # 画图
                    plt.rcParams['font.size'] = 12  # 设置字体大小为12
                    fig_width = 14 * 0.393701  # cm 转换为英寸 图幅大小
                    fig_height = 8 * 0.393701  # cm 转换为英寸
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    plt.plot(x_time.iloc[:train_size + seq_length + 1, 0], y_data[:train_size + seq_length + 1],
                             label="训练数据", color='blue',linewidth=0.5,alpha=0.5)
                    plt.plot(x_time.iloc[train_size + seq_length:, 0], y_test, label="真实值", color='green',linewidth=0.5,alpha=0.5)
                    plt.plot(x_time.iloc[train_size + seq_length:, 0], y_pred, label="预测值", color='red',
                             linestyle='dashed',linewidth=0.5,alpha=0.5)

                    plt.fill_between(range(train_size + seq_length, len(y_pred) + train_size + seq_length),
                                     y_lower.flatten(), y_upper.flatten(), color='gray',
                                     alpha=0.5, label="置信区间")

                    xtick_numbers = np.arange(0, len(x_time), int(len(x_time) / 9))
                    tick_data[0] = tick_data[0].dt.strftime("%Y-%m-%d")
                    custom_xtick_labels = [tick_data[0][i] for i in xtick_numbers]  # 替换为您需要的标签
                    # 设置 X 轴的标签位置
                    xtick_positions = xtick_numbers
                    plt.xlabel("时间 (年-月-日）")
                    plt.ylabel(f"{label_infor[data_list[0]][0]}{task_infor[data_infor[data_list[0]][0]]}数据/{label_infor[data_list[0]][1]}")
                    ax.set_xticks(xtick_positions)
                    ax.set_xticklabels(custom_xtick_labels, rotation=20, fontfamily='Times New Roman', fontsize=10)
                    plt.xticks(fontsize=10)  # x 轴标签旋转 30 度
                    plt.yticks(fontsize=10)
                    plt.title(f"{label_infor[data_list[0]][0]}{channel}号通道{task_infor[data_infor[data_list[0]][0]]}评估图(方法:LSTM)")
                    fig.subplots_adjust(bottom=0.3)  # 额外增加一些空间

                    legend = ax.legend(loc='upper center',
                                       bbox_to_anchor=(0.5, 0.12),
                                       bbox_transform=fig.transFigure,  # 使位置相对于整个 figure
                                       ncol=4, fontsize=10)
                    fig_name = outputconfig['assessment2'][item]['figure_path'][channel_index]
                    plt.savefig(fig_name, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
                    plt.close()

                else:
                    x_time_path = os.path.join(os.getcwd(), 'Post_processing', "rlt_table", f"rlt_time.csv")
                    x_time = pd.read_csv(x_time_path, header=None)
                    x_time[0] = pd.to_datetime(x_time[0], errors='coerce')
                    y_data = np.array(single_channel_data.tolist())
                    train_size = max(400, int(len(full_data) * 0.5))

                    # 划分训练集和测试集
                    X_train, y_train = X_data[:train_size], y_data[:train_size]
                    X_test, y_test = X_data[train_size:], y_data[train_size:]
                    # 3. 训练 ARIMAX 模型
                    model = SARIMAX(y_train, exog=X_train, order=(20, 1, 5))
                    model_fit = model.fit(disp=False)  # 关闭优化日志
                    # 获取训练集上的预测值
                    y_train_pred = model_fit.fittedvalues  # SARIMAX 在训练集上的拟合值
                    # 计算训练集的 MSE
                    mse_loss = mean_squared_error(y_train, y_train_pred)
                    pth_name = os.path.basename(model_save_path)
                    file_path = os.path.join('Post_processing\\ass2_model', pth_name)
                    if not os.path.exists('Post_processing\\ass2_model'):  # 如果目录不存在
                        os.makedirs('Post_processing\\ass2_model')  # 创建目录
                    # **将模型和 loss 一起保存**
                    with open(file_path, "wb") as f:
                        pickle.dump({"model": model_fit, "loss": mse_loss}, f)

                    # **加载模型和 loss**
                    with open(file_path, "rb") as f:
                        loaded_data = pickle.load(f)
                    # 4. 预测
                    test_predictions = []  # 存储预测值
                    current_model = loaded_data["model"]  # 初始训练好的模型

                    batch_size = 10
                    n = len(y_test)

                    for t in range(0, n, batch_size):
                        actual_batch_size = min(batch_size, n - t)  # 避免超出范围
                        pred = current_model.forecast(steps=actual_batch_size, exog=X_test[t:t + actual_batch_size])
                        test_predictions.extend(pred)
                        current_model = current_model.append(y_test[t:t + actual_batch_size],
                                                             exog=X_test[t:t + actual_batch_size])

                    # 计算评估指标
                    mse = loaded_data["loss"]
                    raw_margin = np.sqrt(mse) * 3
                    y_upper = np.array(test_predictions) + raw_margin
                    y_lower = np.array(test_predictions) - raw_margin
                    plt.rcParams['font.size'] = 12  # 设置字体大小为12
                    fig_width = 14 * 0.393701  # cm 转换为英寸 图幅大小
                    fig_height = 8 * 0.393701  # cm 转换为英寸
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    plt.plot(x_time[0][:train_size], y_train, label="训练数据", color='blue',linewidth=0.5,alpha=0.5)
                    plt.plot(x_time[0][train_size:], y_test, label="真实值", color='green',linewidth=0.5,alpha=0.5)
                    plt.plot(x_time[0][train_size:], test_predictions, label="预测值", color='red',
                             linestyle='dashed',linewidth=0.5,alpha=0.5)

                    # **绘制置信区间**
                    plt.fill_between(x_time[0][train_size:], y_upper, y_lower, color='gray',
                                     alpha=0.5, label="置信区间")
                    # 图例 & 细节
                    # 图例 & 细节
                    num_ticks = 9
                    tick_locs = np.linspace(0, len(x_time[0]) - 1, num_ticks, dtype=int)
                    tick_labels = [x_time[0][i] for i in tick_locs]
                    # 2. 设置 tick 的位置和标签
                    plt.xticks(ticks=[x_time[0][i] for i in tick_locs],
                               labels=[x_time[0][i].strftime('%Y-%m-%d') for i in tick_locs],
                               rotation=10, ha='center')  # ha 控制水平对齐方式
                    plt.xlabel("时间 (年-月-日)")
                    plt.ylabel(
                        f"{label_infor[data_list[0]][0]}{task_infor[data_infor[data_list[0]][0]]}数据/{label_infor[data_list[0]][1]}")
                    plt.xticks(fontsize=10, rotation=20)  # x 轴标签旋转 30 度
                    plt.yticks(fontsize=10)
                    plt.title(
                        f"{label_infor[data_list[0]][0]}{channel}号通道{task_infor[data_infor[data_list[0]][0]]}评估图(方法:ARIMAX)")
                    fig.subplots_adjust(bottom=0.3)  # 额外增加一些空间

                    legend = ax.legend(loc='upper center',
                                       bbox_to_anchor=(0.5, 0.12),
                                       bbox_transform=fig.transFigure,  # 使位置相对于整个 figure
                                       ncol=4, fontsize=10)
                    fig_name = outputconfig['assessment2'][item]['figure_path'][channel_index]
                    plt.savefig(fig_name, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
                    plt.close()

            else:
                if data_infor[data_list[0]][2] == 'dl':
                    seq_length = 24
                    x_time_path = os.path.join(os.getcwd(), 'Post_processing', "rlt_table", f"rlt_time.csv")
                    x_time = pd.read_csv(x_time_path, header=None)
                    x_time[0] = pd.to_datetime(x_time[0], errors='coerce')
                    tick_data = x_time.copy()
                    year = x_time.iloc[-1, 0].year  # 获取时间序列的最后一年
                    x_time[0] = x_time[0].dt.strftime("%m.%d-%H")

                    y_data = np.array(single_channel_data.tolist())
                    scaler_X = MinMaxScaler()
                    scaler_y = MinMaxScaler()

                    X_scaled = scaler_X.fit_transform(X_data)  # 归一化输入数据
                    y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1))  # 归一化目标数据
                    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

                    X_test, y_test = X_seq, y_seq

                    # 转换为 PyTorch 张量

                    X_test = torch.tensor(X_test, dtype=torch.float32)
                    y_test = torch.tensor(y_test, dtype=torch.float32)
                    X_test = X_test.to(device)
                    # 训练模型
                    input_dim = X_data.shape[1] # 输入特征维度
                    hidden_dim = 50  # LSTM 隐藏层维度
                    num_layers = 2  # LSTM 层数
                    output_dim = 1  # 预测目标变量
                    checkpoint = torch.load(model_save_path)
                    model = LSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)
                    model.load_state_dict(checkpoint['model'])  # 载入模型参数
                    min_loss = checkpoint['min_loss']  # 载入最小 loss
                    # 预测
                    model.eval()

                    y_pred = model(X_test).detach().cpu().numpy()
                    y_pred = y_pred.reshape(-1, 1)  # 变成 2D 形状# 先移动到 CPU，再转换为 NumPy
                    raw_margin = np.sqrt(min_loss) * 3
                    # 计算上下限
                    y_upper = y_pred + raw_margin
                    y_lower = y_pred - raw_margin
                    # 反归一化
                    y_pred = scaler_y.inverse_transform(y_pred)
                    y_test = scaler_y.inverse_transform(y_test.numpy())
                    y_upper = scaler_y.inverse_transform(y_upper)
                    y_lower = scaler_y.inverse_transform(y_lower)

                    # 画图
                    plt.rcParams['font.size'] = 12  # 设置字体大小为12
                    fig_width = 14 * 0.393701  # cm 转换为英寸 图幅大小
                    fig_height = 8 * 0.393701  # cm 转换为英寸
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

                    plt.plot(x_time.iloc[seq_length:, 0], y_test, label="真实值", color='green',linewidth=0.5,alpha=0.5)
                    plt.plot(x_time.iloc[ seq_length:, 0], y_pred, label="预测值", color='red',
                             linestyle='dashed',linewidth=0.5,alpha=0.5)

                    plt.fill_between(range(0, len(y_pred)), y_lower.flatten(),
                                     y_upper.flatten(), color='gray',
                                     alpha=0.5, label="置信区间")

                    xtick_numbers = np.arange(0, len(x_time), int(len(x_time) / 9))
                    tick_data[0] = tick_data[0].dt.strftime("%Y-%m-%d")
                    custom_xtick_labels = [tick_data[0][i] for i in xtick_numbers]  # 替换为您需要的标签
                    # 设置 X 轴的标签位置
                    xtick_positions = xtick_numbers
                    plt.xlabel("时间 (年-月-日）")
                    plt.ylabel(f"{label_infor[data_list[0]][0]}{task_infor[data_infor[data_list[0]][0]]}数据/{label_infor[data_list[0]][1]}")
                    ax.set_xticks(xtick_positions)
                    ax.set_xticklabels(custom_xtick_labels, rotation=20, fontfamily='Times New Roman', fontsize=10)
                    plt.xticks(fontsize=10)  # x 轴标签旋转 30 度
                    plt.yticks(fontsize=10)
                    plt.title(f"{label_infor[data_list[0]][0]}{channel}号通道{task_infor[data_infor[data_list[0]][0]]}评估图(方法:LSTM)")
                    fig.subplots_adjust(bottom=0.3)  # 额外增加一些空间

                    legend = ax.legend(loc='upper center',
                                       bbox_to_anchor=(0.5, 0.12),
                                       bbox_transform=fig.transFigure,  # 使位置相对于整个 figure
                                       ncol=4, fontsize=10)
                    fig_name = outputconfig['assessment2'][item]['figure_path'][channel_index]
                    plt.savefig(fig_name, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
                    plt.close()
                else:
                    x_time_path = os.path.join(os.getcwd(), 'Post_processing', "rlt_table", f"rlt_time.csv")
                    x_time = pd.read_csv(x_time_path, header=None)
                    x_time[0] = pd.to_datetime(x_time[0], errors='coerce')
                    y_data = np.array(single_channel_data.tolist())

                    # 划分训练集和测试集
                    X_test, y_test = X_data[:], y_data[:]

                    pth_name = os.path.basename(model_save_path)
                    file_path = os.path.join('Post_processing\\ass2_model', pth_name)

                    # **加载模型和 loss**
                    with open(file_path, "rb") as f:
                        loaded_data = pickle.load(f)
                    # 4. 预测
                    current_model = loaded_data["model"]
                    test_predictions = current_model.forecast(steps=len(y_test), exog=X_test)
                    '''
                                    for t in range(len(y_test)):
                        # 进行一步预测
                        pred = current_model.forecast(steps=1, exog=X_test[t:t + 1])  # 预测下一步
                        test_predictions.append(pred[0])
                        # 扩展模型数据但不重新拟合
                        current_model = current_model.append(y_test[t:t + 1], exog=X_test[t:t + 1])
                    '''

                    # 计算评估指标
                    mse = loaded_data["loss"]
                    raw_margin = np.sqrt(mse) * 3
                    y_upper = np.array(test_predictions) + raw_margin
                    y_lower = np.array(test_predictions) - raw_margin
                    plt.rcParams['font.size'] = 12  # 设置字体大小为12
                    fig_width = 14 * 0.393701  # cm 转换为英寸 图幅大小
                    fig_height = 8 * 0.393701  # cm 转换为英寸
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    plt.plot(x_time[0][:], y_test, label="真实值", color='green',linewidth=0.5,alpha=0.5)
                    plt.plot(x_time[0][:], test_predictions, label="预测值", color='red',
                             linestyle='dashed',linewidth=0.5,alpha=0.5)

                    # **绘制置信区间**
                    plt.fill_between(x_time[0][:], y_upper, y_lower, color='gray',
                                     alpha=0.5, label="置信区间")
                    # 图例 & 细节
                    # 图例 & 细节
                    num_ticks = 9
                    tick_locs = np.linspace(0, len(x_time[0]) - 1, num_ticks, dtype=int)
                    tick_labels = [x_time[0][i] for i in tick_locs]
                    # 2. 设置 tick 的位置和标签
                    plt.xticks(ticks=[x_time[0][i] for i in tick_locs],
                               labels=[x_time[0][i].strftime('%Y-%m-%d') for i in tick_locs],
                               rotation=10, ha='center')  # ha 控制水平对齐方式
                    plt.xlabel("时间 (年-月-日）")
                    plt.ylabel(
                        f"{label_infor[data_list[0]][0]}{task_infor[data_infor[data_list[0]][0]]}数据/{label_infor[data_list[0]][1]}")
                    plt.xticks(fontsize=10, rotation=20)  # x 轴标签旋转 30 度
                    plt.yticks(fontsize=10)
                    plt.title(
                        f"{label_infor[data_list[0]][0]}{channel}号通道{task_infor[data_infor[data_list[0]][0]]}评估图(方法:ARIMAX)")
                    fig.subplots_adjust(bottom=0.3)  # 额外增加一些空间

                    legend = ax.legend(loc='upper center',
                                       bbox_to_anchor=(0.5, 0.12),
                                       bbox_transform=fig.transFigure,  # 使位置相对于整个 figure
                                       ncol=4, fontsize=10)
                    fig_name = outputconfig['assessment2'][item]['figure_path'][channel_index]
                    plt.savefig(fig_name, format='png', dpi=300)  # 保存为 PNG 文件，300 dpi 清晰度
                    plt.close()
