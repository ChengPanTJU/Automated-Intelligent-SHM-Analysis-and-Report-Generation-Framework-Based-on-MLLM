"""
数据分析方法，包括OMA模态识别，信号分析，统计分析等
数据输入格式均为np数组
在Acc_Analyze中直接调用的函数，需要保留参数输入接口
各处理方法需要考虑缺失值的处理。
（预处理阶段对数据进行了清洗和填充，则通常一列都为nan或都为实数值，只包含这两种情况）
例：计算均值或方差，如果某一列为nan，则该列结果为nan，其余结果正常，
使用SSI识别模态，若某一列为nan，则所有结果应为nan

Created on 2025

@author: Pan Cheng & Gong Fengzong
Email：2310450@tongji.edu.cn
"""
import numpy as np
from scipy.linalg import svd, qr, pinv
import scipy.signal as signal
from scipy.signal import csd, hilbert, butter, filtfilt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

""" ============ 巴特沃斯滤波器 =================="""


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    简化的巴特沃斯带通滤波器函数
    参数:
        data    - 输入数据（信号）
        lowcut  - 低截止频率 (Hz)
        highcut - 高截止频率 (Hz)
        fs      - 采样频率 (Hz)
        order   - 滤波器阶数 (默认4)
    返回:
        经过滤波的数据
    """
    r, c = data.shape
    if r > c:
        data = data.T
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)  # 使用零相位滤波

# 1. 低通滤波提取温度应变
def low_pass_filter(data, cutoff_freq, fs, order=4):
    """
    输入np数组，对长维度的每一维进行滤波
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    if data.ndim == 1:
        return filtfilt(b, a, data)

    L, D = data.shape
    if L < D:
        data = data.T
        L, D = data.shape

    filtered_data = np.zeros_like(data)
    for i in range(D):
        filtered_data[:, i] = filtfilt(b, a, data[:, i])

    return filtered_data

# 2. 高通滤波提取交通荷载应变
def high_pass_filter(data, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    if data.ndim == 1:
        return filtfilt(b, a, data)

    L, D = data.shape
    if L < D:
        data = data.T
        L, D = data.shape

    filtered_data = np.zeros_like(data)
    for i in range(D):
        filtered_data[:, i]  = filtfilt(b, a, data[:, i])

    return filtered_data
"""============  ACF估计阻尼比和频率 ========
输入单频率分量的信号，估计自相关函数，然后计算阻尼比
"""


def Auto_ACF(x, dt, max_lag):
    # 压缩维度为1的维度
    x = np.squeeze(x)
    # 首先判断是否有nan
    if np.isnan(x).any() or np.isinf(x).any() or np.max(x)>1e3:
        return 0.0, 0.0

    R2 = np.zeros(max_lag)
    for k in range(max_lag):
        R2[k] = np.sum(x[:len(x) - k] * x[k:])
    
    # 使用Hilbert变换识别频率和阻尼比
    analytic_signal = hilbert(R2)
    envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * dt)

    # 选择0.3到0.7之间的频率部分用于估计
    valid_range_fre = slice(int(0.3 * max_lag), int(0.7 * max_lag))
    valid_range_dp = slice(int(0.3 * max_lag), int(0.7 * max_lag))
    fre2 = np.mean(instantaneous_frequency[valid_range_fre])
    
    # 如果波数即：瞬时频率×max_lag/dt, 波数小于3个波，则视为结果无效，直接输出nan
    num_wave = fre2 * max_lag * dt
    if num_wave < 3.0:
        return 0.0, 0.0

    # 计算阻尼比
    log_envelope = np.log(envelope[valid_range_dp])
    poly_fit = np.polyfit(np.arange(len(log_envelope)) * dt, log_envelope, 1)

    zeta2 = -poly_fit[0] / (fre2 * 2 * np.pi)

    return fre2, zeta2


"""============  SSI-data ===============
数据驱动的随机子空间法
输入Y为np数组
示例 SSI-data
A, C, _, R0 = ssidata(Y, order, s)
f, psi, Phi = modalparams(A, C, dt)
IDs, stable_list, unstable_list = out_stab(A, C, dt)

此外还提供CMIF函数，可用于FDD分析，以及绘制稳定图
"""


def ssidata(Y, order, s):
    """
    Data-based stochastic subspace identification (SSI-data).

    INPUTS:
    Y       : sensor data matrix (ns x nt)
    order   : desired maximum model order to identify (scalar)
    s       : number of block rows in the block Hankel matrix, should be at least ceil(order/ns) to obtain results up to the desired model order,
              generally recommended that s > order

    OUTPUTS:
    A       : list of state transition matrices for model orders {i}
    C       : list of output matrices for model orders {i}
    G       : list of next state output covariance matrices for model orders {i}
    R0      : zero-lag output covariances
    """
    r, c = Y.shape  # ns = number of sensors, nt = number of samples

    # Ensure Y is shaped correctly (samples across rows)
    if r > c:
        Y = Y.T

    ns, nt = Y.shape

    # Forming the shifted data matrix (Hankel matrix)
    Yh = np.zeros((ns * 2 * s, nt - 2 * s + 1))
    for i in range(2 * s):
        Yh[i * ns:(i + 1) * ns, :] = Y[:, i:nt - 2 * s + i + 1]  # Fill in the blocks
    Yh = Yh / np.sqrt(nt - 2 * s + 1)

    # QR decomposition and projection of raw data
    R = qr(Yh.T, mode='r')[0].T
    R = R[:2 * s * ns, :2 * s * ns]
    Proj = R[ns * s:2 * ns * s, :ns * s]

    # Singular value decomposition (SVD)
    U, S, _ = svd(Proj)

    # Zero lag output covariance
    R0 = np.dot(R[ns * s:ns * (s + 1), :], R[ns * s:ns * (s + 1), :].T)

    # Output lists for A, C, G
    A = []
    C = []
    G = []

    # Loop over model orders and generate system matrices
    for i in range(1, order + 1):
        U1 = U[:, :i]
        gam = np.dot(U1, np.diag(np.sqrt(S[:i])))
        gamm = np.dot(U1[:ns * (s - 1), :], np.diag(np.sqrt(S[:i])))
        gam_inv = pinv(gam)
        gamm_inv = pinv(gamm)

        A.append(np.dot(gamm_inv, gam[ns:ns * s, :]))  # State transition matrix
        C.append(gam[:ns, :])  # Output matrix

        delta = np.dot(gam_inv, np.dot(R[ns * s:2 * ns * s, :ns * s], R[:ns * s, :ns * s].T))
        G.append(delta[:, ns * (s - 1):ns * s])  # Next state output covariance matrix

    return A, C, G, R0


def modalparams(A, C, dt):
    """
    Modal decomposition of discrete state-space system.

    INPUTS:
    A       : list of system matrices for model order {i} (each element is a numpy array)
    C       : list of output matrices for model order {i} (each element is a numpy array)
    dt      : sampling period of the discrete system

    OUTPUTS:
    f       : list containing the system pole frequencies in Hz for model orders {i}
    zeta    : list containing the damping ratios of each pole for model orders {i}
    Phi     : list containing the mode shape vectors for model orders {i}

    NOTES:
    (1) Modal scaling (normalization) is not performed.
    (2) Complex conjugate pairs are eliminated and modes are sorted by frequency.
    """

    # Initialize output lists
    f = []
    zeta = []
    Phi = []

    # Loop over model orders
    for i in range(len(A)):
        # Compute eigenvalues (lam) and eigenvectors (v) of A[i]
        lam, v = np.linalg.eig(A[i])

        # Compute modal frequencies (Hz) from eigenvalues
        lam_log = np.log(lam) / dt  # Natural logarithm of the eigenvalues
        freq = np.abs(lam_log) / (2 * np.pi)  # Modal frequencies (Hz)

        # Sort frequencies in ascending order
        sorted_indices = np.argsort(freq)
        freq_sorted = freq[sorted_indices]

        # Modal damping ratios (zeta)
        zeta_vals = -np.real(lam_log) / np.abs(lam_log)
        zeta_sorted = zeta_vals[sorted_indices]

        # Mode shapes
        Phi_vals = np.dot(C[i], v)
        Phi_sorted = Phi_vals[:, sorted_indices]

        # Remove complex conjugate pairs
        unique_freq, unique_indices = np.unique(freq_sorted, return_index=True)
        f.append(unique_freq)
        zeta.append(zeta_sorted[unique_indices])
        Phi.append(Phi_sorted[:, unique_indices])

    return f, zeta, Phi


def mac(phi1, phi2):
    """
    计算模态保证准则（MAC），用于评估两个模态形状向量的相似度
    输入:
    phi1 : 第一个模态形状向量
    phi2 : 第二个模态形状向量

    输出:
    m    : 这两个模态形状向量的模态保证准则值
    """
    # 计算模态保证准则
    m = (np.abs(np.dot(np.conj(phi1).T, phi2))) ** 2 / (np.dot(np.conj(phi1).T, phi1) * np.dot(np.conj(phi2).T, phi2))
    return m.real


def cmif(Y, dt):
    """
    计算复模态指标函数（CMIF），返回跨功率谱的奇异值。

    输入：
    Y      : 输入矩阵，包含系统的输出样本
    win    : 可选的窗口，用于计算跨功率谱密度（CPSD），默认为 None
    dt     : 采样周期，默认为 1.0

    输出：
    SV     : 矩阵，包含每个频率点的跨功率谱的奇异值
    F      : 对应于 SV 第二维的频率向量（单位：Hz）
    U      : 包含奇异值分解的左奇异向量（每个频率的 SVD）
    """

    # 输入数据条件处理，假设样本数多于通道数
    r, c = Y.shape
    if r > c:
        Y = Y.T

    # 如果没有给定窗口，使用矩形窗口（'boxcar'）
    # win = 'boxcar'
    win = signal.windows.hamming(r // 8)  # 选择合适的窗口大小

    nperseg = len(win)  # 让 nperseg 与 win 一致
    noverlap = nperseg // 2  # 与 MATLAB 默认设置一致

    # 获取输出数据的状态数
    ns, nt = Y.shape
    nfft = max(1024, nperseg)

    n_freqs = nfft // 2 + 1

    # 计算跨功率谱（CPSD）
    # csd计算返回的是复数跨谱密度
    Pxy_all = np.zeros((nfft // 2 + 1, ns, ns), dtype=complex)
    for i in range(ns):
        for j in range(i, ns):  # 避免重复计算，使用对称性
            f, Pxy = csd(Y[i], Y[j], fs=1 / dt, window=win, nfft=nfft, nperseg=nperseg, noverlap=noverlap,
                         scaling='density')
            Pxy_all[:, i, j] = Pxy
            if i != j:
                Pxy_all[:, j, i] = Pxy  # 对称性：Pxy(i,j) == Pxy(j,i)

    SV = np.zeros((ns, n_freqs))  # 存储每个频率点的奇异值
    U = np.zeros((ns, ns, n_freqs), dtype=np.complex128)  # 存储每个频率点的奇异向量
    for i in range(n_freqs):
        # 获取第 i 个频率点的交叉功率谱矩阵（二维矩阵）
        Syy_i = Pxy_all[i, :, :]  # Syy 的形状是 (n_channels, n_channels)

        # 对功率谱矩阵进行奇异值分解
        U_i, S_i, _ = np.linalg.svd(Syy_i)

        # 存储奇异值
        SV[:, i] = S_i  # 只取对角线上的奇异值

        # 存储奇异向量
        U[:, :, i] = U_i  # 存储每个频率点的奇异向量

    return SV, f, U


def out_stab(A, C, dt, err=None):
    """
    Creates a stabilization diagram for modal analysis purposes in the current figure.

    INPUTS:
    A       List of system matrices (each element is a numpy array)
    C       List of output matrices (each element is a numpy array)
    Y       Test data used for model identification purposes (numpy array)
    dt      Sampling period of output data y
    win     Optional (replace with None) window to be used for estimation of
            output power spectrums
    err     3-element list of percent errors for stability criteria
            (frequency, damping, and modal assurance criterion),
            default = [0.01, 0.05, 0.98]

    OUTPUT:
    IDs     List of logical arrays (masks) of stable mode indices for each model order
    """
    # Set default error if not provided
    if err is None:
        # print('No stabilization criteria specified, using default settings for stabilization criteria')
        err = [0.01, 0.05, 0.98]

    # Generate modal decompositions
    f, psi, Phi = modalparams(A, C, dt)

    # Loop over model orders
    IDs = []
    stable_list = []
    unstable_list = []
    for i in range(len(A) - 1):
        f1, I1 = np.unique(f[i], return_index=True)
        f2, I2 = np.unique(f[i + 1], return_index=True)

        psi1 = psi[i][I1]
        psi2 = psi[i + 1][I2]
        phi1 = Phi[i][:, I1]
        phi2 = Phi[i + 1][:, I2]

        # Frequency stability criteria
        ef = np.sum(np.abs((f1[:, None] - f2) / f1[:, None]) <= err[0], axis=1).astype(bool)

        # Damping stability criteria
        epsi = np.sum(np.abs((psi1[:, None] - psi2) / psi1[:, None]) <= err[1], axis=1).astype(bool)

        # Modal Assurance Criterion (MAC) stability criteria
        mac_vals = np.zeros(len(f2))
        ephi = np.zeros(len(f1), dtype=bool)

        for j in range(len(f1)):
            for k in range(len(f2)):
                mac_vals[k] = mac(phi1[:, j], phi2[:, k])
            ephi[j] = np.sum(mac_vals >= err[2])

        # Valid (stable) poles
        stable_indices = I1[ef & epsi & ephi]
        stable = f1[ef & epsi & ephi]
        unstable = f1[~(ef & epsi & ephi)]

        # Add stable modes to the list
        IDs.append(stable_indices)
        stable_list.append(stable)
        unstable_list.append(unstable)

        # Optional: plot unstable poles (if needed)

    return IDs, stable_list, unstable_list


def AutoSSI(Y, dt, order, err, order_num):
    # 识别
    if order < 15:
        order = 15
    s = 2 * order
    A, C, _, _ = ssidata(Y, order, s)
    f, psi, Phi = modalparams(A, C, dt)
    IDs, stable_list, unstable_list = out_stab(A, C, dt, err)

    # 聚类
    # 初始化存储数组
    num_stb = 2 * order_num
    stb_fre = np.zeros((20, num_stb))
    stb_dp = np.zeros((20, num_stb))

    for i in range(order - 21, order - 1):
        stb_id = IDs[i]
        min_stb = np.min([len(stb_id), num_stb])
        stb_fre[i - (order - 21), :min_stb] = f[i][stb_id[:min_stb]]
        stb_dp[i - (order - 21), :min_stb] = psi[i][stb_id[:min_stb]]

    # 聚合所有频率和DP值
    flat_freqs = stb_fre.flatten()
    flat_dps = stb_dp.flatten()

    db = DBSCAN(eps=0.06, min_samples=np.floor(10).astype(int))
    idx_clt = db.fit_predict(np.column_stack((flat_freqs, flat_dps)))

    # 计算每个聚类的频率和DP值的均值
    freq_orders = []
    dp_orders = []
    for i in range(np.max(idx_clt) + 1):
        # freq_orders.append(np.mean(flat_freqs[idx_clt == i]))
        # dp_orders.append(np.mean(flat_dps[idx_clt == i]))
        # 取中位数
        freq_orders.append(np.median(flat_freqs[idx_clt == i]))
        dp_orders.append(np.median(flat_dps[idx_clt == i]))

    # 对结果进行排序
    Idx = np.argsort(freq_orders)
    dp_orders = np.array(dp_orders)[Idx]
    freq_orders = np.array(freq_orders)[Idx]
    # 如果第一阶频率和阻尼比都为0，则说明聚类数量不够，需要去除
    if freq_orders[0] == 0 and dp_orders[0] == 0:
        dp_orders = dp_orders[1:]
        freq_orders = freq_orders[1:]

    od_num = np.min([len(freq_orders), order_num])
    # 输出前order_num个模态

    return freq_orders[:od_num], dp_orders[:od_num]


def AutoSSIwithMS(Y, dt, order, err, order_num):
    # 识别
    if order < 15:
        order = 15
    s = 2 * order
    A, C, _, _ = ssidata(Y, order, s)
    f, psi, Phi = modalparams(A, C, dt)
    IDs, stable_list, unstable_list = out_stab(A, C, dt, err)

    # 聚类
    # 初始化存储数组
    num_stb = 2 * order_num
    stb_fre = np.zeros((20, num_stb))
    stb_dp = np.zeros((20, num_stb))

    for i in range(order - 21, order - 1):
        stb_id = IDs[i]
        min_stb = np.min([len(stb_id), num_stb])
        stb_fre[i - (order - 21), :min_stb] = f[i][stb_id[:min_stb]]
        stb_dp[i - (order - 21), :min_stb] = psi[i][stb_id[:min_stb]]

    # 聚合所有频率和DP值
    flat_freqs = stb_fre.flatten()
    flat_dps = stb_dp.flatten()

    db = DBSCAN(eps=0.06, min_samples=np.floor(10).astype(int))
    idx_clt = db.fit_predict(np.column_stack((flat_freqs, flat_dps)))

    # 计算每个聚类的频率和DP值的均值
    freq_orders = []
    dp_orders = []
    for i in range(np.max(idx_clt) + 1):
        # freq_orders.append(np.mean(flat_freqs[idx_clt == i]))
        # dp_orders.append(np.mean(flat_dps[idx_clt == i]))
        # 取中位数
        freq_orders.append(np.median(flat_freqs[idx_clt == i]))
        dp_orders.append(np.median(flat_dps[idx_clt == i]))

    # 对结果进行排序
    Idx = np.argsort(freq_orders)
    dp_orders = np.array(dp_orders)[Idx]
    freq_orders = np.array(freq_orders)[Idx]
    # 如果第一阶频率和阻尼比都为0，则说明聚类数量不够，需要去除
    if freq_orders[0] == 0 and dp_orders[0] == 0:
        dp_orders = dp_orders[1:]
        freq_orders = freq_orders[1:]

    od_num = np.min([len(freq_orders), order_num])
    # 输出前order_num个模态
    # 展开f和phi
    flat_f = np.concatenate(f)
    flat_Phi = np.concatenate(Phi, axis=1)

    Phi_orders = []
    for target_freq in freq_orders[:od_num]:
        idx = np.argmin(np.abs(flat_f - target_freq))  # 找到最接近的索引
        Phi_orders.append(flat_Phi[:, idx].real)

    Phi_orders = np.stack(Phi_orders)
    return freq_orders[:od_num], dp_orders[:od_num], Phi_orders


def plot_stab(Y, dt, stable_list, unstable_list):
    # 绘制CMIF图
    SV, F, U = cmif(Y, dt)
    fig, ax1 = plt.subplots(figsize=(16/2.54, 9/2.54))
    ax1.plot(F, SV[0, :], label=f'Singular Value {1}')
    # 设置左侧 y 轴标签和标题
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Singular Values (CMIF)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 创建右侧 y 轴（用于绘制稳定点）
    ax2 = ax1.twinx()
    # 绘制稳定点
    for i in range(len(stable_list)):
        ax2.scatter(stable_list[i], np.zeros_like(stable_list[i]) + i + 1, label='Stable Point', marker='o',
                    facecolors='none', edgecolors='green')
    # 绘制不稳定点
    for i in range(len(unstable_list)):
        ax2.scatter(unstable_list[i], np.zeros_like(unstable_list[i]) + i + 1, label='Stable Point', marker='x',
                    color='red')

    ax2.set_ylabel('Order', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    # plt.legend()
    plt.tight_layout()
    plt.show()

def rmoutliers_gesd(data, alpha=0.05):
    """
    使用GESD方法检测数据中的离群点，并用NaN填充。
    参数:
    data (array_like): 输入数据，可以是一维列表或NumPy数组。
    alpha (float, 可选): 显著性水平，默认为0.05。
    max_outliers (int, 可选): 最大离群点数，默认根据数据长度自动计算。

    返回:
    tuple: (填充NaN后的数据, 离群点的索引列表)
    """
    import scipy.stats as stats
    max_outliers =int(len(data)/2)
    data = np.asarray(data, dtype=np.float64)  # 确保是浮点型数组
    mask_non_nan = ~np.isnan(data)
    clean_data = data[mask_non_nan]
    original_indices = np.where(mask_non_nan)[0]
    n_clean = len(clean_data)

    if n_clean < 2:
        return data, []  # 数据不足以检测离群点

    if max_outliers is None:
        max_outliers = min(10, n_clean - 1)
    else:
        max_outliers = min(max_outliers, n_clean - 1)

    if max_outliers < 1:
        return data, []

    current_data = clean_data.copy()
    current_indices = np.arange(n_clean)
    outliers_in_clean = []

    for i in range(1, max_outliers + 1):
        m = len(current_data)
        if m < 2:
            break

        mean = np.mean(current_data)
        std = np.std(current_data, ddof=1)
        if std == 0 or np.isnan(std):  # 避免除零错误
            break
        deviations = np.abs(current_data - mean)
        max_dev_idx = np.argmax(deviations)
        max_dev = deviations[max_dev_idx]
        R = max_dev / std

        df = n_clean - i - 1
        if df < 0:
            break
        p = 1 - (alpha / (2 * (n_clean - i + 1)))
        try:
            t = stats.t.ppf(p, df)
        except:
            break

        numerator = (n_clean - i) * t
        denominator = np.sqrt((n_clean - i - 1 + t ** 2) * (n_clean - i + 1))
        lambda_i = numerator / denominator

        if R > lambda_i:
            outlier_clean_idx = current_indices[max_dev_idx]
            outliers_in_clean.append(outlier_clean_idx)
            current_data = np.delete(current_data, max_dev_idx)
            current_indices = np.delete(current_indices, max_dev_idx)
        else:
            break

    outlier_indices = original_indices[outliers_in_clean]
    data_with_nans = data.copy()
    data_with_nans[outlier_indices] = np.nan

    return data_with_nans, outlier_indices.tolist()