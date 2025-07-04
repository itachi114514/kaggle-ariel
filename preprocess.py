import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dotenv
import torch
import pyarrow.parquet as pq
import torch.nn.functional as F

dotenv.load_dotenv()
base_url = os.getenv("BASE_URL", "/kaggle/input/ariel-data-challenge-2025")

def show_figure(array, title=None, cmap="gray", aspect="auto"):
    image = array
    plt.figure(figsize=(8, 4))
    plt.imshow(image, cmap=cmap, aspect=aspect, vmin=-840, vmax=-660)
    if title:
        plt.title(title)
    plt.xlabel("Spectral direction — 356 px")
    plt.ylabel("Spatial direction — 32 px")
    plt.colorbar(label="Counts")
    plt.tight_layout()
    plt.show()

def reverse_adc(raw_signal:torch.Tensor):
    global AIRS_offset, AIRS_gain
    return raw_signal*AIRS_gain + AIRS_offset

def subtract_dark(raw_signal:torch.Tensor, dark_frame:torch.Tensor):
    """
    Subtract the dark frame from the raw signal.

    Parameters:
    - raw_signal: PyTorch tensor of the raw signal.
    - dark_frame: PyTorch tensor of the dark frame.

    Returns:
    - PyTorch tensor of the processed signal.
    """
    dt = pq.read_table("/mnt/windows/Downloads/ariel-data-challenge-2025/axis_info.parquet")["AIRS-CH0-integration_time"].drop_null().to_numpy()
    # 奇数增加4.5
    dt = np.where(np.arange(len(dt)) % 2 == 0, dt, dt + 4.5)
    dt = torch.tensor(dt,dtype=torch.float32,device="cuda")
    dt = dt.view(-1,1,1)
    return raw_signal - dark_frame* dt


def fix_dead_pixels_vectorized(signal, dead_pixels):
    """
    使用高效的卷积操作替换坏点。

    Args:
        signal (torch.Tensor): 输入信号，形状为 (B, H, W)，例如 (1250, 32, 356)。
        dead_pixels (torch.Tensor): 坏点掩码，形状为 (H, W)，例如 (32, 356)。

    Returns:
        torch.Tensor: 修复后的信号，形状与输入相同。
    """
    # 确保输入是 PyTorch Tensors
    # if not isinstance(signal, torch.Tensor):
    #     signal = torch.tensor(signal, dtype=torch.float32)
    # if not isinstance(dead_pixels, torch.Tensor):
    #     dead_pixels = torch.tensor(dead_pixels, dtype=torch.float32)

    # --- 1. 准备卷积 ---
    # a. 将2D掩码转换为布尔型，并适配3D信号的形状
    # mask 形状: (32, 356) -> (1, 32, 356)，以便广播到 (1250, 32, 356)
    mask = (dead_pixels == 1).unsqueeze(0)

    # b. 创建一个3x3的求和卷积核
    # conv2d需要4D输入: (out_channels, in_channels, kH, kW)
    kernel = torch.ones((1, 1, 3, 3), device=signal.device, dtype=signal.dtype)

    # c. 信号需要是4D: (B, C, H, W)。我们的信号是(B, H, W)，所以增加一个channel维度
    # signal_4d 形状: (1250, 32, 356) -> (1250, 1, 32, 356)
    signal_4d = signal.unsqueeze(1)

    # # --- 2. 计算邻居像素的和 ---
    # # 使用反射填充，与你原始代码的意图一致
    # # padding=1确保卷积后尺寸不变
    # sum_of_9_pixels = F.conv2d(signal_4d, kernel, padding='same', padding_mode='reflect')

    # --- 2. 【核心修正】分开执行填充和卷积 ---

    # 步骤 2a: 手动进行 'reflect' 填充
    # 我们需要在最后两个维度（H和W）的上下左右各填充1个像素
    # pad元组格式: (pad_left, pad_right, pad_top, pad_bottom)
    padded_signal_4d = F.pad(signal_4d, (1, 1, 1, 1), mode='reflect')

    # 步骤 2b: 在已填充的张量上执行卷积，此时 padding 设置为 0 或 'valid'
    # 'valid' 意味着不进行任何填充，这正是我们现在需要的
    sum_of_9_pixels = F.conv2d(padded_signal_4d, kernel, padding='valid')

    # ----------------------------------------------

    # 从4D转回3D，方便后续计算
    sum_of_9_pixels = sum_of_9_pixels.squeeze(1)

    # --- 3. 计算邻居像素的均值 ---
    # 9个像素的和 - 中心像素 = 8个邻居的和
    sum_of_8_neighbors = sum_of_9_pixels - signal
    mean_of_8_neighbors = sum_of_8_neighbors / 8.0

    # --- 4. 使用掩码替换坏点 ---
    # torch.where是最高效、最清晰的条件替换方法
    # torch.where(condition, value_if_true, value_if_false)
    # mask会被自动广播到signal的形状
    fixed_signal = torch.where(mask, mean_of_8_neighbors, signal)

    return fixed_signal

def linearity_correction(signal, linear_corr):
    """
    Perform linearity correction on the signal using polynomial coefficients and Horner's Method.
    Parameters:
    - signal: PyTorch tensor of the raw signal.
    - linear_corr: PyTorch tensor of the polynomial coefficients for linearity correction.
    Returns:
    - PyTorch tensor of the corrected signal.
    """
    # Ensure the input is a PyTorch tensor
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)

    # Reshape linear_corr to match the signal shape
    linear_corr = linear_corr.view(6, 32, 356)

    # Initialize the corrected signal with the first coefficient
    corrected_signal = linear_corr[5]

    # Apply Horner's method for polynomial evaluation
    for i in range(4, -1, -1):
        corrected_signal = corrected_signal * signal + linear_corr[i]

    return corrected_signal

def flat_correction(signal, flat_frame):
    """
    Perform flat field correction on the signal using the flat frame.

    Parameters:
    - signal: PyTorch tensor of the raw signal.
    - flat_frame: PyTorch tensor of the flat frame.

    Returns:
    - PyTorch tensor of the corrected signal.
    """
    # Ensure the input is a PyTorch tensor
    # if not isinstance(signal, torch.Tensor):
    #     signal = torch.tensor(signal, dtype=torch.float32)

    # Normalize the flat frame
    flat_frame_avg = flat_frame.mean()
    epsilon = 1e-8
    normalized_flat = flat_frame / (flat_frame_avg+ epsilon)

    # Apply flat field correction
    corrected_signal = signal / normalized_flat

    return corrected_signal

def process_data(raw_signal, calibration_data):
    """
    Process the raw signal by reversing ADC and subtracting the dark frame.

    Parameters:
    - raw_signal: PyTorch tensor of the raw signal.

    Returns:
    - PyTorch tensor of the processed signal.
    """
    # Reverse ADC
    processed_signal = reverse_adc(raw_signal)

    # Subtract dark frame
    dark_frame = calibration_data['dark']
    # processed_signal = subtract_dark(processed_signal, dark_frame)

    # Substitute dead pixels
    dead_pixels = calibration_data['dead']
    processed_signal = fix_dead_pixels_vectorized(processed_signal, dead_pixels)

    # Perform linearity correction
    linear_corr = calibration_data['linear_corr']
    # processed_signal = linearity_correction(processed_signal, linear_corr)

    # Perform flat field correction
    flat_frame = calibration_data['flat']
    # processed_signal = flat_correction(processed_signal, flat_frame)


    return processed_signal


def read_signal(path):
    """
    Read data from a parquet file and return a PyTorch tensor.
    """
    # Read the parquet file using PyArrow
    arrow_table = pq.read_table(path)

    # Convert to Pandas DataFrame and then to NumPy array
    numpy_array = arrow_table.to_pandas(zero_copy_only=False).to_numpy()

    # Convert to PyTorch tensor
    tensor = torch.tensor(numpy_array, dtype=torch.float32, device="cuda").reshape(-1, 32, 356)

    return tensor

def read_calibration(calibration_base_path):
    """
    Read calibration data from a parquet file and return a PyTorch tensor.
    """
    calibration_data=dict()
    calibration_name = ["dark", "dead", "linear_corr", "flat"]
    for name in calibration_name:
        path = os.path.join(calibration_base_path, f"{name}.parquet")
        # Read the parquet file using PyArrow
        arrow_table = pq.read_table(path)

        # Convert to Pandas DataFrame and then to NumPy array
        numpy_array = arrow_table.to_pandas(zero_copy_only=False).to_numpy()

        # Convert to PyTorch tensor
        tensor = torch.tensor(numpy_array, dtype=torch.float32, device="cuda")
        calibration_data[name] = tensor

    return calibration_data

import time
ID = "1010375142"
adc_info_path = os.path.join(base_url, "adc_info.csv")
adc_info = pd.read_csv(adc_info_path)
FGS1_offset,FGS1_gain,AIRS_offset,AIRS_gain = adc_info.iloc[0,0], adc_info.iloc[0,1],adc_info.iloc[0,2],adc_info.iloc[0,3]
base_url = os.path.join(os.getenv("BASE_URL", "/kaggle/input/ariel-data-challenge-2025"), "train")
example_url = os.path.join(base_url, ID)
raw_signal = read_signal(os.path.join(example_url, "AIRS-CH0_signal_0.parquet"))
calibration_base_path = os.path.join(example_url, "AIRS-CH0_calibration_0")
calibration_data = read_calibration(calibration_base_path)
time0= time.time()
processed_signal = process_data(raw_signal, calibration_data)
time1 = time.time()
print(f"Processing time: {time1 - time0:.2f} seconds")
# Show the processed signal
show_figure(processed_signal[0].cpu().numpy().reshape(32, 356), title="Processed AIRS-CH0 Signal")
# save ten image to file
processed_path = os.path.join(os.getcwd(), f"processed_{ID}")
processed_signal = processed_signal.cpu().numpy()
if not(os.path.exists(processed_path)):
    os.makedirs(processed_path)
for i in range(len(processed_signal)):
    plt.imsave(os.path.join(processed_path,f"processed_signal_{i}.png"), processed_signal[i], cmap='gray')