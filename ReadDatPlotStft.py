import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


HEADER_BYTES = 176
DTYPE = np.float32
FS = 250e6

NUM_SEGMENTS = 20
SEGMENT_NUMBER = 10

WINDOW_LENGTH = 4096
OVERLAP_LENGTH = WINDOW_LENGTH // 2
NFFT = WINDOW_LENGTH

FIG_WIDTH = 6.5
FIG_HEIGHT = 4.5
SAVE_DPI = 800

CMAP = "jet"
NORMALIZE_DB = False

OUTPUT_DIR = Path("./3 数据集绘图")

DATA_FILES = [
    Path(r"G:\2025-02-27无人机数据采集\2025-03-05无人机数据采集-操场\101-1.dat"),
    Path(r"G:\2025-02-27无人机数据采集\2025-03-05无人机数据采集-操场\200-1.dat"),
    Path(r"G:\2025-02-27无人机数据采集\2025-03-05无人机数据采集-操场\300-1.dat"),
    Path(r"G:\2025-02-27无人机数据采集\2025-03-05无人机数据采集-操场\400-1.dat"),
    Path(r"G:\2025-02-27无人机数据采集\2025-03-05无人机数据采集-操场\500-1.dat"),
    Path(r"G:\2025-02-27无人机数据采集\2025-03-05无人机数据采集-操场\非大疆无人机\80-1.dat"),
]


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": SAVE_DPI,
})


def read_iq_segment(
    file_path: Path,
    num_segments: int,
    segment_number: int,
) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not 1 <= segment_number <= num_segments:
        raise ValueError(
            f"SEGMENT_NUMBER 必须位于 1 至 {num_segments} 之间，"
            f"当前值为 {segment_number}"
        )

    file_size = file_path.stat().st_size

    if file_size <= HEADER_BYTES:
        raise ValueError(
            f"文件大小异常，无法跳过 {HEADER_BYTES} 字节文件头: {file_path}"
        )

    available_bytes = file_size - HEADER_BYTES
    float_count = available_bytes // np.dtype(DTYPE).itemsize

    # 保证数据数量为偶数，以便配对为 I/Q 数据
    float_count -= float_count % 2

    if float_count < 2:
        raise ValueError(f"文件中没有有效的 I/Q 数据: {file_path}")

    complex_count = float_count // 2
    segment_length = complex_count // num_segments

    if segment_length < WINDOW_LENGTH:
        raise ValueError(
            f"分段后的数据长度为 {segment_length}，"
            f"小于 STFT 窗长 {WINDOW_LENGTH}: {file_path}"
        )

    segment_index = segment_number - 1
    complex_start = segment_index * segment_length

    if segment_number == num_segments:
        complex_end = complex_count
    else:
        complex_end = complex_start + segment_length

    float_start = complex_start * 2
    float_end = complex_end * 2

    mapped_data = np.memmap(
        file_path,
        dtype=DTYPE,
        mode="r",
        offset=HEADER_BYTES,
        shape=(float_count,),
    )

    interleaved_iq = np.asarray(
        mapped_data[float_start:float_end],
        dtype=np.float32,
    )

    i_data = interleaved_iq[0::2]
    q_data = interleaved_iq[1::2]

    min_length = min(len(i_data), len(q_data))

    complex_signal = (
        i_data[:min_length].astype(np.float32)
        + 1j * q_data[:min_length].astype(np.float32)
    )

    del mapped_data

    return complex_signal.astype(np.complex64, copy=False)


def calculate_stft(complex_signal: np.ndarray):
    window = signal.windows.hamming(
        WINDOW_LENGTH,
        sym=False,
    )

    frequencies, times, stft_data = signal.stft(
        complex_signal,
        fs=FS,
        window=window,
        nperseg=WINDOW_LENGTH,
        noverlap=OVERLAP_LENGTH,
        nfft=NFFT,
        detrend=False,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    # 将零频率移动到频率轴中心
    frequencies = np.fft.fftshift(frequencies)
    stft_data = np.fft.fftshift(stft_data, axes=0)

    # 转换为功率谱并计算 dB
    power = np.abs(stft_data) ** 2
    power_db = 10.0 * np.log10(
        power + np.finfo(np.float32).eps
    )

    if NORMALIZE_DB:
        power_db = power_db - np.max(power_db)

    return frequencies, times, power_db


def plot_and_save_stft(
    frequencies: np.ndarray,
    times: np.ndarray,
    power_db: np.ndarray,
    file_path: Path,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    lower_limit = np.percentile(power_db, 1.0)
    upper_limit = np.percentile(power_db, 99.8)

    fig = plt.figure(
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        frameon=False,
    )

    # 坐标轴占满整张图片
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(
        power_db,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=CMAP,
        vmin=lower_limit,
        vmax=upper_limit,
        rasterized=True,
    )

    # 移除横纵坐标、刻度、标题和边框
    ax.set_axis_off()

    output_path = output_dir / (
        f"{file_path.stem}_segment_{SEGMENT_NUMBER}_STFT.png"
    )

    fig.savefig(
        output_path,
        dpi=SAVE_DPI,
        format="png",
        bbox_inches=None,
        pad_inches=0,
        transparent=False,
        facecolor="white",
    )

    plt.close(fig)

    return output_path


def process_single_file(
    file_path: Path,
    output_dir: Path,
):
    print(f"\n正在处理: {file_path}")

    complex_signal = read_iq_segment(
        file_path=file_path,
        num_segments=NUM_SEGMENTS,
        segment_number=SEGMENT_NUMBER,
    )

    duration = len(complex_signal) / FS

    print(f"复数采样点数: {len(complex_signal):,}")
    print(f"片段持续时间: {duration:.6f} s")
    print("正在计算 STFT...")

    frequencies, times, power_db = calculate_stft(
        complex_signal
    )

    print(f"STFT 矩阵尺寸: {power_db.shape}")
    print("正在保存无坐标、无标题的 800 dpi PNG...")

    output_path = plot_and_save_stft(
        frequencies=frequencies,
        times=times,
        power_db=power_db,
        file_path=file_path,
        output_dir=output_dir,
    )

    print(f"图片已保存: {output_path}")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful_files = []
    failed_files = []

    for file_path in DATA_FILES:
        try:
            process_single_file(
                file_path,
                OUTPUT_DIR,
            )
            successful_files.append(file_path)

        except Exception as error:
            failed_files.append(
                (file_path, str(error))
            )

            print(f"处理失败: {file_path}")
            print(f"错误信息: {error}")

    print("\n" + "=" * 70)
    print(
        f"处理完成，成功生成 "
        f"{len(successful_files)} 张 STFT 图片。"
    )
    print(f"输出目录: {OUTPUT_DIR.resolve()}")

    if failed_files:
        print(
            f"\n处理失败的文件数量: "
            f"{len(failed_files)}"
        )

        for file_path, error_message in failed_files:
            print(f"文件: {file_path}")
            print(f"原因: {error_message}")


if __name__ == "__main__":
    main()