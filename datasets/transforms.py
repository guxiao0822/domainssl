from torchvision import transforms
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline


def temporal_jitter(time_series, max_shift=5):
    num_channels, num_time_points = time_series.shape
    shift_amount = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(time_series, shift_amount, axis=1)


def temporal_scaling(time_series, scale_range=(0.8, 1.2)):
    num_channels, num_time_points = time_series.shape
    scaling_factor = np.random.uniform(*scale_range)

    # Calculate the new length after scaling
    new_length = int(num_time_points * scaling_factor)

    scaled_series = np.zeros((num_channels, num_time_points), dtype=time_series.dtype)

    for channel in range(num_channels):
        scaled_channel = scipy.signal.resample(time_series[channel, :], new_length)

        # Zero-pad or truncate the time series to match the original length
        if new_length < num_time_points:
            scaled_series[channel, :new_length] = scaled_channel
        else:
            scaled_series[channel, :] = scaled_channel[:num_time_points]

    return scaled_series

def magnitude_warping(time_series, magnitude_range=(0.8, 1.2)):
    num_channels, num_time_points = time_series.shape
    scaling_factor = np.random.uniform(*magnitude_range)
    return time_series * scaling_factor


def add_gaussian_noise(time_series, noise_std=0.1):
    num_channels, num_time_points = time_series.shape
    noise = np.random.normal(0, noise_std, size=(num_channels, num_time_points))
    return time_series + noise


def time_series_cutout(time_series, cutout_ratio=0.25):
    time_series_mask = np.copy(time_series)
    num_channels, num_time_points = time_series.shape
    cutout_length = int(cutout_ratio * num_time_points)
    start = np.random.randint(0, num_time_points - cutout_length)

    time_series_mask[:, start:start + cutout_length] = 0
    return time_series_mask


def time_series_normalization(time_series):
    num_channels, num_time_points = time_series.shape
    time_series_norm = time_series - np.mean(time_series, axis=1, keepdims=True)
    time_series_norm = time_series_norm / (np.std(time_series_norm, axis=1, keepdims=True) + 1e-8)
    # time_series[:, start:start + cutout_length] = 0
    return time_series_norm

def time_series_normalization_mc(time_series):
    mean_all = np.mean(time_series)
    std_all = np.std(time_series) + 1e-8

    # Subtract and divide
    time_series_norm = (time_series - mean_all) / std_all
    return time_series_norm

class Normalize(object):
    def __call__(self, time_series):
        # Normalize each channel separately
        if len(time_series.shape)>1:
            for i in range(time_series.shape[0]):
                channel = time_series[i, :]
                mean = np.mean(channel)
                std = np.std(channel)
                time_series[i, :] = (channel - mean) / (std + 1e-8)  # Avoid division by zero
        else:
            time_series = (time_series - time_series.mean()) / (time_series.std() + 1e-8)
        return time_series




