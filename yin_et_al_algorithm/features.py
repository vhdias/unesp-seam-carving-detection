import numpy as np
import collections


def get_features_Ryu_Lee(image_descriptor):
    # Detecting Trace of Seam Carving for Forensic Analysis; 2014
    result = collections.OrderedDict()

    # 4 features based on average energy
    # Table 1
    average_column_energy = np.abs(image_descriptor.gradient_x).sum() / image_descriptor.size
    average_row_energy = np.abs(image_descriptor.gradient_y).sum() / image_descriptor.size
    average_energy = image_descriptor.energy.sum() / image_descriptor.size
    average_energy_difference = np.abs(np.abs(image_descriptor.gradient_x) -
                                       np.abs(image_descriptor.gradient_y)).sum() / image_descriptor.size

    # 10 features based on the vertical and horizontal seam energy
    # Table 2
    vertical_seam_max = np.max(
        image_descriptor.vertical_cumulative_energy_transposed[:, image_descriptor.height - 1])
    vertical_seam_min = np.min(
        image_descriptor.vertical_cumulative_energy_transposed[:, image_descriptor.height - 1])
    vertical_seam_mean = np.mean(
        image_descriptor.vertical_cumulative_energy_transposed[:, image_descriptor.height - 1])
    vertical_seam_std = np.std(
        image_descriptor.vertical_cumulative_energy_transposed[:, image_descriptor.height - 1])
    vertical_seam_diff = vertical_seam_max - vertical_seam_min
    horizontal_seam_max = np.max(image_descriptor.horizontal_cumulative_energy[:, image_descriptor.width - 1])
    horizontal_seam_min = np.min(image_descriptor.horizontal_cumulative_energy[:, image_descriptor.width - 1])
    horizontal_seam_mean = np.mean(image_descriptor.horizontal_cumulative_energy[:, image_descriptor.width - 1])
    horizontal_seam_std = np.std(image_descriptor.horizontal_cumulative_energy[:, image_descriptor.width - 1])
    horizontal_seam_diff = horizontal_seam_max - horizontal_seam_min

    # 4 features based on the noise level
    # Table 3
    noise_mean = image_descriptor.noise.sum() / image_descriptor.size  # feature
    noise_less_mean = image_descriptor.noise - noise_mean
    noise_standard_deviation = noise_less_mean.std()  # feature
    noise_less_mean_divided_std = noise_less_mean / noise_standard_deviation
    noise_skewness = (noise_less_mean_divided_std ** 2).sum() / image_descriptor.size  # feature
    noise_kurtosis = (noise_less_mean_divided_std ** 3).sum() / image_descriptor.size  # feature

    result[image_descriptor.image_name] = {
        'average_column_energy': average_column_energy,
        'average_energy': average_energy,
        'average_energy_difference': average_energy_difference,
        'average_row_energy': average_row_energy,
        'vertical_seam_max': vertical_seam_max,
        'vertical_seam_min': vertical_seam_min,
        'vertical_seam_mean': vertical_seam_mean,
        'vertical_seam_std': vertical_seam_std,
        'vertical_seam_diff': vertical_seam_diff,
        'horizontal_seam_max': horizontal_seam_max,
        'horizontal_seam_min': horizontal_seam_min,
        'horizontal_seam_mean': horizontal_seam_mean,
        'horizontal_seam_std': horizontal_seam_std,
        'horizontal_seam_diff': horizontal_seam_diff,
        'noise_mean': noise_mean,
        'noise_standard_deviation': noise_standard_deviation,
        'noise_kurtosis': noise_kurtosis,
        'noise_skewness': noise_skewness
    }

    return result


def get_features_half_seam(image_descriptor):
    # Detecting seam carving based image resizing using local binary patterns; 2015
    # Ting Yin, Gaobo Yang, Leida Li, Dengyong Zhang, Xingming Sun
    result = collections.OrderedDict()

    # 6 features based on the vertical and horizontal seam energy
    # Table 1
    vertical_half_seam_max = np.max(
        image_descriptor.vertical_cumulative_energy_transposed[0:round(image_descriptor.width / 2),
                                                               image_descriptor.height - 1])
    vertical_half_seam_min = np.min(
        image_descriptor.vertical_cumulative_energy_transposed[0:round(image_descriptor.width / 2),
                                                               image_descriptor.height - 1])
    vertical_half_seam_mean = np.mean(
        image_descriptor.vertical_cumulative_energy_transposed[0:round(image_descriptor.width / 2),
                                                               image_descriptor.height - 1])
    horizontal_half_seam_max = np.max(
        image_descriptor.horizontal_cumulative_energy[0:round(image_descriptor.height / 2), image_descriptor.width - 1])
    horizontal_half_seam_min = np.min(
        image_descriptor.horizontal_cumulative_energy[0:round(image_descriptor.height / 2), image_descriptor.width - 1])
    horizontal_half_seam_mean = np.mean(
        image_descriptor.horizontal_cumulative_energy[0:round(image_descriptor.height / 2), image_descriptor.width - 1])

    result[image_descriptor.image_name] = {
        'vertical_half_seam_max': vertical_half_seam_max,
        'vertical_half_seam_min': vertical_half_seam_min,
        'vertical_half_seam_mean': vertical_half_seam_mean,
        'horizontal_half_seam_max': horizontal_half_seam_max,
        'horizontal_half_seam_min': horizontal_half_seam_min,
        'horizontal_half_seam_mean': horizontal_half_seam_mean
    }

    return result
