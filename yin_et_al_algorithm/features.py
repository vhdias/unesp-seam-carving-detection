import numpy as np
import collections


def get_features_Ryu_Lee(image_descriptor):
    # Detecting Trace of Seam Carving for Forensic Analysis; 2014
    result = collections.OrderedDict()

    # 4 features based on average energy
    # Table 1
    average_column_energy = np.abs(image_descriptor.gradient_x).mean()
    average_row_energy = np.abs(image_descriptor.gradient_y).mean()
    average_energy = image_descriptor.energy.mean()
    average_energy_difference = np.abs(np.abs(image_descriptor.gradient_x) -
                                       np.abs(image_descriptor.gradient_y)).mean()

    # 10 features based on the vertical and horizontal seam energy
    # Table 2
    width_vertical = image_descriptor.vertical_cumulative_energy_transposed.shape[1]
    width_horizontal = image_descriptor.horizontal_cumulative_energy.shape[1]
    vertical_seams = image_descriptor.vertical_cumulative_energy_transposed[:, width_vertical - 1]
    horizontal_seams = image_descriptor.horizontal_cumulative_energy[:, width_horizontal - 1]

    vertical_seam_max = np.max(vertical_seams)
    vertical_seam_min = np.min(vertical_seams)
    vertical_seam_mean = np.mean(vertical_seams)
    vertical_seam_std = np.std(vertical_seams)
    vertical_seam_diff = vertical_seam_max - vertical_seam_min
    horizontal_seam_max = np.max(horizontal_seams)
    horizontal_seam_min = np.min(horizontal_seams)
    horizontal_seam_mean = np.mean(horizontal_seams)
    horizontal_seam_std = np.std(horizontal_seams)
    horizontal_seam_diff = horizontal_seam_max - horizontal_seam_min

    # 4 features based on the noise level
    # Table 3
    noise_mean = image_descriptor.noise.mean()  # feature
    noise_less_mean = image_descriptor.noise - noise_mean
    noise_standard_deviation = noise_less_mean.std()  # feature
    noise_less_mean_divided_std = noise_less_mean / noise_standard_deviation
    noise_skewness = (noise_less_mean_divided_std ** 2).mean()  # feature
    noise_kurtosis = (noise_less_mean_divided_std ** 3).mean()  # feature

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
    width_vertical = image_descriptor.vertical_cumulative_energy_transposed.shape[1]
    width_horizontal = image_descriptor.horizontal_cumulative_energy.shape[1]
    half_vertical_seams = image_descriptor.vertical_cumulative_energy_transposed[:, round((width_vertical - 1) / 2)]
    half_horizontal_seams = image_descriptor.horizontal_cumulative_energy[:, round((width_horizontal - 1) / 2)]
    vertical_half_seam_max = np.max(half_vertical_seams)
    vertical_half_seam_min = np.min(half_vertical_seams)
    vertical_half_seam_mean = np.mean(half_vertical_seams)
    horizontal_half_seam_max = np.max(half_horizontal_seams)
    horizontal_half_seam_min = np.min(half_horizontal_seams)
    horizontal_half_seam_mean = np.mean(half_horizontal_seams)

    result[image_descriptor.image_name] = {
        'vertical_half_seam_max': vertical_half_seam_max,
        'vertical_half_seam_min': vertical_half_seam_min,
        'vertical_half_seam_mean': vertical_half_seam_mean,
        'horizontal_half_seam_max': horizontal_half_seam_max,
        'horizontal_half_seam_min': horizontal_half_seam_min,
        'horizontal_half_seam_mean': horizontal_half_seam_mean
    }

    return result
