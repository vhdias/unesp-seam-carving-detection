import numpy
import scipy
import skimage.feature as sfe
import skimage.filters as sfi
import os.path
import cv2


class ImageDescriptor:
    def __init__(self, image_path, save_images=True):
        self.image_path = image_path
        self.image_name = os.path.basename(self.image_path)
        # Convert image to grayscale
        self.image_array_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.height, self.width = self.image_array_grayscale.shape
        self.size = self.image_array_grayscale.size
        self.image_lbp = self.get_lbp_image(self.image_array_grayscale)
        self.gradient_x, self.gradient_y = self.image_gradient(self.image_lbp)
        self.energy = numpy.abs(self.gradient_x) + numpy.abs(self.gradient_y)
        self.wiener = scipy.signal.wiener(self.image_lbp, 5)
        self.noise = self.image_lbp - self.wiener
        self.vertical_cumulative_energy_transposed = self.minimum_cumulative_energy(self.energy.transpose())
        self.horizontal_cumulative_energy = self.minimum_cumulative_energy(self.energy)
        if save_images:
            path = os.path.join('.', 'processed', 'yin', '{}_' + self.image_name)
            cv2.imwrite(path.format('lbp'), self.image_lbp)
            cv2.imwrite(path.format('gx'), self.gradient_x)
            cv2.imwrite(path.format('gy'), self.gradient_y)
            cv2.imwrite(path.format('wiener'), self.wiener)
            cv2.imwrite(path.format('noise'), self.noise)

    @staticmethod
    def get_lbp_image(image, p=8, r=1.0, method='default'):
        """Get the LBP image from a grayscale image

        Args:
            P: Number of circularly symmetric neighbour set points (quantization of the angular space).
            R: Radius of circle (spatial resolution of the operator).
            method: {‘default’, ‘ror’, ‘uniform’, ‘var’}
                    Method to determine the pattern.
                        ‘default’: original local binary pattern which is gray scale but not rotation invariant.
                        ‘ror’: extension of default implementation which is gray scale and rotation invariant.
                        ‘uniform’: improved rotation invariance with uniform patterns and finer quantization of
                                   the angular space which is gray scale and rotation invariant.
                        ‘nri_uniform’: non rotation-invariant uniform patterns variant which is only gray scale
                                   invariant (http://scikit-image.org/docs/dev/api/skimage.feature.html#r648eb9e75080-2).
                        ‘var’: rotation invariant variance measures of the contrast of local image texture which is
                               rotation but not gray scale invariant.

        Returns:
            LBP image array

        """

        # Generate the LBP from the array_image
        lbp_image = sfe.local_binary_pattern(image, p, r, method)

        return lbp_image

    @staticmethod
    def image_gradient(image):
        x_derivative = sfi.sobel_h(image)
        y_derivative = sfi.sobel_v(image)
        return x_derivative, y_derivative

    @staticmethod
    def minimum_cumulative_energy(energy):
        result = numpy.copy(energy)
        m, n = energy.shape
        for j in range(1, n):
            result[0, j] += min(result[0, j - 1],
                                result[1, j - 1])
            for i in range(1, m - 1):
                result[i, j] += min(result[i - 1, j - 1],
                                    result[i, j - 1],
                                    result[i + 1, j - 1])

            result[m - 1, j] += min(result[m - 2, j - 1],
                                    result[m - 1, j - 1])
        return result
