# -*- coding: utf-8 -*-

import cv2
import numpy as np


class LDP:
	ORDER_INDEXES = (1, 2)
	RADIUS_INDEXES = (1, 2)
	ANGLE_INDEXES = (0, 90)
	CORNER_INDEXES = range(1, 5)
	ANGLE_DIS = {0: (0, 1), 90: (-1, 0)}
	SQRT_2 = np.sqrt(2).astype(np.float32)
	SQRT_2_DIV_2 = SQRT_2 / 2
	INTER_POINTS = {
		1: {
			1: (1 - SQRT_2_DIV_2, 1 - SQRT_2_DIV_2),
			2: (1 - SQRT_2_DIV_2, -1 + SQRT_2_DIV_2),
			3: (-1 + SQRT_2_DIV_2, -1 + SQRT_2_DIV_2),
			4: (-1 + SQRT_2_DIV_2, 1 - SQRT_2_DIV_2)},
		2: {
			1: (1 - SQRT_2, 1 - SQRT_2),
			2: (1 - SQRT_2, -1 + SQRT_2),
			3: (-1 + SQRT_2, -1 + SQRT_2),
			4: (-1 + SQRT_2, 1 - SQRT_2)
		}
	}

	def set_image(self, image_path):
		try:
			# Convert image to grayscale
			self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
			# Shape = image dimensions
			self.height, self.width = self.image.shape
		except ValueError:
			print("Image must be class 'numpy.ndarray'")

	# Bilinear interpolation
	@staticmethod
	def bi_inter(matrix, y, x):
		xn = {1: int(x)}
		xn[2] = xn[1] + 1
		yn = {1: int(y)}
		yn[2] = yn[1] + 1
		fat = [(xn[2] - x), (yn[2] - y), (x - xn[1]), (y - yn[1])]

		# In this application, the denominator (xn [2] - xn [1]) * (yn [2] - yn [1]) is always equal to 1
		try:
			return (matrix[yn[1], xn[1]] * fat[0] * fat[1] + matrix[yn[1], xn[2]] * fat[2] * fat[1] +
			        matrix[yn[2], xn[1]] * fat[0] * fat[3] + matrix[yn[2], xn[2]] * fat[2] * fat[3])
		except IndexError:
			return 0

	def __init__(self, image_path):
		self.image = []
		self.height, self.width = 0, 0
		self.set_image(image_path)
		# inter_values/derivatives - values ​​outside the pixel center
		self.derivative, self.ldp, self.inter_values, self.inter_derivatives = {}, {}, {}, {}
		self.histograms = {}
		bounds = (self.height, self.width)

		for radius in LDP.RADIUS_INDEXES:
			for corner in LDP.CORNER_INDEXES:
				self.inter_values[radius, corner] = np.zeros(bounds, np.float32)

				for i in range(self.height):
					for j in range(self.width):
						self.inter_values[radius, corner][i, j] = LDP.bi_inter(self.image, i +
						                                                       LDP.INTER_POINTS[radius][corner][
							                                                       0],
						                                                       j +
						                                                       LDP.INTER_POINTS[radius][corner][
							                                                       1])

		# For ldp, 1 and 2 mean second and third order respectively
		for order in LDP.ORDER_INDEXES:
			for angle in LDP.ANGLE_INDEXES:
				for radius in LDP.RADIUS_INDEXES:
					self.derivative[order, angle, radius] = np.zeros(bounds, np.int16)

					for corner in LDP.CORNER_INDEXES:
						self.inter_derivatives[order, angle, radius, corner] = np.zeros(bounds, np.float32)

					self.ldp[order, angle, radius] = np.zeros(bounds, np.uint8)

	@staticmethod
	def calculate_bin(z0, zi):
		return 0 if int(z0) * int(zi) > 0 else 1

	def calculate_byte(self, order, ang, radius, i, j):
		matrix, _dir = self.derivative[order, ang, radius], self.ldp[order, ang, radius][i, j]

		for corner, dis_inter, dis in [(1, (-1, -1), (-1, 0)), (2, (-1, 1), (0, 1)),
		                               (3, (1, 1), (1, 0)), (4, (1, -1), (0, -1))]:
			inter_matrix = self.inter_derivatives[order, ang, radius, corner]
			_dir <<= 1
			_dir += LDP.calculate_bin(matrix[i, j], inter_matrix[i + dis_inter[0], j + dis_inter[1]])
			_dir <<= 1
			_dir += LDP.calculate_bin(matrix[i, j], matrix[i + dis[0] * radius, j + dis[1] * radius])

		self.ldp[order, ang, radius][i, j] = _dir

	def calculate_features(self):
		# Calculate derivatives
		for order in LDP.ORDER_INDEXES:
			for angle in LDP.ANGLE_INDEXES:
				for radius in LDP.RADIUS_INDEXES:
					limits = [(1 + order) * radius, radius]
					matrix = self.image if order == 1 else self.derivative[order - 1, angle, radius]

					for i in range(limits[0], self.height - limits[1]):
						for j in range(limits[1], self.width - limits[0]):
							self.derivative[order, angle, radius][i, j] = int(matrix[i, j]) - int(matrix[
								                                                                      i + LDP.ANGLE_DIS[
									                                                                      angle][
									                                                                      0] * radius,
								                                                                      j + LDP.ANGLE_DIS[
									                                                                      angle][
									                                                                      1] * radius])

		for order in LDP.ORDER_INDEXES:
			for angle in LDP.ANGLE_INDEXES:
				for radius in LDP.RADIUS_INDEXES:
					for corner in LDP.CORNER_INDEXES:
						limits = [(1 + order) * radius, radius]
						matrix = self.inter_values[radius, corner] if order == 1 else self.inter_derivatives[
							order - 1,
							angle, radius, corner]

						for i in range(limits[0], self.height - limits[1]):
							for j in range(limits[1], self.width - limits[0]):
								try:
									self.inter_derivatives[order, angle, radius, corner][i, j] = \
										matrix[i, j] - matrix[i + LDP.ANGLE_DIS[angle][0] * radius,
										                      j + LDP.ANGLE_DIS[angle][1] * radius]
								except ValueError:
									pass

		# Calculate LDPs
		for order in LDP.ORDER_INDEXES:
			for angle in LDP.ANGLE_INDEXES:
				for radius in LDP.RADIUS_INDEXES:
					limits = [(1 + order) * radius, radius]

					for i in range(limits[0], self.height - limits[1]):
						for j in range(limits[1], self.width - limits[0]):
							self.calculate_byte(order, angle, radius, i, j)

		# Calculate histograms
		for order in LDP.ORDER_INDEXES:
			for angle in LDP.ANGLE_INDEXES:
				for radius in LDP.RADIUS_INDEXES:
					limits = [(1 + order) * radius, radius]
					bounds = [self.height - limits[1] - limits[0], self.width - limits[0] - limits[1]]
					mask = np.zeros((self.height, self.width), np.uint8)
					mask[limits[0]:self.height - limits[1], limits[1]:self.width - limits[0]] = 255

					self.histograms[order, angle, radius] = cv2.calcHist(
						[self.ldp[order, angle, radius]], [0], mask, [256], [0, 256]) / (bounds[0] * bounds[1])

