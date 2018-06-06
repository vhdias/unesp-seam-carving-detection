# -*- coding: utf-8 -*-

import cv2
import numpy as np


class LDP:
	ORDER_INDEXES = (1, 2)
	RADIUS_INDEXES = (1, 2)
	ANGLE_INDEXES = (0, 90)
	ANGLE_DIS = {0: (0, 1), 90: (-1, 0)}

	def set_image(self, image):
		self.image = image

		try:
			# Shape = image dimensions
			self.height, self.width, channel = image.shape
			# Convert image to grayscale
			self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		except ValueError:
			print("Image must be class 'numpy.ndarray'")

	def __init__(self, image):
		self.image = None
		self.height, self.width = 0, 0
		self.set_image(image)
		self.derivative, self.ldp = {}, {}
		bounds = (self.height, self.width)

		# For ldp, 1 and 2 mean second and third order respectively
		for order in (1, 2):
			for angle in (0, 90):
				for radius in (1, 2):
					self.derivative[order, angle, radius] = np.zeros(bounds, np.int16)
					self.ldp[order, angle, radius] = np.zeros(bounds, np.uint8)

	# Bilinear interpolation
	def bi_inter(self, x, y):
		pass

	@staticmethod
	def calculate_bin(z0, zi):
		return 0 if z0 * zi > 0 else 1

	def calculate_byte(self, ang, order, radius, i, j):
		matrix, _dir = self.derivative[order, ang, radius], self.ldp[order, ang, radius][i, j]

		for dis_lin, dis_col in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
			_dir <<= 1
			_dir += LDP.calculate_bin(matrix[i, j], matrix[i + dis_lin, j + dis_col])

	def calculate_features(self):
		# Calculate derivatives
		for order in LDP.ORDER_INDEXES:
			for angle in LDP.ANGLE_INDEXES:
				for radius in LDP.RADIUS_INDEXES:
					limits = [(1 + order) * radius, radius]
					matrix = self.image if order == 1 else self.derivative[order - 1, angle, radius]

					for i in range(limits[0], self.height - limits[1]):
						for j in range(limits[1], self.width - limits[0]):
							self.derivative[order, angle, radius][i, j] = matrix[i, j] - matrix[
								i + LDP.ANGLE_DIS[angle][0] * radius,
								j + LDP.ANGLE_DIS[angle][1] * radius]

		# Calculate LDPs
		for order in LDP.ORDER_INDEXES:
			for angle in LDP.ANGLE_INDEXES:
				for radius in LDP.RADIUS_INDEXES:
					limits = [(1 + order) * radius, radius]

					for i in range(limits[0], self.height - limits[1]):
						for j in range(limits[1], self.width - limits[0]):
							self.calculate_byte(order, angle, radius, i, j)
