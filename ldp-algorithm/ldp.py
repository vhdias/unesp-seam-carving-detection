# -*- coding: utf-8 -*-

import cv2
import numpy as np

class DirectionalDerivative:
	def __init__(self, bounds, np_type):
		try:
			# Directional derivatives with radius 1 e 2
			self.dir0  = { 1 : np.zeros(bounds, np_type), 2 : np.zeros(bounds, np_type) }
			self.dir90 = { 1 : np.zeros(bounds, np_type), 2 : np.zeros(bounds, np_type) }
		except:
			print("Bounds must be a tuple of int and np_type a numpy type")

class LDP:
	def set_image(self, image):
		self.image = image
		
		try:
			# Shape = image dimensions
			self.height, self.width, channel = image.shape
			# Convert image to grayscale
			self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		except:
			print("Image must be class 'numpy.ndarray'")
	
	def __init__(self, image):
		set_image(self, image)
	
	# Bilinear interpolation
	def bi_inter(x, y):
		pass
		
	def calculate_bin(z0, zi):
		return 0 if z0 * zi > 0 else 1
		
	def calculate_byte(ang, order, radius, i, j):
		matrix, _dir = self.derivative[order - 1], self.ldp[order]
		matrix, _dir = (matrix.dir0, _dir.dir0) if ang == 0 else (matrix.dir90, _dir.dir90)
		matrix, _dir = matrix[radius], _dir[radius][i, j]
		
		for dis_lin, dis_col in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
			_dir << 1
			_dir += calculate_bin( matrix[i, j], matrix[i + dis_lin, j + dis_col] )
		
	def calculate_features(self):
		bounds = (height, width)
		
		# First and second derivative of each pixel
		self.derivative = { 1 : DirectionalDerivative( bounds, np.int16), 2 :  DirectionalDerivative( bounds, np.int16) }
		# LDP
		self.ldp = { 2 : DirectionalDerivative( bounds, np.uint8), 3 : DirectionalDerivative( bounds, np.uint8)}
		
		limits = []
		matrix =  []
		
		# Calculate derivatives
		for order in range(1, 3):
			for radius in range(1, 3):
				limits = [ (1 + order) * radius, radius]
				matrix =  [self.image, self.image] if order == 1 else [self.derivative[1].dir0, self.derivative[1].dir90]
				
				for i in range(limits[0], height - limits[1]):
					for j in range(limits[1], width - limits[0]):
						self.derivative[order].dir0[radius][i, j]  = matrix[0][i, j] - matrix[0][i, j + radius]
						self.derivative[order].dir90[radius][i, j] = matrix[1][i, j] - matrix[1][i - radius, j]
		
		
		
		# Calculate LDPs
		for ldp_order in range(2, 4):
			for radius in range(1, 3):
				limits = [ ldp_order * radius, radius]
				
				for i in range(limits[0], height - limits[1]):
					for j in range(limits[1], width - limits[0]):
						calculate_byte(0, ldp_order, radius, i, j)
						calculate_byte(90, ldp_order, radius, i, j)
											
						
						
