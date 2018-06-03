# -*- coding: utf-8 -*-

import cv2
import numpy as np

class LDP:
	def set_image(self, image):
		self.image = image
		try:
			self.height, self.width, channel = image.shape
			self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
			self.init_first_derivative = np.zeros((height, width), np.uint8)
			self.init_second_derivative = np.zeros((height, width), np.uint8)
			self.first_derivative = { 0 : np.zeros((height, width), np.uint8), 90: np.zeros((height, width), np.uint8)}
			self.second_derivative = { 0 : np.zeros((height, width), np.uint8), 90: np.zeros((height, width), np.uint8)}
		except:
			print("Image must be class 'numpy.ndarray'")
	
	def __init__(self, image):
		set_image(self, image)
	
	
