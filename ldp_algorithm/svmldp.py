# -*- coding: utf-8 -*-

from ldp_algorithm.ldp import LDP
import numpy as np
import tensorflow as tf


class SVMLDP:
	label_feature_list = []
	column_dict = {}
	class_list = []

	def set_edited_list(self, path_list):
		self.edited_path_list = path_list

	def set_not_edited_list(self, path_list):
		self.not_edited_path_list = path_list

	def __init__(self, edited_path_list, not_edited_path_list):
		self.edited_path_list = []
		self.not_edited_path_list = []
		self.metrics = {}
		self.results = {}
		self.set_edited_list(edited_path_list)
		self.set_not_edited_list(not_edited_path_list)
		self.num_train_images = len(self.edited_path_list) + len(self.not_edited_path_list)

		self.svm = tf.contrib.learn.SVM(feature_columns=SVMLDP.label_feature_list, example_id_column='example_id',
		                                model_dir="./output")

	@staticmethod
	def input_fn():
		return SVMLDP.column_dict, SVMLDP.class_list

	@staticmethod
	def predict_fn():
		return SVMLDP.column_dict

	def fill_features(self, image_paths):
		for column, image_path in list(enumerate(image_paths)):
			print("Processing " + image_path + "...")
			image = LDP(image_path)
			image.calculate_features()

			line = 0
			for order in LDP.ORDER_INDEXES:
				for angle in LDP.ANGLE_INDEXES:
					for radius in LDP.RADIUS_INDEXES:
						for byte in range(256):
							self.column_dict[SVMLDP.label_feature_list[line][0]][column] = image.histograms[order, angle, radius][byte]
							line += 1

		for line in range(len(SVMLDP.label_feature_list)):
			self.column_dict[SVMLDP.label_feature_list[line][0]] = tf.constant(
				self.column_dict[SVMLDP.label_feature_list[line][0]])

	def fit(self):
		self.column_dict = {'example_id': tf.constant([str(k) for k in range(1, self.num_train_images + 1)])}
		self.class_list = np.zeros(self.num_train_images, np.uint8)
		self.class_list[:len(self.edited_path_list)] = 1
		self.class_list = tf.constant(self.class_list)

		for feature in SVMLDP.label_feature_list:
			self.column_dict[feature[0]] = np.zeros(self.num_train_images, np.float32)

		image_paths = self.edited_path_list + self.not_edited_path_list

		self.fill_features(image_paths)

		SVMLDP.column_dict = self.column_dict
		SVMLDP.class_list = self.class_list
		self.svm.fit(input_fn=SVMLDP.input_fn)

	def evaluate(self):
		SVMLDP.column_dict = self.column_dict
		SVMLDP.class_list = self.class_list
		self.metrics = self.svm.evaluate(input_fn=SVMLDP.input_fn)
		return self.metrics

	def predict(self, path_list):
		del self.column_dict['example_id']
		self.fill_features(path_list)
		SVMLDP.column_dict = self.column_dict
		self.results = self.svm.predict(input_fn=SVMLDP.predict_fn)
		return self.results


for _order in LDP.ORDER_INDEXES:
	for _angle in LDP.ANGLE_INDEXES:
		for _radius in LDP.RADIUS_INDEXES:
			for _byte in range(256):
				SVMLDP.label_feature_list.append(tf.contrib.layers.real_valued_column('%d_%d_%d_%d' %
				                                                                      (_order, _angle, _radius, _byte)))

calc_svm = SVMLDP(["imagens/tux.jpg", "imagens/minion.jpg"], ["imagens/tigre.jpg", "imagens/lobo.jpg"])
calc_svm.fit()
m = calc_svm.evaluate()
r = calc_svm.predict(["imagens/minion.jpg", "imagens/lobo.jpg"])
print(m)
print(r)
input("Sair")
