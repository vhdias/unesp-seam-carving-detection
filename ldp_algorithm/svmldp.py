# -*- coding: utf-8 -*-

from ldp_algorithm.ldp import LDP
import numpy as np
import tensorflow as tf


class SVMLDP:
	label_column_list = []
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

		self.svm = tf.contrib.learn.SVM(feature_columns=SVMLDP.label_column_list, example_id_column='example_id')

	@staticmethod
	def input_fn():
		return SVMLDP.column_dict, SVMLDP.class_list

	@staticmethod
	def predict_fn():
		return SVMLDP.column_dict

	def fit(self):
		self.column_dict = {'example_id': tf.constant([str(k) for k in range(1, self.num_train_images + 1)])}
		self.class_list = np.zeros(self.num_train_images, np.uint8)
		self.class_list[:len(self.edited_path_list)] = 1
		# TODO
		# Generate columns
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
		# TODO
		# Generate columns
		self.results = self.svm.predict(input_fn=SVMLDP.predict_fn)
		return self.results


for order in LDP.ORDER_INDEXES:
	for angle in LDP.ANGLE_INDEXES:
		for radius in LDP.RADIUS_INDEXES:
			SVMLDP.label_column_list.append(tf.contrib.layers.real_valued_column('(%d, %d, %d)' % (order, angle, radius)))
