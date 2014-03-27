import random

import numpy as np
import cPickle

import sys
from util import read_images


def split_with_proportion(X, y, p):
	"""
	
	Split set [X, y] to train and test, with proportion p (for train).
	p should be in [0,1]

	"""
	size = len(X)
	X_train = []
	y_train = []
	X_test = []
	y_test = []

	train_size = int(size*p)
	index = range(size)
	random.shuffle(index)
	for i in index[0:train_size]:
		X_train.append(X[i])
		y_train.append(y[i])

	for i in index[train_size:-1]:
		X_test.append(X[i])
		y_test.append(y[i])

	return [X_train, y_train, X_test, y_test]


def cPickle_image_save(path, filename):
	[X, y] = read_images(path, sz=(28,28))
	x_new = matrixAsVector(X)
	y_new = np.asarray(y, dtype=np.float32)
	[x_train, y_train, x_, y_] = split_with_proportion(x_new, y_new, 0.5)
	[x_valid, y_valid, x_test, y_test] = split_with_proportion(x_, y_, 0.5)
	obj = ((x_train, y_train),(x_valid, y_valid),(x_test, y_test))
	f = file(filename, 'wb')
	cPickle.dump(obj, f, cPickle.HIGHEST_PROTOCOL)
	f.close()

def matrixAsVector(X):
	x_new = []
	for x in X:
		x_new.append(x.flatten(1))
	return np.asarray(x_new, dtype=np.float32)