import numpy as np
from distance import EuclideanDistance

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

class Classifier(object):

	def __init__():
		pass

	def train():
		pass

	def predict():
		pass


class Distance(Classifier):
	def __init__(self, dist_metric, X, y):
		self.dist_metric = dist_metric()
		self.X = X
		self.y = y

	def train(self):
		pass


	def predict(self, train_set, X):
		minDist = np.finfo('float').max
		minClass = -1
		for i, x_i in enumerate(train_set):
			dist = self.dist_metric(x_i, X)
			if dist < minDist:
				minDist = dist
				minClass = self.y[i]
		return minClass


class EDistance(Distance):
	def __init__(self, X, y):
		super(EDistance, self).__init__(EuclideanDistance, X, y)


class NeuralNetwork(Classifier):
	def __init__(self, X=None, y=None, hidden_size=100, num_of_epochs=600):
		self.hidden_size = hidden_size
		self.num_of_epochs = num_of_epochs
		self.X = X
		self.y = y

	def train(self, X, y):
		self.ds = ClassificationDataSet(len(X[0][0]), 1, nb_classes=max(y)+1)
		#print self.projections
		for i, proj in enumerate(X):
			self.ds.addSample(proj[0], y[i])

		self.ds._convertToOneOfMany( )
		
		self.net = buildNetwork(self.ds.indim, self.hidden_size, self.ds.outdim, outclass=SoftmaxLayer)
		trainer = BackpropTrainer(self.net, self.ds)
		error = 0
		for i in range(self.num_of_epochs):
			error = trainer.train()

		return error

	def predict(self, X):
		indim = len(X[0])
		test = ClassificationDataSet(indim, 1, nb_classes=max(self.y) + 1)
		test.addSample(X[0], [0])
		test._convertToOneOfMany( )
		#print 'checkpoint'
		#print test.indim, test.outdim, self.net.indim, self.net.outdim
		prediction = self.net.activateOnDataset(test)
		#print prediction
		#return round(prediction.index(max(prediction[0])))
		return round(np.argmax(prediction[0]))