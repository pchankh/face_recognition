import numpy as np
from feature_extractor import PCA, LDA
from classifier import EDistance, NeuralNetwork


class BaseModel(object):
	def __init__(self, X=None, y=None):
		pass

	def train(self, X, y):
		raise NotImplementedError("Every BaseModel must implement the compute method.")
		
	def predict(self, X):
		raise NotImplementedError("Every BaseModel must implement the predict method.")

	def test_model(self, test_set, test_set_lable, print_nonexpected=None):
		error = 0
		for i, point in enumerate(test_set):
			predicted = self.predict(point)
			if predicted != test_set_lable[i]:
				error += 1
				if print_nonexpected:
					print "label - %d\t predicted - %d"%(test_set_lable[i], predicted)
		return float(error) / len(test_set_lable) * 100


class EigenfacesModel(BaseModel):

	def __init__(self, X=None, y=None, feature_extr=PCA, classifier=EDistance):
		#super(EigenfacesModel, self).__init__(X=X,y=y,dist_metric=dist_metric,num_components=num_components)
		self.feature_extr = feature_extr()
		self.classifier = classifier(X, y)

	def train(self, X, y):
		self.feature_extr.compute_feature_space(X)

	def predict(self, X):
		projected = self.feature_extr.project(X)
		return self.classifier.predict(self.feature_extr.projections, projected)


class FisherfacesModel(BaseModel):

	def __init__(self, X=None, y=None, feature_extr=LDA, classifier=EDistance):
		#super(EigenfacesModel, self).__init__(X=X,y=y,dist_metric=dist_metric,num_components=num_components)
		self.feature_extr = feature_extr()
		self.classifier = classifier(X, y)

	def train(self, X, y):
		self.feature_extr.compute_feature_space(X, y)

	def predict(self, X):
		projected = self.feature_extr.project(X)
		return self.classifier.predict(self.feature_extr.projections, projected)


class NeuralNetworkModel(BaseModel):
	def __init__(self, X=None, y=None, feature_extr=LDA, classifier=NeuralNetwork):
		self.feature_extr = feature_extr()
		self.classifier = classifier(X, y)
		self.X = X
		self.y = y

	def train(self, X, y):
		self.feature_extr.compute_feature_space(X, y)
		self.classifier.train(self.feature_extr.projections, self.y)

	def predict(self, X):
		projected = self.feature_extr.project(X)
		return self.classifier.predict(projected)