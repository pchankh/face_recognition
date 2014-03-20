from subspace import pca, lda, fisherfaces, project
from util import asRowMatrix


class FeatureExtractor(object):
	def compute_feature_space():
		pass

	def project(x):
		pass



class PCA(FeatureExtractor):
	def __init__(self, num_components=0):
		self.num_components = num_components
		self.projections = []
		self.W = []
		self.mu = []

	def compute_feature_space(self, X):
		[D, self.W, self.mu] = pca(asRowMatrix(X), self.num_components)

		for xi in X:
			self.projections.append(project(self.W, xi.reshape(1,-1), self.mu))

	def project(self, X):
		return project(self.W, X.reshape(1,-1), self.mu)

class LDA(FeatureExtractor):
	def __init__(self, num_components=0):
		self.num_components = num_components
		self.projections = []
		self.W = []
		self.mu = []

	def compute_feature_space(self, X, y):
		[D, self.W, self.mu] = fisherfaces(asRowMatrix(X), y, self.num_components)

		for xi in X:
			self.projections.append(project(self.W, xi.reshape(1,-1), self.mu))

	def project(self, X):
		return project(self.W, X.reshape(1,-1), self.mu)

