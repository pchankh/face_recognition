from model import EigenfacesModel, FisherfacesModel, NeuralNetworkModel
from util import read_images
from split import split_with_proportion

if __name__ == "__main__":
	#image_path = '/home/myxo/face_recognition/data/att_faces/'
	image_path = '/home/myxo/face_recognition/data/yale_a/yalestruct/'
	#image_path = '/home/myxo/face_recognition/data/yale_b_cropped/CroppedYale/'
	[X, y] = read_images(image_path)

	[X_train, y_train, X_test, y_test] = split_with_proportion(X, y, 0.70) 

	#model = EigenfacesModel(X_train, y_train)
	model = NeuralNetworkModel(X_train, y_train)
	model.train(X_train, y_train)
	error_rate = model.test_model(X_test, y_test, print_nonexpected=True)
	print "error rate = %f"%(error_rate) 