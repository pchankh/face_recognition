import sys
# append tinyfacerec to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacerec modules
from tinyfacerec.util import read_images
from tinyfacerec.model import EigenfacesModel, NeuronNetworkModel
from numpy import genfromtxt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pyplot as plt

import random

from split import split_with_proportion

if __name__ == '__main__':

    #image_path = '/home/myxo/face_recognition/data/yale_a/yalestruct/'
    image_path = '/home/myxo/face_recognition/data/att_faces/'
    #image_path = '/home/myxo/face_recognition/data/yale_b_cropped/CroppedYale/'
    [X, y] = read_images(image_path)

    [X_train, y_train, X_test, y_test] = split_with_proportion(X, y, 0.70) 
    print "splited"
    #[X,y] = read_images("~/face_recognition/data/att_faces/")
    # model = NeuronNetworkModel(X_train, y_train, 100, 200)
    # [error, test_error] = model.compute(X, y, X_test, y_test)

    # plt.plot(error)
    # plt.plot(test_error)
    # plt.show()
    # print error
    # print test_error
    model = NeuronNetworkModel(X_train, y_train, 70, 400)
    error_rate = model.test_model(X_test, y_test)
    #print "iter = %d\t hsize = 70\t error = %f"%(iter, error_rate)

    print "\nerror rate = " + str(error_rate) + "%"
