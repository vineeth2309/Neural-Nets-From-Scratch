import numpy as np
import cv2
import math
import time
from sklearn import datasets
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils.Loss import loss
from utils.Linear import Linear_Layer
from utils.tensor import tensor
		
class Neural_Network:
	def __init__(self, learning_rate = 1e-4):
		self.layer1 = Linear_Layer(2, 16, activation="leaky_relu", learning_rate = learning_rate)
		self.layer2 = Linear_Layer(16, 2, activation="softmax", learning_rate = learning_rate)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		return x
	
	def backward(self, yhat):
		x = self.layer2.backward(yhat)
		x = self.layer1.backward(x)
	
	def update(self):
		self.layer2.update()
		self.layer1.update()
	
	def clear(self):
		self.layer2.clear()
		self.layer1.clear()

class Main:
	def __init__(self):
		self.X1, self.Y1 = datasets.make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)
		self.plot_data(self.X1,self.Y1)
		self.Y1 = self.one_hot(self.Y1)
		self.X_tensor = tensor(self.X1, requires_grad=True)
		self.Y_tensor = tensor(self.Y1, requires_grad=False)	
		self.net = Neural_Network()
		self.loss = loss("cross_entropy")
		self.forward()
	
	def one_hot(self, Y):
		output_neurons = np.max(Y) + 1
		data_processed = np.zeros((Y.shape[0], output_neurons))
		for i in range(data_processed.shape[0]):
			idx = Y[i]
			data_processed[i][idx] = 1
		Y = data_processed
		del data_processed
		return Y
	
	def un_one_hot(self, Y):
		if type(Y) == tensor:
			Y = Y.data
		return np.argmax(Y,axis=1)

	def plot_data(self,X,Y):
		plt.scatter(X[:,0],X[:,1],c=Y)
		plt.show()

	def forward(self):
		for i in range(1000):
			out = self.net.forward(self.X_tensor)
			loss, out = self.loss(out, self.Y_tensor)
			print("EPOCH {}: {}".format(str(i), str(loss.data)))
			self.net.backward(out)
			self.net.update()
			self.net.clear()
		
		out = self.net.forward(self.X_tensor)
		out = self.un_one_hot(out)
		self.plot_data(self.X1, out)



if __name__ == "__main__":
	Main()