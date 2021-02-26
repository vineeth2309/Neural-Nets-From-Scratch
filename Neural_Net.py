import numpy as np
import cv2
import math
import time


class tensor:
	def __init__(self, x, requires_grad=True):
		self.data = x
		self.requires_grad = requires_grad
		self.shape = x.shape
		self.grad = None
		
class Linear:
	def __init__(self, input_shape, output_shape, activation="sigmoid", leaky_slope=0.01):
		self.input_dims = input_shape
		self.output_dims = output_shape
		self.weights = np.random.randn(self.input_dims, self.output_dims)
		self.bias = np.zeros((self.output_dims, 1))
		self.activation = Activation(name=activation)
		self.dlw = np.zeros((self.input_dims, self.output_dims))
		self.dlb = np.zeros((self.output_dims, 1))
		self.dl = None
	
	def __call__(self, x):
		self.backward(x)
		return tensor(self.activation(np.matmul(x.data, self.weights) + self.bias.T))

	def backward(self, x):
		if x.grad == None:
			print("HERE")

class Activation:
	def __init__(self, name='sigmoid', leaky_slope=0.01):
		self.name = name
		self.leaky_slope = 0.01

	def __call__(self, x):
		if self.name is not None:
			if self.name == "sigmoid":
				return 1 / (1 + np.exp(-x))
			elif self.name == "relu":
				return np.maximum(x, 0)
			elif self.name == "tanh":
				return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
			elif self.name == "leaky_relu":
				return np.maximum(x, -self.leaky_slope * x)
			elif self.name =="softmax":
				return np.exp(x) / sum(np.exp(x)[0])
		else:
			return None
	
	def backward(self, x):
		pass

class Neural_Network:
	def __init__(self):
		self.net = {}
		self.layer1 = Linear(2, 16, activation="sigmoid")
		self.layer2 = Linear(16, 32, activation="sigmoid")
		self.layer3 = Linear(32, 64, activation="sigmoid")
		self.layer4 = Linear(64, 2, activation="softmax")

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		return x

class Main:
	def __init__(self):
		self.X = tensor(np.ones((10, 2)), requires_grad=True)
		self.net = Neural_Network()
		print(self.net.forward(self.X).data)
		

if __name__ == "__main__":
	Main()