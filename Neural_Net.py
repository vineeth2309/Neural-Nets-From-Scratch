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
		
class loss:
	def __init__(self, name="cross_entropy"):
		self.name = name
	
	def __call__(self, yhat , y):
		""" yhat-> model prediction, y -> label"""
		loss = tensor(np.zeros_like(yhat.data)) 
		if self.name == "cross_entropy":
			loss.data = np.mean(np.sum(-((y.data * np.log(yhat.data)) + ((1-y.data) * np.log(1 - yhat.data))), axis=1))
			if yhat.requires_grad:
				loss.grad = (yhat.data - y.data) / (yhat.data * (1 - yhat.data))  
				yhat.grad = (yhat.data - y.data) / (yhat.data * (1 - yhat.data))  

		elif self.name == "mse":
			loss.data = np.mean((np.sum((yhat.data - y.data)**2, axis=1)) / y.data.shape[1])
			if yhat.requires_grad:
				loss.grad = yhat.data - y.data
				yhat.grad = yhat.data - y.data

		elif self.name == "rmse":
			loss.data = np.mean((np.sum((yhat.data - y.data)**2, axis=1) ** 0.5) / y.data.shape[1])
			if yhat.requires_grad:
				loss.grad = (yhat.data - y.data)**0.5
				yhat.grad = (yhat.data - y.data)**0.5
		return loss, yhat

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
				return np.maximum(x, self.leaky_slope * x)
			elif self.name =="softmax":
				return np.exp(x) / sum(np.exp(x)[0])
		else:
			return None
	
	def backward(self, x):
		if self.name == "sigmoid":
			x.grad = x.grad * (x.data * (1 - x.data))
			return x 
		elif self.name == "relu":
			x.grad = x.grad * (x.data > 0) * 1
			return x
		elif self.name == "leaky_relu":
			dx = np.ones_like(x.data)
			dx[x.data < 0] = self.leaky_slope
			x.grad = x.grad * dx
			return x
		elif self.name =="tanh":
			x.grad = x.grad * (1 - (x.data**2))
			return x
		elif self.name == "softmax":
			x.grad = x.grad * (x.data * (1 - x.data))
			return x

class Linear:
	def __init__(self, input_shape, output_shape, activation="sigmoid", leaky_slope=0.01):
		self.input_dims = input_shape
		self.output_dims = output_shape
		self.weights = np.random.randn(self.input_dims, self.output_dims)
		self.bias = np.zeros((1, self.output_dims))
		self.activation = Activation(name=activation)
		self.dlw = np.zeros((self.input_dims, self.output_dims))
		self.dlb = np.zeros((self.output_dims, 1))
		self.data = None
		self.h_past = None
	
	def __call__(self, x):
		self.data = x
		return tensor(self.activation(np.matmul(x.data, self.weights) + self.bias))

	def backward(self, x):
		dla = self.activation.backward(x)
		self.dlw = np.matmul(self.data.data.T, dla.grad)
		self.dlb = dla.grad
		x.grad = np.matmul(dla.grad, self.weights.T)
		x.data = self.data.data
		return x
		
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
	
	def backward(self, yhat):
		x = self.layer4.backward(yhat)
		x = self.layer3.backward(x)
		x = self.layer2.backward(x)
		x = self.layer1.backward(x)

class Main:
	def __init__(self):
		self.X = tensor(np.ones((10, 2)), requires_grad=True)
		self.Y = tensor(np.zeros((10,2)), requires_grad=False)		
		self.net = Neural_Network()
		self.loss = loss("cross_entropy")
		self.forward()
	
	def forward(self):
		out = self.net.forward(self.X)
		loss, out = self.loss(out, self.Y)
		self.net.backward(out)


if __name__ == "__main__":
	Main()