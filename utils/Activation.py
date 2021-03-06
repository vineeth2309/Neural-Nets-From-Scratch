import numpy as np
from utils.tensor import tensor

class Activation_Class:
	def __init__(self, name='sigmoid', leaky_slope=0.01):
		self.name = name
		self.leaky_slope = 0.01

	def __call__(self, x):
		if self.name is not None:
			if self.name == "sigmoid":
				x.data = 1 / (1 + np.exp(-x.data))
				return x
			elif self.name == "relu":
				x.data = np.maximum(x.data, 0)
				return x
			elif self.name == "tanh":
				x.data = (np.exp(x.data) - np.exp(-x.data)) / (np.exp(x.data) + np.exp(-x.data))
				return x
			elif self.name == "leaky_relu":
				x.data = np.maximum(x.data, self.leaky_slope * x.data)
				return x
			elif self.name =="softmax":
				x.data = np.exp(x.data) / np.sum(np.exp(x.data),axis=1).reshape(-1,1)
				return x
		else:
			return x
	
	def backward(self, x):
		if self.name == "sigmoid":
			x.grad = np.multiply(x.grad , x.data* (1 - x.data))
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
			x.grad = np.multiply(x.grad , x.data * (1 - x.data))
			return x