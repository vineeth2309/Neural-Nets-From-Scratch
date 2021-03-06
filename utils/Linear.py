import numpy as np
import copy
from utils.Activation import Activation_Class

class Linear_Layer:
	def __init__(self, input_shape, output_shape, activation="sigmoid", leaky_slope=0.01, learning_rate = 1):
		self.input_dims = input_shape
		self.output_dims = output_shape
		# self.weights = np.random.randn(self.input_dims, self.output_dims) * np.sqrt(2 / self.input_dims + self.output_dims)
		self.weights = np.random.normal(0,np.sqrt(2.0/self.input_dims),(self.input_dims, self.output_dims))
		self.bias = np.zeros((1, self.output_dims))
		self.activation = Activation_Class(name=activation)
		self.dlw = np.zeros((self.input_dims, self.output_dims))
		self.dlb = np.zeros((1,self.output_dims))
		self.data = None
		self.learning_rate = learning_rate
	
	def __call__(self, x):
		self.data = copy.deepcopy(x)	# Store input to the layer for backprop
		x.data = np.matmul(x.data, self.weights) + self.bias
		return self.activation(x)

	def backward(self, x):
		dla = self.activation.backward(x)
		self.dlw = np.matmul(self.data.data.T, dla.grad)
		self.dlb = np.sum(dla.grad, axis=0).reshape(1,-1)
		x.grad = np.matmul(dla.grad, self.weights.T)
		x.data = copy.deepcopy(self.data.data)
		return x
	
	def update(self):
		self.weights -= self.learning_rate * self.dlw
		self.bias -= self.learning_rate * self.dlb

	def clear(self):
		self.dlw = np.zeros((self.input_dims, self.output_dims))
		self.dlb = np.zeros((self.output_dims, 1))
		self.data = None