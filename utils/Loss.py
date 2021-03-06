import numpy as np
from utils.tensor import tensor

class loss:
	def __init__(self, name="cross_entropy"):
		self.name = name
	
	def __call__(self, yhat , y):
		""" yhat-> model prediction, y -> label"""
		loss = tensor(np.zeros_like(yhat.data))
		assert yhat.data.shape == y.data.shape
		if self.name == "cross_entropy":
			loss.data = np.mean(np.nan_to_num(np.sum(-((y.data * np.log(yhat.data)) + ((1-y.data) * np.log(1 - yhat.data))), axis=1)))
			if yhat.requires_grad:
				loss.grad = (yhat.data - y.data) / ((yhat.data * (1 - yhat.data)) + 1e-8)
				yhat.grad = (yhat.data - y.data) / ((yhat.data * (1 - yhat.data)) + 1e-8)

		elif self.name == "mse":
			loss.data = np.mean((np.sum((yhat.data - y.data)**2, axis=1)) / y.shape[1])
			if yhat.requires_grad:
				loss.grad = yhat.data - y.data
				yhat.grad = yhat.data - y.data

		elif self.name == "rmse":
			loss.data = np.mean((np.sum((yhat.data - y.data)**2, axis=1) ** 0.5) / y.shape[1])
			if yhat.requires_grad:
				loss.grad = (yhat.data - y.data)**0.5
				yhat.grad = (yhat.data - y.data)**0.5
		return loss, yhat