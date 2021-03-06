import numpy as np


class tensor:
	def __init__(self, x, requires_grad=True):
		self.data = x
		self.requires_grad = requires_grad
		self.shape = x.shape
		self.grad = None