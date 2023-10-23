# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from typing import OrderedDict
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum=0.9):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features
		self.momentum=momentum

		# Parameters
		self.weight = Parameter(torch.zeros(num_features))
		self.bias = Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter
		init.ones_(self.weight)
		init.zeros_(self.bias)

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.train:
			mean=torch.mean(input=input, dim=(0,2,3))
			var=torch.var(input=input, dim=(0,2,3))

			self.running_mean = (1-self.momentum) * \
                self.running_mean+self.momentum*mean
			self.running_var = (1-self.momentum) * \
                self.running_var+self.momentum*var
			
			std_deviation=torch.sqrt(var+1e-10)
			return (input-mean[:,None,None])/std_deviation[:,None,None]*self.weight[:,None,None]+self.bias[:,None,None]
		else:
			std_deviation=torch.sqrt(self.running_var+1e-10)
			return (input-self.running_mean[:,None,None])/std_deviation[:,None,None]*self.weight[:,None,None]+self.bias[:,None,None]
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		mask=torch.bernoulli(torch.zeros(input.shape[-2:]).to(input.device),1-self.p)/(1-self.p)
		return mask*input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.model=nn.Sequential(OrderedDict([
			('conv1',nn.Conv2d(3,32,kernel_size=3,padding=1)),
			('bn1',BatchNorm2d(32)),
			('relu1',nn.ReLU()),
			('dropout1',Dropout()),
			('mp1',nn.MaxPool2d(kernel_size=(2,2))),
			('conv2',nn.Conv2d(32,64,kernel_size=3,padding=1)),
			('bn',BatchNorm2d(64)),
			('relu2',nn.ReLU()),
			('dropout2',Dropout()),
			('mp2',nn.MaxPool2d(kernel_size=(2,2))),
		]))
		self.fc=nn.Linear(64*8*8,10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		temp=self.model(x)
		logits = self.fc(torch.reshape(temp,(x.shape[0],-1)))
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
