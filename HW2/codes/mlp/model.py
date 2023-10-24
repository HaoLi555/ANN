# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from typing import OrderedDict


class BatchNorm1d(nn.Module):
    # TODO START
    def __init__(self, num_features, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum

        # Parameters
        self.weight = Parameter(torch.ones(self.num_features))
        self.bias = Parameter(torch.zeros(self.num_features))

        # Store the average mean and variance
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # Initialize your parameter

    def forward(self, input):
        # input: [batch_size, num_feature_map * height * width]
        if self.train:
            mean = torch.mean(input=input, dim=0)
            var = torch.var(input=input, dim=0)

            self.running_mean = (1-self.momentum) *self.running_mean+self.momentum*mean
            self.running_var = (1-self.momentum) *self.running_var+self.momentum*var

            std_deviation = torch.sqrt(var+1e-10)
            return ((input-mean)/std_deviation)*self.weight+self.bias
        else:
            std_deviation = torch.sqrt(self.running_var+1e-10)
            return ((input-self.running_mean)/std_deviation)*self.weight+self.bias

    # TODO END


class Dropout(nn.Module):
    # TODO START
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input):
        # input: [batch_size, num_feature_map * height * width]
        if self.train:         
            mask=torch.bernoulli(torch.zeros(input.shape,device=input.device), 1-self.p)/(1-self.p)
            return input*mask
        else:
            return input

    # TODO END


class Model(nn.Module):
    def __init__(self, drop_rate=0.5):
        super(Model, self).__init__()
        # TODO START
        # Define your layers here
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(3*32*32, 1024)),
            ('bn', BatchNorm1d(1024)),
            ('relu', nn.ReLU()),
            ('dropout',Dropout(drop_rate)),
            ('fc2', nn.Linear(1024, 10))
        ]))
        # TODO END
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # TODO START
        # the 10-class prediction output is named as "logits"
        logits = self.model(x)
        # TODO END

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        loss = self.loss(logits, y)
        correct_pred = (pred.int() == y.int())
        # Calculate the accuracy in this mini-batch
        acc = torch.mean(correct_pred.float())

        return loss, acc
