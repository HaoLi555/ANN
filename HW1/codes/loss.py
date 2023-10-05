from __future__ import division
import numpy as np


class MSELoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        # input (bs, n), target (bs, n)
        return np.sum(np.power(input-target,2))/input.shape[0]
        # TODO END

    def backward(self, input, target):
		# TODO START
        return (input-target)*2/input.shape[0]
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        exp=np.exp(input)
        # one-hot, reduce computational cost
        ln=np.where(target==0,0,np.log(exp/(np.sum(exp,axis=1)[:,np.newaxis])))
        return np.sum(ln*(-target))/input.shape[0]
        # TODO END

    def backward(self, input, target):
        # TODO START
        exp=np.exp(input)
        ratio=exp/(np.sum(exp,axis=1)[:,np.newaxis])
        return np.where(target==0,ratio,ratio-1)/input.shape[0]
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        '''Your codes here'''
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END


# Bonus
class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=2.0):
        self.name = name
        if alpha is None:
            self.alpha = [0.1 for _ in range(10)]
        self.gamma = gamma

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END
