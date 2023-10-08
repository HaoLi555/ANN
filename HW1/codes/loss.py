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
        # storage
        self.exp=exp
        # one-hot, reduce computational cost (hopefully)
        ln=np.where(target==0,0,np.log(exp/(np.sum(exp,axis=1)[:,np.newaxis])))
        return np.sum(ln*(-target))/input.shape[0]
        # TODO END

    def backward(self, input, target):
        # TODO START
        ratio=self.exp/(np.sum(self.exp,axis=1)[:,np.newaxis])
        return np.where(target==0,ratio,ratio-1)/input.shape[0]
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        # one_index and correct_score (batch_size)
        one_index=np.argmax(target,axis=1)
        correct_score=input[np.arange(0,input.shape[0]),one_index]
        h=np.where(target==1,0,np.maximum(0,self.margin-correct_score[:,np.newaxis]+input))
        # storage
        self.h=h
        return np.sum(h)/input.shape[0]
        # TODO END

    def backward(self, input, target):
        # TODO START
        # the implement should be related to the definition of max
        non_zero_count=np.sum(self.h>0,axis=1)
        return np.where(target==1,-non_zero_count[:,np.newaxis]/input.shape[0],np.where(self.h==0,0,1/input.shape[0]))
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
        exp=np.exp(input)
        sum=np.sum(exp,axis=1)
        self.softmax=exp/sum[:,np.newaxis]
        self.alpha_for_each_sample=np.sum(self.alpha*target,axis=1)
        self.softmax_for_each_sample=np.sum(target*self.softmax,axis=1)
        return -np.sum(self.alpha_for_each_sample*np.power(1-self.softmax_for_each_sample,self.gamma)*np.log(self.softmax_for_each_sample))/len(input)
        # TODO END

    def backward(self, input, target):
        # TODO START
        factor_for_each_sample=-self.alpha_for_each_sample*np.power(1-self.softmax_for_each_sample,self.gamma-1)*(1-self.softmax_for_each_sample+self.gamma*self.softmax_for_each_sample*np.log(self.softmax_for_each_sample))
        return np.where(target==0,factor_for_each_sample[:,np.newaxis]*(-self.softmax),factor_for_each_sample[:,np.newaxis]*(1-self.softmax))/len(input)
        # TODO END
