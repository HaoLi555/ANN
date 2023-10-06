import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Relu(Layer):
	def __init__(self, name):
		super(Relu, self).__init__(name)

	def forward(self, input):
		self._saved_for_backward(input)
		return np.maximum(0, input)

	def backward(self, grad_output):
		input = self._saved_tensor
		return grad_output * (input > 0)

class Sigmoid(Layer):
	def __init__(self, name):
		super(Sigmoid, self).__init__(name)

	def forward(self, input):
		output = 1 / (1 + np.exp(-input))
		self._saved_for_backward(output)
		return output

	def backward(self, grad_output):
		output = self._saved_tensor
		return grad_output * output * (1 - output)

class Selu(Layer):
    def __init__(self, name):
        super(Selu, self).__init__(name)
        self.lbd=1.0507
        self.alpha=1.67326
        self.mul=self.lbd*self.alpha

    def forward(self, input):
        # TODO START
        output=np.where(input>0,self.lbd*input,self.mul*(np.exp(input)-1))
        self._saved_for_backward(output)     
        return output   
        # TODO END

    def backward(self, grad_output):
        # TODO START
        output=self._saved_tensor
        return np.where(output>0,grad_output*self.lbd,grad_output*(input+self.mul))
        # TODO END

class Swish(Layer):
    def __init__(self, name):
        super(Swish, self).__init__(name)

    def forward(self, input):
        # TODO START
        output=input/(1+np.exp(-input))
        self._saved_for_backward(np.vstack((input,output)))
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        input=self._saved_tensor[0]
        output=self._saved_tensor[1]
        ratio=output/input
        return grad_output*(ratio+output*(1-ratio))
        # TODO END

class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)
        self.alpha=np.sqrt(2/np.pi)
        self.beta=0.044715

    def forward(self, input):
        # TODO START
        x=self.alpha*(input+self.beta*np.power(input,3))
        e_x=np.exp(x)
        e_neg_x=-e_x
        denominator=e_x+e_neg_x
        y=1+(e_x-e_neg_x)/denominator
        self._saved_for_backward(np.vstack(input,y,denominator))
        return 0.5*input*y
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        input=self._saved_tensor[0]
        y=self._saved_tensor[1]
        denominator=self._saved_tensor[2]
        return grad_output*0.5*(y+input*(4/np.power(denominator,2))*self.alpha*(1+3*self.beta*np.power(input,2)))
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        return input@self.W+self.b
        # TODO END

    def backward(self, grad_output):
        # TODO START
        # grad_output (batch_size, out_dim), input (batch_size, in_dim)
        input=self._saved_tensor
        self.grad_W=np.matmul(input.T,grad_output)
        self.grad_b=np.sum(grad_output,axis=0)
        return grad_output@self.W.T
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
