from layers import Dropout

class Network(object):
    def __init__(self):
        self.layer_list = []
        self.params = []
        self.num_layers = 0
        self.train=True

    def add(self, layer):
        self.layer_list.append(layer)
        self.num_layers += 1

    def forward(self, input):
        output = input
        for i in range(self.num_layers):
            if isinstance(self.layer_list[i],Dropout):
                output=self.layer_list[i].forward(output,self.train)
            else:
                output = self.layer_list[i].forward(output)

        return output

    def backward(self, grad_output, train=True):
        grad_input = grad_output
        for i in range(self.num_layers - 1, -1, -1):
            if isinstance(self.layer_list[i],Dropout):
                grad_input = self.layer_list[i].backward(grad_input,self.train)
            else:
                grad_input = self.layer_list[i].backward(grad_input)

    def update(self, config):
        for i in range(self.num_layers):
            if self.layer_list[i].trainable:
                self.layer_list[i].update(config)
