from network import Network
from utils import LOG_INFO
from layers import Selu, Swish, Linear, Gelu, Relu, Sigmoid
from loss import MSELoss, SoftmaxCrossEntropyLoss, HingeLoss, FocalLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import numpy as np
import matplotlib.pyplot as plt
import argparse

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture

loss_map={
    'M':MSELoss,
    'S':SoftmaxCrossEntropyLoss,
    'H':HingeLoss,
    'F':FocalLoss
}

activation_map={
    'R':Relu,
    'Si':Sigmoid,
    'Se':Selu,
    'Ge':Gelu,
    'Sw':Swish
}

parser=argparse.ArgumentParser()
parser.add_argument('-n',type=int,choices=[1,2,3],default=1,help='number of layers')
parser.add_argument('-l',type=str,choices=['M','S','H','F'],default='M',help="loss function")
parser.add_argument('-a',type=str,choices=['R','Si','Se','Ge','Sw'],default='R',help="actiation function")

args=parser.parse_args()

def one_layer_net(args):
    model=Network()
    model.add(Linear('fc1', 784, 10, 0.01))
    model.add(activation_map[args.a](args.a))
    loss=loss_map[args.l]('loss')
    return model,loss

def two_layers_net(args):
    model=Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(activation_map[args.a](args.a))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(activation_map[args.a](args.a))
    loss=loss_map[args.l]('loss')
    return model,loss

def three_layers_net(args):
    model=Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(activation_map[args.a](args.a))
    model.add(Linear('fc2', 256, 49, 0.01))
    model.add(activation_map[args.a](args.a))
    model.add(Linear('fc3', 49, 10, 0.01))
    model.add(activation_map[args.a](args.a))
    loss=loss_map[args.l]('loss')
    return model,loss

net_map={
    1:one_layer_net,
    2:two_layers_net,
    3:three_layers_net
}

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0007,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 200,
    'test_epoch': 5
}

model,loss=net_map[args.n](args)

loss_train=[]
acc_train=[]
loss_test=[]
acc_test=[]
train_epochs=np.arange(0,config['max_epoch'])
test_epochs=[]

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_metric=train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    loss_train.append(train_metric[0])
    acc_train.append(train_metric[1])

    if (epoch+1) % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_metric=test_net(model, loss, test_data, test_label, config['batch_size'])
        test_epochs.append(epoch)
        loss_test.append(test_metric[0])
        acc_test.append(test_metric[1])

final_training=f"Final training loss: {loss_train[-1]}, final training acc: {acc_train[-1]}"
final_test=f"Final test loss: {loss_test[-1]}, final test acc: {acc_test[-1]}"

print(final_training)
print(final_test)

fig,axs=plt.subplots(2)

axs[0].plot(train_epochs,loss_train,'-g',label='loss_train')
axs[0].plot(test_epochs,loss_test,':k',label='loss_test')
axs[0].set_xlabel('epoch')
axs[0].set_ylabel('loss')
axs[0].legend()

axs[1].plot(train_epochs,acc_train,'-g',label='acc_train')
axs[1].plot(test_epochs,acc_test,':k',label='acc_test')
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('acc')
axs[1].legend()

plt.subplots_adjust(hspace=0.5)

name=f"{args.n}_{args.a}_{args.l}"
fig.savefig(name+'_result.png')

with open('results.txt','a') as f:
    f.write(name+'\n')
    f.write(final_training+'\n')
    f.write(final_test+'\n')
    f.write('\n')