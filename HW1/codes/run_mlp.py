from network import Network
from utils import LOG_INFO
from layers import Selu, Swish, Linear, Gelu
from loss import MSELoss, SoftmaxCrossEntropyLoss, HingeLoss, FocalLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import numpy as np
import matplotlib.pyplot as plt

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Linear('fc1', 784, 10, 0.01))

loss = MSELoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.5,
    'weight_decay': 0.0,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 50,
    'test_epoch': 5
}

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

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_metric=test_net(model, loss, test_data, test_label, config['batch_size'])
        test_epochs.append(epoch)
        loss_test.append(test_metric[0])
        acc_test.append(test_metric[1])

fig,axs=plt.subplots(2)

axs[0].plot(train_epochs,loss_train,'-g',label='loss_train')
axs[0].plot(test_epochs,loss_test,':c',label='loss_test')
axs[0].set_xlabel('epoch')
axs[0].set_ylabel('loss')
axs[0].legend()

axs[1].plot(train_epochs,acc_train,label='acc_train')
axs[1].plot(test_epochs,acc_test,label='acc_test')
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('loss')
axs[1].legend()

fig.savefig('result.png')