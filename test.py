# import numpy as np
# import matplotlib.pyplot as plt

# train_epochs=np.arange(0,100)
# test_epochs=np.arange(0,100,5)
# loss_train=np.random.randn(100)
# acc_train=np.random.randn(100)
# loss_test=np.random.randn(20)
# acc_test=np.random.randn(20)


# fig,ax=plt.subplots(2)

# ax[0].plot(train_epochs,loss_train,'-g',label='loss_train')
# ax[0].plot(test_epochs,loss_test,':c',label='loss_test')
# # 注意这里只能使用set_xlabel不能使用xlabel
# ax[0].set_xlabel('epoch')
# ax[0].set_ylabel('loss')

# ax[1].plot(train_epochs,acc_train,label='acc_train')
# ax[1].plot(test_epochs,acc_test,label='acc_test')
# ax[1].set_xlabel('epoch')
# ax[1].set_ylabel('loss')

# ax[0].legend()
# ax[1].legend()

# plt.subplots_adjust(hspace=0.5)
# plt.show()

# fig.savefig('adasd.png')

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-d',action="store_true")
args=parser.parse_args()


if args.d:
    print("yes")
else:
    print("no")