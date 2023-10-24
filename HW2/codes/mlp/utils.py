import matplotlib.pyplot as plt


def plot(train_epochs,valid_epochs,loss_train,loss_valid,acc_train,acc_valid,name):
    fig,axs=plt.subplots(2)

    axs[0].plot(train_epochs,loss_train,'-g',label='loss_train')
    axs[0].plot(valid_epochs,loss_valid,':k',label='loss_test')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].legend()

    axs[1].plot(train_epochs,acc_train,'-g',label='acc_train')
    axs[1].plot(valid_epochs,acc_valid,':k',label='acc_test')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('acc')
    axs[1].legend()

    plt.subplots_adjust(hspace=0.5)

    fig.savefig(name)

def save_result(name,test_acc):
    with open('results.txt','a') as f:
        f.write(name+'\n')
        f.write('test_acc: '+test_acc+'\n')
        f.write('\n')