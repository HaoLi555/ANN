# 修改说明

## layers.py

- 代码填空实现各个激活函数以及线性层
- 多出的修改：给部分激活函数定义一些成员用来表示计算中的常量系数
- 添加Dropout层

## loss.py

- 完成基础的损失函数以及bonus的focal loss

## run_mlp.py

自行修改以完成以下功能

- 用命令行参数确定模型（层数、激活函数、损失函数、是否dropout）
- 绘制图像
- 保存最终结果至文本文件

## solve_net.py

- 为train和test对应的函数增加返回值（acc和loss的列表），用于后面绘制图线
- train, test对应的函数中分别把model.train设为真，假

## network.py

- 修改forward, backward函数以适应dropout（训练与非训练模式的差异）