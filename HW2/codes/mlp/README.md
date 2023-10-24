# 代码修改
- `main.py` 
  - 使用 wandb 绘图
  - 加入更多命令行参数，便于进行Batch Normalization, Dropout实验以及记录实验结果到wandb
- `models.py` 
  - 增加一个args的初始化参数，便于调整网络结构（是否使用BN, Dropout）

# 运行示例

`python3 mlp/main.py --drop_rate 0.6 --run_name mlp --num_epochs 100`
