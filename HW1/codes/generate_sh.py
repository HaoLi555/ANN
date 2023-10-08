with open('/home/haoooooooooooookkkkk/sophomore_autumn/ANN/HW1/run.sh','w') as f:
    for i in ['M','S','H','F']:
        for j in ['R','Si','Se','Ge','Sw']:
            f.write(f"python3 run_mlp.py -n 3 -a {j} -l {i}\n")