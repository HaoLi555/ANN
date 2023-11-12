python3 main.py --name scratch1
python3 main.py --test scratch1 --decode_strategy random --temperature 1.0 
python3 main.py --test scratch1 --decode_strategy random --temperature 0.7
python3 main.py --test scratch1 --decode_strategy top-p --top_p 0.9 --temperature 1.0
python3 main.py --test scratch1 --decode_strategy top-p --top_p 0.9 --temperature 0.7

python3 main.py --pretrain pretrain --name pretrain
python3 main.py --test pretrain --decode_strategy random --temperature 1.0
python3 main.py --test pretrain --decode_strategy random --temperature 0.7
python3 main.py --test pretrain --decode_strategy top-p --top_p 0.9 --temperature 1.0
python3 main.py --test pretrain --decode_strategy top-p --top_p 0.9 --temperature 0.7