import os
import random

cwd=os.getcwd()
files_and_directories=os.listdir(cwd)

txts=[file for file in files_and_directories if file.endswith('.txt')]

with open("selected_sentences.txt",'w') as f:
    for txt in txts:
        with open(txt,'r') as r:
            lines=r.readlines()
            selected=random.sample(lines,10)
            
            temp=txt.split('_')
            model="Tfmr Scratch" if temp[1].startswith('scratch') else "Tfmr Finetune"
            decode_strategy="random" if temp[2].startswith('random') else "top-p"
            temperature=temp[3][:3]

            f.write(f"Model: {model}, Decode Strategy: {decode_strategy}, Temperature: {temperature}\n")

            f.writelines(selected)
            f.write('\n')
            