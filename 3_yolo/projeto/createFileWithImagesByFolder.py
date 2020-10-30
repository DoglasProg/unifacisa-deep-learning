import glob2
import os
import sys
from pathlib import Path

rootPathTrain = "./dataset/train"
rootPathVal = "./dataset/valid"
TRAIN_FILE = "./treino.txt"
VAl_FILE = "./val.txt"


jpg_files = [f for f in glob2.glob(rootPathTrain + "/**/*.jpg", recursive=True)]

with open (TRAIN_FILE, "w") as train_file:
    train_file.write("\n".join(str(item) for item in jpg_files))


    """
    Crie o scrip para gerar traino e validação -> faça isso utilizando métodos e condicional para identidade da classe
    """


    