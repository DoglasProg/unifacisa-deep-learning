import glob2
import os
from sklearn.model_selection import train_test_split 
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

def obtemTodosOsArquivos(diretorio_base, tipo_de_arquivo, eRecursivo=False):
    regra = "/**/*.{}" if eRecursivo else "/*.{}"
    caminho = diretorio_base + regra.format(tipo_de_arquivo)
    arquivos = glob2.glob(caminho , recursive=eRecursivo)
    return arquivos

def obtemTodosOsDiretorio(diretorio_base, eRecursivo=False):
    return [d for d in os.listdir(diretorio_base) if os.path.isdir(os.path.join(diretorio_base, d))]

def obtemLinhas(arquivo):
    with open(arquivo, "r") as f:
        return [l.strip() for l in f]

def obterLarguraAltura(imagem):
    img = cv2.imread(imagem)
    return img.shape[0], img.shape[1]

def obtemDataSet(arquivo, tamanho=0.30):
    df = obtemDataFrame(arquivo)

    X = df.drop('classe', axis=1)
    y = df.classe

    # Obtem dados para treinamento e dados intermediarios
    X_train, X_inter, y_train, y_inter = train_test_split(X, y, test_size=tamanho)
    
    # Treinamento, Validacao e teste
    return X_train, X_inter, y_train, y_inter     

def obtemNomeDoArquivo(arquivo):
    return Path(arquivo).name

def obtemDataFrame(arquivo):
    return pd.read_csv(arquivo, delimiter=";")

# Para a utilizacao do data Transform
def obtemImagem(imagem):
    return Image.open(imagem).convert('RGB')

def obtemAnotacao(arquivo):
    arquivo = arquivo.replace(".png", ".txt")
    linhas = obtemLinhas(arquivo)
    anotacoes = [item for item in linhas]
    
    box = ""
    for anotacao in anotacoes:
        box = anotacao.split()
    return box

def obtem_coor_x_y_w_h(image, bbox):
    (H, W) = image.shape[:2]
    
    # Get x and y base
    w = float(bbox[2])
    h = float(bbox[3])
    x = float(bbox[0]) 
    y = float(bbox[1])

    # Get x and y base
    x = x - w / 2
    y = y - h / 2

    # Update values according to image dimensions
    x = int(x * W)
    w = int(w * W)
    y = int(y * H)
    h = int(h * H)

    return [x,y,w,h]

# Convert bounding box format from [x, y, w, h] to [y,x,y2,x2]
def converte_xywh_para_xyxy(image, bbox):
    x, y, w, h =  obtem_coor_x_y_w_h(image, bbox)

    x2 = x + w
    y2 = y + h

    return [x,y,x2,y2]


import torchvision.transforms.functional as F

class FixedHeightResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return ImageOps.fit(img, (self.size, self.size), Image.ANTIALIAS)