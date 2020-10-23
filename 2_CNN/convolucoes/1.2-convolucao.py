from PIL import Image
from util import Preprocessador
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import math
from util import apresenta_grid

# Imagem
imagem_url = "./imagens/goku.jpg"

# Visualiza a imagem no formato original    
imagem = Image.open(imagem_url)
#imagem.show()

# Realiza pre-processamento na imagem
proc = Preprocessador()
imagem = proc.executa(imagem)

# Visualizando o shape da imagem
print("Shape da imagem - [c, h, w]: ", imagem.shape)

# Prepara a imagem para o formato exigido pela convolucao
imagem = imagem.unsqueeze(0)
print("Novo shape - [i, c, h, w]: ", imagem.shape)

# ******************************** Preparacao das funcoes ****************************************
#  CAMADA 1
conv_1 = nn.Conv2d(3, 32, kernel_size=11, stride=2, padding=2)
re_1 = nn.ReLU()
m_1 = nn.MaxPool2d(kernel_size=3, stride=2)

#  CAMADA 2
conv_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2)
re_2 = nn.ReLU()
m_2 = nn.MaxPool2d(kernel_size=3, stride=2)

#  CAMADA 3
conv_3 = nn.Conv2d(64, 32, kernel_size=5, padding=2, stride=2)
re_3 = nn.ReLU()
m_3 = nn.MaxPool2d(kernel_size=3, stride=2)


# ******************************** Executando as funcoes preparadas ****************************************
# 1
saida_conv1 = conv_1(imagem)
print(saida_conv1.shape)
saida_relu1 = re_1(saida_conv1)
print(saida_relu1.shape)
saida_maxpool1 = m_1(saida_relu1)
print(saida_maxpool1.shape)

# 2
saida_conv2 = conv_2(saida_maxpool1)
print(saida_conv2.shape)
saida_relu2 = re_2(saida_conv2)
saida_maxpool2 = m_2(saida_relu2)
print(saida_maxpool2.shape)


# 3
saida_conv3 = conv_3(saida_maxpool2)
print(saida_conv3.shape)
saida_relu3 = re_3(saida_conv3)
saida_maxpool3 = m_3(saida_relu3)
print(saida_maxpool3.shape)


# Obtem as imagens
kernels = saida_conv1.detach().clone()
imagens = kernels[0].squeeze()

# Apresenta imagens
apresenta_grid(imagens)


# OBS: Calculo dos parametros aprendidos
# (n*m*l+1)*k 
# n e m = dimensoes
# l = camada
# k = saida
