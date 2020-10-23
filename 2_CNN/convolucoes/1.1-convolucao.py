from PIL import Image
from util import Preprocessador
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import math
import numpy as np

# Imagem
arquivo = "./imagens/goku.jpg"

# Visualiza a imagem no formato original    
imagem = Image.open(arquivo)

print(np.asarray(Image.open(arquivo)).shape)
imagem.show()

# Realiza pre-processamento na imagem
proc = Preprocessador()
imagem = proc.executa(imagem)
# Visualizando o shape da imagem
print("Shape da imagem - [c, h, w]: ", imagem.shape)

# Prepara a imagem para o formato exigido pela convolucao
imagem = imagem.unsqueeze(0)
print("Novo shape - [i, c, h, w]: ", imagem.shape)

# ******************************** Preparacao das funcoes ****************************************
# Conv 1
conv1 = nn.Conv2d(3, 9, kernel_size=11, stride=4, padding=2)
# Func. Relu
re = nn.ReLU(conv1)
# Max. Pool.
m = nn.MaxPool2d(kernel_size=3, stride=2)

# ******************************** Executando as funcoes preparadas ****************************************
# conv -> relu -> max_pool
saida_conv = conv1(imagem)
saida_relu = re(saida_conv)
saida_maxpool = m(saida_relu)

print(sum(p.numel() for p in conv1.parameters() if p.requires_grad))

# Visualizando as saídas
kernels = saida_conv.detach().clone()
# ((n + 2*p - k) / s ) + 1
# ((224 + 2*2 - 11) / 4 ) + 1 =~ 55
print("Shape da kernels - [c, h, w]: ", kernels.shape)
imagens = kernels[0].squeeze()

max_pool_shape = saida_maxpool.detach().clone().shape
print(max_pool_shape)
# Variaveis auxiliares para construcao do grid
colunas = math.ceil(math.sqrt(len(imagens)))
linhas = colunas if len(imagens) % colunas == 0 else colunas-1

fig, axarr = plt.subplots(linhas, colunas,
                       sharex='col', 
                       sharey='row')
i = 0
for linha in range(linhas):
    for coluna in range(colunas):
        if i < len(imagens):
            axarr[linha, coluna].imshow(imagens[i].numpy())
        i+=1

plt.show()

x = torch.flatten(saida_maxpool, 1)

print(x.shape)