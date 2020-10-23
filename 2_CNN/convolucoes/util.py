from torchvision import transforms
import json
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import math

class Preprocessador:
    def __init__(self):
        self.preprocessamento = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
    def executa(self, imagem):
        return self.preprocessamento(imagem)


def apresenta_grid(imagens):
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