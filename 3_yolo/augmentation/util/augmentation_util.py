
import cv2
import numpy as np
import pandas as pd

arquivos = pd.read_csv("dataset_sem_classes.csv", delimiter=";")

diretorio = "../data/completo"

def salvar(imagem , path):
    """Salva uma imagem no diretório fornecido

    Args:
        imagem ([float]): Imagem para modificação
        path ([type]): diretório onde será salfo a imagem
    """
    cv2.imwrite(f"{diretorio}/{path}", imagem)
    cv2.waitKey(0)

def redimencionar(imagem, arquivo):
    """Redimenciona uma imagem para o formato 416

    Args:
        imagem (float): Imagem para modificação
        arquivo (str): Path do arquivo

    Returns:
        [float]: Imagem modificada
    """
    (H, W) = imagem.shape[:2]
    size = 416
    f = H if H >= W else W
    r = size/H if H >= W else size/W
    dim = (size, int(f * r))

    # Redimensionamento com base na largura
    imagem_largura = cv2.resize(imagem, dim, interpolation = cv2.INTER_AREA)
    path = arquivo[0].split("/")

    salvar(imagem_largura, f"redi_{path[3]}-{path[4]}")
    return imagem_largura

def flip_image(imagem, arquivo, tipo_flip = 1):
    """Método que realiza o flip em imagem

    Args:
        imagem (float): Imagem para modificação
        arquivo (str): Path do arquivo
        tipo_flip (int, optional): Tipo do flip a ser feito, valor 0 para flip vertical,
        valor 1 para flip horizontal e -1 para flip vertical e horizontal. Defaults to 1.
    
    Returns:
        [float]: Imagem modificada
    """
    imagem_com_flip = cv2.flip(imagem, tipo_flip)
    path = arquivo[0].split("/")

    salvar(imagem_com_flip, f"flip_{tipo_flip}_{path[3]}-{path[4]}")
    return imagem_com_flip

try:
    for idx, arquivo in arquivos.iterrows():
       # Carregando a imagem
        imagem = cv2.imread(arquivo[0])
        cv2.waitKey(0)

        # redimencionar
        redimencionar(imagem=imagem, arquivo=arquivo)

        """
        Desafio ->  Adicione o método de flip
        OBS: teste se deu certo realmente...

        """
        
except:
    logging.error("Erro ao processar criacao de arquivo de dados.")
    sys.exit()


 
