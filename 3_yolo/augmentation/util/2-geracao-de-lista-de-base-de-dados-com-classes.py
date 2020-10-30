import util

import argparse
import logging
from tqdm import tqdm
import sys
import datetime
import os

# Variaveis
diretorio_base = "../data/por_classe"
dataset = "./dataset_com_classes.csv"
tipo = "jpg"

# Inicia log
logging.basicConfig(level=logging.DEBUG)

# Obtem informacoes para avaliar o tempo de processamento
inicio = datetime.datetime.now()
logging.info("Iniciando processo em: {}".format(inicio))
try:
    # Obtem todos os diretorio apartir de um diretorio base
    diretorios = util.obtemTodosOsDiretorio(diretorio_base)
    # Cria um conjunto de classes a partir dos de um diretorio base
    classes = [diretorio for diretorio in diretorios]
    
    with open (dataset, "w") as arquivo_de_dados:
        for i, classe in enumerate(classes):
            # Para cada classe, obtenha os dados correspondentes
            diretorio = os.path.join(diretorio_base,classe)
            
            arquivos = util.obtemTodosOsArquivos(diretorio, tipo, False)
            print(arquivos)
            for _, arquivo in tqdm(enumerate(arquivos), total=len(arquivos)): # Cria barra de progressao
                
                h, w = util.obterLarguraAltura(arquivo)
                arquivo = util.obtemNomeDoArquivo(arquivo)
                # Formacao ( caminho, w, h, classe)
                arquivo_de_dados.write("{};{};{};{}\n".format(arquivo,w, h, i))
except:
    logging.error("Erro ao processar criacao de arquivo de dados.")
    sys.exit()

# Fim de processo
fim = datetime.datetime.now()
tempo_de_processamento = (fim-inicio).total_seconds()
logging.info("Tempo de processamento: {}".format(str(tempo_de_processamento)))