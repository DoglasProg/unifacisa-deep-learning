import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import util

import argparse
import logging
from tqdm import tqdm
import datetime

# Variaveis
diretorio = "../data/por_classe"
tipo = "jpg"
arquivos = util.obtemTodosOsArquivos(diretorio,tipo, True)
dataset = "./dataset_sem_classes.csv"

# Inicia log
logging.basicConfig(level=logging.DEBUG)

print(len(arquivos))

if len(arquivos) > 0:
    # Obtem informacoes para avaliar o tempo de processamento
    inicio = datetime.datetime.now()
    logging.info("Iniciando processo em: {}".format(inicio))
    try:
        with open (dataset, "w") as arquivo_de_dados:
            for index, arquivo in tqdm(enumerate(arquivos), total=len(arquivos)): # Cria barra de progressao
                arquivo_de_dados.write("{}\n".format(arquivo))
    except:
        logging.error("Erro ao processar criacao de arquivo de dados.")
        sys.exit()

    # Fim de processo
    fim = datetime.datetime.now()
    tempo_de_processamento = (fim-inicio).total_seconds()
    logging.info("Tempo de processamento: {}".format(str(tempo_de_processamento)))

