from datetime import datetime
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utils.util import dateBack
from PIL import Image
from utils.arquivo_util import FixedHeightResize
from torchsummary import summary

# Modelo
import model as modeloCNN 

# Modulos para auxilio na estrutura do projeto.
from tqdm import tqdm
import argparse
import logging

def main(parser):

# ************************************ DADOS ***************************************************

  tamanho_da_imagem = 80

  # seleção se existe placa de vídeo
  if torch.cuda.is_available() and parser.device:
    dev = "cuda:0" 
  else:  
    dev = "cpu"

  print(f'Modelo será executado no device {dev}')

  device = torch.device(dev)

  # transformação e normalização dos dados
  transformacoes = transforms.Compose([
              FixedHeightResize(tamanho_da_imagem),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])


  train_dataset = torchvision.datasets.ImageFolder(
    root=parser.data_train_path,
    transform=transformacoes
  )

  # Dataset
  train_dataset = torchvision.datasets.ImageFolder(transform= transformacoes, root= parser.data_train_path)
  val_dataset = torchvision.datasets.ImageFolder(transform= transformacoes, root= parser.data_val_path)

  # Data loader
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=parser.batch_size)
  val_loader = torch.utils.data.DataLoader(dataset=val_dataset, shuffle=True, batch_size=parser.batch_size)

 # ************************************* REDE ************************************************
  rede = modeloCNN.ModelClassificadorPulmonarSequencialFlating()

# carrega a rede treinada anteriormente
  if parser.pesos:
    print('Carregando pesos...')
    rede.load_state_dict(torch.load(parser.dir_save + '/' + parser.save_name + '.pt'))
    print('Pesos Carregados')

  rede.to(device)

  def total_certo(labels, saida):
    total = 0
    for i, val in enumerate(saida):
      val = val.tolist()
      max_idx = val.index(max(val))
      if labels[i] == max_idx:
        total += 1
    return total

  function_loss = nn.CrossEntropyLoss()
  optimizer = optim.SGD(rede.parameters(), lr=0.001, momentum=0.9)
  best_acc = 0
  # confira o link: https://pytorch.org/docs/stable/nn.html

# ************************************ TREINAMENTO E VALIDACAO ********************************************
  for epoch in range(parser.epochs):

    # indica que a rede está em modo treinamento
    rede.train()

    logging.info('Treinamento: {}'.format(str(epoch)))
    train_loss = 0
    acertou = 0
    total = 0

    for batch, (entrada, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
      optimizer.zero_grad()

      # aplica o device nas variaveis que necessitam
      if parser.device:
        entrada = entrada.to(device)
        label = label.to(device)

      saida = rede(entrada)

      acertou += total_certo(label, saida)

      total += len(label)
      loss = function_loss(saida, label)
      loss.backward()

      optimizer.step()

      train_loss += loss.item()

    logging.info('Validacao: {}'.format(str(epoch)))

    val_loss = 0
    acertos = 0

    # aplica o device nas variaveis que necessitam
    rede.eval() 

    with torch.no_grad():
      for batch, (entrada, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
        optimizer.zero_grad()

        # aplica o device nas variaveis que necessitam
        if parser.device:
          entrada = entrada.to(device)
          label = label.to(device)

        saida = rede(entrada)
        acertos += total_certo(label, saida)
        total += len(label)
        loss = function_loss(saida, label)
        
        optimizer.step()
        val_loss += loss.item()

      acc = float(acertos) / (len(val_loader)*parser.batch_size)
      print("Epoch: {}/{} | Train loss error: {:.4f} | Val loss error: {:.4f} | Accuracy:{:.4f}".format(epoch + 1, parser.epochs, train_loss, val_loss, acc))

      if acc > best_acc:
        # Imprime mensagem
        print("Um novo modelo foi salvo")
        # Nome do arquivo dos pesos
        pesos = "{}/{}.pt".format(parser.dir_save, parser.save_name)
        torch.save(rede.state_dict(), pesos)
        best_acc = acc
  print(f'A melhor acurácia alcançada foi de {best_acc}')


  summary(rede, (3, tamanho_da_imagem, tamanho_da_imagem))
  #print(rede)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--dir_save', default="./pesos")
  parser.add_argument('--nomes_labels', type=str, default=['covid', 'normal', 'viral'])
  parser.add_argument('--data_train_path', type=str, default='data/train')
  parser.add_argument('--data_val_path', type=str, default='data/test')
  parser.add_argument('--device', type=bool, default=False)
  parser.add_argument('--batch_size', type=int, default=10)
  parser.add_argument('--pesos', type=bool, default=False)
  parser.add_argument('--save_name', type=str, default="best")
  parser = parser.parse_args()
 
  inicio = datetime.now()
  main(parser)
  fim = datetime.now()
  print('Rede treinada em {}'.format(dateBack(inicio, fromDate=fim)))
  