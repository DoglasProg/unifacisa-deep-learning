import torch.nn as nn
import torch.nn.functional as F

class ModelClassificadorPulmonar(nn.Module):
    def __init__(self):
      super(ModelClassificadorPulmonar, self).__init__()
      self.conv1 = nn.Conv2d(3, 6, 5)
      self.conv2 = nn.Conv2d(6, 16, 5)
      self.linear1 = nn.Linear(4624, 120)
      self.linear2 = nn.Linear(120, 84)
      self.linear3 = nn.Linear(84, 3)
      self.pool = nn.MaxPool2d(2, 2)
      self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 4624)
      x = F.relu(self.linear1(x))
      x = F.relu(self.linear2(x))
      x = self.dropout(x)
      x = self.linear3(x)
      return x

class ModelClassificadorPulmonarSequencial(nn.Module):
    def __init__(self):
      super(ModelClassificadorPulmonarSequencial, self).__init__()
      self.featutes = nn.Sequential(
                      nn.Conv2d(3, 6, 5),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Conv2d(6, 16, 5),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2)
      )

      self.classificador = nn.Sequential(
                     nn.Linear(4624, 120),
                     nn.ReLU(),
                     nn.Linear(120, 84),
                     nn.ReLU(),
                     nn.Dropout2d(p=0.2),
                     nn.Linear(84, 3)
      )

    def forward(self, x):
      out = self.featutes(x)
      #print(out.shape)
      out = out.view(-1, 4624)
      out = self.classificador(out)
      return out

class ModelClassificadorPulmonarSequencialFlating(nn.Module):
    def __init__(self):
      super(ModelClassificadorPulmonarSequencialFlating, self).__init__()
      self.featutes = nn.Sequential(
                      nn.Conv2d(3, 6, 5),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Conv2d(6, 16, 5),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Flatten()
      )

      self.classificador = nn.Sequential(
                     nn.Linear(4624, 120),
                     nn.ReLU(),
                     nn.Linear(120, 84),
                     nn.ReLU(),
                     nn.Dropout2d(p=0.2),
                     nn.Linear(84, 3)
      )

    def forward(self, x):
      out = self.featutes(x)
      #print(out.shape)
      #out = flatten(out)
      out = self.classificador(out)
      return out