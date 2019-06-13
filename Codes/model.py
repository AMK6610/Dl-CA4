import torch.nn as nn
import torch.nn.functional as F
import torch

class DNN(nn.Module):
    def __init__(self, input_dim=39 * 9, state_num=6):
        super(CNN, self).__init__()
        
        self.linear = nn.Sequential(nn.Linear(input_dim, 40), nn.Linear(40, state_num), nn.LogSoftmax())

    def forward(self, input):
        y_pred = self.linear(input)
        return y_pred
