import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.sigmoid(self.hidden1(x))
        out = self.sigmoid(self.hidden2(out))
        out = self.output(out)  # No softmax here (handled in CrossEntropyLoss)
        return out
    
    def init_weights(self, init_type='xavier'):
        if init_type == 'normal':
            init = torch.nn.init.normal_
        elif init_type == 'xavier':
            init = torch.nn.init.xavier_uniform_
        elif init_type == 'kaiming':
            init = torch.nn.init.kaiming_uniform_
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
