import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class TicTacToeEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b0'):
        super(TicTacToeEfficientNet, self).__init__()
        self.base = EfficientNet.from_pretrained(model_name)
        num_features = self.base._fc.in_features
        self.base._fc = nn.Linear(num_features, 9 * 3)

    def forward(self, x):
        x = self.base(x)
        x = x.view(-1, 9, 3)
        return x

def create_model():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_features = model._fc.in_features
    model._fc = nn.Linear(num_features, 9 * 3)
    return model
