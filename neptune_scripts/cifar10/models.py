from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = self.hidden_layer(x.view(-1, 3 * 32 * 32))
        return x


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(3 * 32 * 32, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.hidden_layers(x.view(-1, 3 * 32 * 32))
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64)
        )
        self.linear = nn.Linear(64 * 8 * 8, 10)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.linear(x.view(-1, 64 * 8 * 8))
        return x


MODELS = {
    'logistic': LogisticRegression(),
    'mlp': MLP(),
    'conv': ConvNet()
}
