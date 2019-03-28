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


class ConvNet1(nn.Module):
    def __init__(self,
                 n_classes=10,
                 channel_sequence=[32, 64, 64, 64],
                 fc_type='linear'):
        super().__init__()

        cs = [3] + channel_sequence
        layers = [self._block(cs[i], cs[i+1]) for i in range(len(channel_sequence))]
        self.convs = nn.Sequential(*layers)

        self.last_conv_res = 32 // 2**len(self.convs)
        assert self.last_conv_res != 0

        self.pre_fc_size = channel_sequence[-1] * self.last_conv_res**2

        if fc_type == 'linear':
            self.fc = nn.Linear(self.pre_fc_size, n_classes)
        elif fc_type == 'nonlinear':
            self.fc = nn.Sequential(
                nn.Linear(self.pre_fc_size, self.pre_fc_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.pre_fc_size, n_classes)
            )
        elif fc_type == 'nonlinear_dropout':
            self.fc = nn.Sequential(
                nn.Linear(self.pre_fc_size, self.pre_fc_size),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(self.pre_fc_size, n_classes)
            )
        else:
            raise Exception("No fc_type: " + fc_type)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x.view(-1, self.pre_fc_size))
        return x

# here we want to only use convolutional models, so we won'y use linear regression on MLP

MODELS = {
    'convnet1': ConvNet1
}
