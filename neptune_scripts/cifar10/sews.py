import torch
from torch import nn

def get_out_channels(submodel):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(1, 3, 32, 32).to(device)
    return submodel(x).size(1)

def get_in_channels(submodel):
    if isinstance(submodel, nn.Sequential):
        return get_in_channels(submodel[0])
    elif isinstance(submodel, nn.Linear):
        return submodel.in_features
    elif isinstance(submodel, nn.Conv2d):
        return submodel.in_channels
    else:
        raise Exception("Layer not supported for in_channels: {}".format(submodel.__class__))


class ModelLeftPart(nn.Module):
    def __init__(self, original_model, cut_at):
        super().__init__()
        self.convs = original_model.convs[:cut_at]

    def forward(self, x):
        x = self.convs(x)
        return x


class ModelRightPart(nn.Module):
    def __init__(self, original_model, cut_at):
        super().__init__()
        self.convs = original_model.convs[cut_at:]
        self.fc = original_model.fc

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x.view(x.size(0), -1))
        return x

def make_identity_connector():
    return nn.Sequential()

def make_connector(in_channels, out_channels, intermediate_channels=None):
    if not intermediate_channels:
        intermediate_channels = in_channels

    return nn.Sequential(
        nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, padding=0),
        nn.ReLU(inplace=True)
    )

class SewnConvNet(nn.Module):
    def __init__(self, net_before, net_after, connector, check=True):
        super().__init__()
        self.net_before = net_before.eval()
        self.net_after = net_after.eval()
        self.connector = connector

        if check:
            self._assert_channels()

    def forward(self, x):
        x = self.net_before(x)
        x = self.connector(x)
        x = self.net_after(x)
        return x

    def _assert_channels(self):
        '''
        net_before, net_after, connector - nn.Sequential or nn.ModuleList
        '''
        before_out_channels = self._find_out_channels(self.net_before)
        connector_in_channels = self._find_in_channels(self.connector)
        if before_out_channels != connector_in_channels:
            raise ValueError('Connector has {} input channels, expected {}.'.format(
                connector_in_channels, before_out_channels)
            )

        connector_out_channels = self._find_out_channels(self.connector)
        after_in_channels = self._find_in_channels(self.net_after)
        if after_in_channels != connector_out_channels:
            raise ValueError('Connector has {} output channels, expected {}.'.format(
                connector_out_channels, after_in_channels)
            )

    def parameters(self, recurse=True):
        return self.connector.parameters()

    def train(self, mode=True):
        self.training = mode
        self.connector.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def _find_in_channels(self, half):
        for m in half.modules():
            if isinstance(m, nn.Conv2d):
                return m.in_channels
            if isinstance(m, nn.Linear):
                return m.in_features

    def _find_out_channels(self, half):
        out_channels = None
        for m in half.modules():
            if isinstance(m, nn.Conv2d):
                out_channels = m.out_channels
            if isinstance(m, nn.Linear):
                out_channels = m.out_features
        return out_channels
