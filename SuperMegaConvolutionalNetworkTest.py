from multiprocessing import freeze_support

from torch import nn, optim
import SuperFirstMegaNetwork
from SuperFirstMegaNetwork import PrimeNet, train_model, SewnConvNet, make_connector_1conv, make_connector_2conv

if __name__ == '__main__':
    freeze_support()

model1 = PrimeNet(nn.ReLU, nn.MaxPool2d, 2, 0, 0, k_size=5)
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=1e-4)
epoch_num = 20

if __name__ == '__main__':
    freeze_support()
model1_trained = train_model(model1, criterion, optimizer1, num_epochs=epoch_num)
