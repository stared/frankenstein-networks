# for sewing models

from torch import nn, optim

import utils
import models
import data

import neptune
import json

ctx = neptune.Context()

model1_cut_at = ctx.params['model1_cut_at']
model2_cut_at = ctx.params['model2_cut_at']
epochs = ctx.params['epochs']
learning_rate = ctx.params['learning_rate']
batch_size = ctx.params['batch_size']

# data
dataloaders = data.get_dataloaders('/input/cifar_pytorch', batch_size=batch_size)

# loading base models

model1 = torch.load("/input/model1.pth")
print("Model 1 loaded")
model2 = torch.load("/input/model2.pth")
print("Model 2 loaded")

# making a connection

c1out = sews.get_out_channels(model1cut)
c2in = sews.get_in_channels(model2cut)
ctx.channel_send('M1 Channels Out', 0, c1out)
ctx.channel_send('M2 Channels In', 0, c2in)
connector = sews.make_connector(c1out, c2in)
print("Connector created with {}->{}".format(c1out, c2in))

# making a connection

model1cut = sews.ModelLeftPart(model1, model1_cut_at)
print("Model 1:")
print(model1cut)
print("---\n")

print("Connector:")
print(connector)
print("---\n")

model2cut = sews.ModelLeftPart(model2, model2_cut_at)
print("Model 2:")
print(model2cut)
print("---\n")

sewn_model = sews.SewnConvNet(model1, model2, connector)

print("Network created. Number of parameters:")
print(utils.count_params(sewn_model))
print(sewn_model)

# optimizers

optimizer = optim.Adam(sewn_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(size_average=False)
print("Optimizers created")

# training
utils.train_model(sewn_model, criterion, optimizer, dataloaders, num_epochs=epochs)
utils.save_all(sewn_model, full_model=True)
