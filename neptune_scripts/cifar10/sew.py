# for sewing models

import torch
from torch import nn, optim
import json
from os import listdir

import utils
import models
import data
import sews

import neptune

ctx = neptune.Context()

model1_cut_at = ctx.params['model1_cut_at']
model2_cut_at = ctx.params['model2_cut_at']
epochs = ctx.params['epochs']
learning_rate = ctx.params['learning_rate']
batch_size = ctx.params['batch_size']

# debugging

print("Inputs:")
print(listdir("/input"))

# data
dataloaders = data.get_dataloaders('/input/cifar_pytorch', batch_size=batch_size)

# loading base models

model1 = torch.load("/input/model1.pth")
print("Model 1 loaded")
model2 = torch.load("/input/model2.pth")
print("Model 2 loaded")

# cutting models

model1cut = sews.ModelLeftPart(model1, model1_cut_at)
model2cut = sews.ModelRightPart(model2, model2_cut_at)
print("Models cut")

# now here for debugging

print("Model 1:")
print(model1cut)
print("---\n")


print("Model 2:")
print(model2cut)
print("---\n")

# making a connection

c1out = 64  # dirty hack
#c1out = sews.get_out_channels(model1cut.convs)
c2in = sews.get_in_channels(model2cut.convs)
ctx.channel_send('M1 Channels Out', 0, c1out)
ctx.channel_send('M2 Channels In', 0, c2in)
connector = sews.make_connector(c1out, c2in)
print("Connector created with {}->{}".format(c1out, c2in))

# making a connection

print("Model 1:")
print(model1cut)
print("---\n")

print("Connector:")
print(connector)
print("---\n")

print("Model 2:")
print(model2cut)
print("---\n")

sewn_model = sews.SewnConvNet(model1cut, model2cut, connector, check=False)
# checking is not general enough

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
