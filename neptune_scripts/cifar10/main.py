# for training models from scratch

from torch import nn, optim

import utils
import models
import data

import neptune
import json

ctx = neptune.Context()

model_name = ctx.params['model']
epochs = ctx.params['epochs']
learning_rate = ctx.params['learning_rate']
batch_size = ctx.params['batch_size']

channel_sequence = json.loads(ctx.params['channel_sequence'])
fc_type = ctx.params['fc_type']

ctx.tags.append(model_name)

# data
dataloaders = data.get_dataloaders('/input/cifar_pytorch', batch_size=batch_size)

# network
model = models.MODELS[model_name](
    channel_sequence=channel_sequence,
    fc_type=fc_type
)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(size_average=False)

print("Network created. Number of parameters:")
print(utils.count_params(model))
print(model)

# training
utils.train_model(model, criterion, optimizer, dataloaders, num_epochs=epochs)
utils.save_all(model, full_model=True)
