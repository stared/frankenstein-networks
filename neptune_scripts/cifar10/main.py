from torch import nn, optim

import utils
import models
import data

from deepsense import neptune

ctx = neptune.Context()

model_name = ctx.params['model']
epochs = ctx.params['epochs']
learning_rate = ctx.params['learning_rate']
batch_size = ctx.params['batch_size']

ctx.tags.append(model_name)

# data
dataloaders = data.get_dataloaders('/input', batch_size=batch_size)

# network
model = models.MODELS[model_name]()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(size_average=False)

print("Network created. Number of parameters:")
print(utils.count_params(model))

# training
utils.train_model(model, criterion, optimizer, dataloaders, num_epochs=epochs)
utils.save_all(model)
