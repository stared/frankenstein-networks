import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from deepsense import neptune
from PIL import Image

# neptune

ctx = neptune.Context()


def neptune_log_scalars(epoch, logs={}):
    # logging numeric channels
    # ctx.channel_send('Dice coef training', self.epoch_id, logs['dice_coef'])
    # ctx.channel_send('Dice coef validation', self.epoch_id, logs['val_dice_coef'])
    ctx.channel_send('Accuracy training', epoch, logs['accuracy'])
    ctx.channel_send('Accuracy validation', epoch, logs['val_accuracy'])
    ctx.channel_send('Log-loss training', epoch, logs['log loss'])
    ctx.channel_send('Log-loss validation', epoch, logs['val_log loss'])


def img_denormalize(img):
    return img * np.array([[[0.5]], [[0.5]], [[0.5]]]) + np.array([[[0.5]], [[0.5]], [[0.5]]])


def array_2d_to_image(array, autorescale=True):
    array = img_denormalize(array).clip(0., 1.)
    assert len(array.shape) in [2, 3]
    if len(array.shape) == 3:
        array = np.moveaxis(array, 0, 2)  # PyTorch (channels, W, H) to PIL (W, H, channels)
    if autorescale:
        array = 255 * array
    array = array.astype('uint8')
    return Image.fromarray(array)


categories = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck']


def neptune_log_images(channel_name, epoch, Xs, Ys, model):
    outputs = F.softmax(model(Xs), dim=1)
    _, predictions = torch.max(outputs, 1)
    for i in range(len(Xs)):
        prediction = predictions[i]
        actual = Ys[i]
        if prediction != actual:
            ctx.channel_send(channel_name, neptune.Image(
                name='[{}] {} X {} V'.format(epoch, categories[prediction], categories[actual]),
                description="\n".join([
                    "{:5.1f}% {} {}".format(100 * score, categories[idx], "!!!" if i == actual else "")
                    for idx, score in enumerate(outputs[i])]),
                data=array_2d_to_image(Xs[i].cpu().numpy())))

# training

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_all(model):
    torch.save(model.state_dict(), '/output/weights.h5')
    print("Model saved.")


def train_model(model, criterion, optimizer, dataloaders, num_epochs=3):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(num_epochs):
        logs = {}
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                running_corrects += (preds == labels.data).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            prefix = ''
            if phase == 'validation':
                prefix = 'val_'

            logs[prefix + 'log loss'] = epoch_loss
            logs[prefix + 'accuracy'] = epoch_acc
        neptune_log_scalars(epoch, logs)

        inputs, labels = next(iter(dataloaders['validation']))
        inputs = inputs.to(device)
        labels = labels.to(device)
        neptune_log_images('False predictions', epoch, inputs, labels, model)

    return model
