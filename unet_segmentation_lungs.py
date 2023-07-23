# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 20:50:12 2023

@author: Rebahath
"""

from models.UNet import UNet
import matplotlib.pyplot as plt
import numpy as np

x_train = np.load('./dataset/x_train.npy')
y_train = np.load('./dataset/y_train.npy')
x_test = np.load('./dataset/x_test.npy')
y_test = np.load('./dataset/y_test.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

seg_model = UNet(img_shape=x_train[0].shape, num_of_class=1, learning_rate=2e-4, do_drop=True, drop_rate=0.5)
seg_model.show_model()

history = seg_model.train(x_train, y_train, epoch=100, batch_size=64)

def plot_dice(history, title=None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['dice_coef'])
    plt.plot(history['val_dice_coef'])
    if title is not None:
        plt.title(title)

    plt.ylabel('Dice_coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc=0)
    
def plot_loss(history, title=None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc=0)

    
plot_dice(history)
plot_loss(history)

preds = seg_model.predict(x_test)
show_num = 10
fig, ax = plt.subplots(show_num, 3, figsize=(15, 50))

for i, pred in enumerate(preds[:show_num]):
    ax[i, 0].imshow(x_test[i].squeeze(), cmap='gray')
    ax[i, 1].imshow(y_test[i].squeeze(), cmap='gray')
    ax[i, 2].imshow(pred.squeeze(), cmap='gray')