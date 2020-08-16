# Explore the dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

nrows= 4
ncols = 4

def random_images():
    cats_dir = []
    dogs_dir = []
    cats_src = 'data/dogs_cats/train/cats/'
    dogs_src = 'data/dogs_cats/train/dogs/'
    for file in os.listdir(cats_src):
        cats_dir.append(file)
    for file in os.listdir(dogs_src):
        dogs_dir.append(file)

    print(cats_dir)

    #Randomly select 4 dogs and cats to plot
    images = []
    for _ in range(8):
        i =random.randint(0, len(dogs_dir))
        images.append(cats_src+cats_dir[i])
        images.append(dogs_src+dogs_dir[i])

    for i,path in enumerate(images):
        plt.subplot(nrows,ncols, i+1)
        img = mpimg.imread(path)
        plt.imshow(img)
        plt.axis('Off')
    plt.show()

def model_performance(history):
    # Plot training vs validation accuracy and loss
    train_acc = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(train_acc))

    plt.plot(epochs, train_acc, label='Training acuuracy')
    plt.plot(epochs, val_accuracy, label='Validation accuracy')
    plt.title('Training , Validation accuracy vs Epochs')
    plt.legend()

    plt.figure()

    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training , Validation loss vs Epochs')
    plt.legend()

    plt.show()