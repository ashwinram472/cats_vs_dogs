# Importing Libraries
from cnn_model import cnn
from inception_model import inception_model
import tensorflow as tf

#Turning off warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
from eda import model_performance


# Doing Image Augmentation to prevent overfitting and capture different features
def train_model(model):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                                                    rotation_range=40,
                                                                    width_shift_range=0.2,
                                                                    height_shift_range=0.2,
                                                                    shear_range=0.2,
                                                                    zoom_range=0.2,
                                                                    horizontal_flip=True,
                                                                    fill_mode = 'nearest')
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory('data/dogs_cats/train/',
                                                        # There are 2000 images
                                                        batch_size=20,
                                                        target_size=(150,150),
                                                        class_mode = 'binary')

    validation_generator = test_datagen.flow_from_directory('data/dogs_cats/test/',
                                                        batch_size=20,
                                                        target_size=(150,150),
                                                        class_mode = 'binary')

    #Fitting model with generator.
    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch=100,
                                  epochs=15,
                                  validation_steps=50,verbose = 2)

    return history,model

if __name__ =="__main__":
    # Using Cnn model
    cnn_model = cnn()
    print('Running CNN Model:')
    cnn_history, cnn_model = train_model(cnn_model)
    cnn_model.save('models\cnn_model.h5')

    # Using Inception model
    inception_model = inception_model()


    # PLotting Model Performance
    #Cnn Model Performance
    model_performance(cnn_history)

