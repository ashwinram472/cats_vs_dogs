from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model

local_weights_file = 'data/Inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

def inception_model():
#Instatiate the Inception model
# Donot use built in weights
# Use downloaded inception weights

    pretrained_model = InceptionV3(input_shape = (150,150,3), # Use our own input shape
                                   include_top = False,
                                   weights = None)

    #Loading in the saved weights of Inception
    pretrained_model.load_weights(local_weights_file)

    # Locking the pretrained Inception layers
    for layer in pretrained_model.layers:
        layer.trainable = False

    #Using the Mized 7 layer as the last layer
    last_layer = pretrained_model.get_layer('mixed7')



    last_output = last_layer.output


    #Defining our model below the Inception model
    # Flattening the output layer
    x = layers.Flatten()(last_output)

    #Adding a dense layer of 1024 units and Relu activation
    x = layers.Dense(1024 , activation= 'relu')(x)

    # Adding a Dropout of 20%
    x = layers.Dropout(0.2)(x)

    #Final Dense layer output with Sigmoid activation for binary output
    x = layers.Dense(1, activation= 'sigmoid')(x)

    #Creating the mode
    model = Model(pretrained_model.input , x)

    #Compiling the model
    model.compile(optimizer = RMSprop(lr = 0.0001),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    return model