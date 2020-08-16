
#importing libraries
import tensorflow as tf

def cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3) , activation='relu' , input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3) , activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3) , activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),

        #Flattening to insert into DNN
        tf.keras.layers.Flatten(),
        #512 hidden neuron layer
        tf.keras.layers.Dense(512, activation='relu'),
        #One output neuron because binary classification. Using sigmoid activation function
        # 0 for cats 1 for dogs
        tf.keras.layers.Dense(1 , activation = 'sigmoid')
    ])

    model.summary()
    model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

    return model