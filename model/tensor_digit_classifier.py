from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

class KerasModel:
    @staticmethod
    def build(width,height,depth,classes):
        model=Sequential()
        input_shape=(depth,height,width)
        model.add(Conv2D(32,(5,5),input_shape=(28,28,1), strides=1, activation="relu"))
        model.add(Conv2D(64,(5,5),strides=1, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10,activation="softmax"))

        return model



