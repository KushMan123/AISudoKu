from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

class DigitClassifier:
    @staticmethod
    def build(width,height,depth, classes):
        model=Sequential()
        input_shape=(depth,height,width)
        model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(16,(3,3),activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128,activation="relu"))
        model.add(Dense(64,activation="relu"))
        model.add(Dense(classes,activation="softmax"))

        return model



