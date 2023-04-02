from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Flatten, Dense, Dropout

class SudokuNet:
    @staticmethod
    def build(width, height, depth, classes):
        model=Sequential()
        inputShape=(height,width,depth)
        #first set of Conv2D => RELU => MaxPool
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(28,28,1),
          activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (3,3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(classes, activation='softmax'))

        return model

