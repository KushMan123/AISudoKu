from keras.datasets import mnist
from keras.utils import np_utils
from model.digit_classifier import DigitClassifier
import matplotlib.pyplot as plt
import numpy as np
import argparse 

ap=argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="output/digit_classifier.h5")
args=vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train
# for, and batch size
INIT_LR = 1e-3
EPOCHS = 10
BS = 200

print("[INFO] Loadinf MNIST Data")
(X_train, y_train),(X_test, y_test)=mnist.load_data()

#Reshaping the Test and Train Images
X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)

#Converting the Test and Train image to 32 bit precision
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Normalizing the value between 0.0 and 1.0
X_train = X_train/255
X_test= X_test/255

#To enable label into hot vector. For Eg.7 -> [0,0,0,0,0,0,0,1,0,0]
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]

#Initializng Model and Optimizer
print("[INFO] Compling the Model")
model=DigitClassifier.build(width=28,height=28,depth=1,classes=num_classes)
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
model.fit(X_train,y_train, validation_data=(X_test,y_test),epochs=EPOCHS, batch_size=BS)
scores=model.evaluate(X_test,y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

#Testing the above Model
test_images=X_test[1:5]
test_images=test_images.reshape(test_images.shape[0],28,28)
print("Test images shape:{}".format(test_images.shape))
for i, test_image in enumerate(test_images, start=1):
    org_image=test_image
    test_image=test_image.reshape(1,28,28,1)
    prediction=model.predict(test_image)
    prediction_class=np.argmax(prediction,axis=1)
    print("Predicted digit:{}".format(prediction_class))
    plt.subplot(220+i)
    plt.axis("off")
    plt.title("Predicted digit:{}".format(prediction_class))
    plt.imshow(org_image,cmap=plt.get_cmap("gray"))
plt.show()

# serialize the model to disk
print("[INFO] serializing digit model...")
model.save(args["model"], save_format="h5")