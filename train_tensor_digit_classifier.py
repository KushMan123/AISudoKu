from DigitDataset import DigitDataset
from keras.utils import np_utils
from model.tensor_digit_classifier import KerasModel
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="output/tf_digit_classifier.h5")
args=vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train
# for, and batch size
INIT_LR = 1e-3
EPOCHS = 10
BS = 200
IMG_SIZE = 28

# Showing 100 datasets
def showDatasets(dataset):
    images=[]
    for r in range(10):
        hor_images=[]
        for d in range(10):
            img=dataset[10*r+d][0].reshape(IMG_SIZE,IMG_SIZE).detach().numpy()
            digit_img= cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_CONSTANT, value=(128,))
            hor_images.append(digit_img)
        images.append(np.concatenate(hor_images, axis=1))

    cv2.imshow("Dataset", np.concatenate(images, axis=0))
    cv2.waitKey(0)

def load_data():
    training_set=DigitDataset(IMG_SIZE,60000)
    test_set=DigitDataset(IMG_SIZE,10000)
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    for data in training_set:
        X_train.append(data[0].numpy())
        y_train.append(data[1])
    for data in test_set:
        X_test.append(data[0].numpy())
        y_test.append(data[1])
    return (np.array(X_train), np.array(y_train)),(np.array(X_test),np.array(y_test))

print("[INFO] Loading MNIST Data")

(X_train,y_train),(X_test,y_test)=load_data()

print(X_train.shape)
# #Reshaping the Test and Train Images
X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)

#To enable label into hot vector. For Eg.7 -> [0,0,0,0,0,0,0,1,0,0]
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]

#Initializing Model and Optimizer
print("[INFO] Compling the Model")
model=KerasModel.build(width=28,height=28,depth=1,classes=num_classes)
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=EPOCHS, batch_size=BS)
scores=model.evaluate(X_test,y_test,verbose=0)
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