from model.pytorch_digit_classifier import Model
from DigitDataset import DigitDataset
import cv2
import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F

EPOCHS=12
BATCH_SIZE=10
LOG_INTERVAL=600
MODEL_FILE_NAME="output/ocr_model.pt"
IMG_SIZE=32

# Use GPU if available
use_cuda=torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

# Generating the dataset
dataset=DigitDataset(IMG_SIZE, 10000)

# train_loader=torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# # Instantiating Model
# model=Model().to(device)

# # Training Model
# model.train()
# print("training model, use CUDA:", use_cuda)
# optimizer=optim.Adadelta(model.parameters(),lr=0.01)
# for epoch in range(EPOCHS):
#     print(f"EPOCHS {epoch+1} of {EPOCHS}")
#     t1=time.monotonic()
#     total_loss=0
#     for batch_index, (data,target) in enumerate(train_loader):
#         data, target=data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output=model(data)
#         loss=F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_index % LOG_INTERVAL == 0:
#             print('Train[{}/{} ({:.0f}%)] Loss: {:0.6f}'. format(batch_index* len(data), len(train_loader.dataset), 100*batch_index/len(train_loader), loss.item()))
#         total_loss+=loss.item()
#     print(f"Total Loss: {total_loss}, dT={time.monotonic()-t1}s")

# torch.save(model.state_dict(), MODEL_FILE_NAME)
# print("Model svaed to", MODEL_FILE_NAME)

# Showing 100 datasets
def showDatasets():
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

showDatasets()
