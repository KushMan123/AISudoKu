from model.pytorch_digit_classifier import Model
from torchvision import transforms
import torch
import numpy as np
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def normalize_points(pts):
    rect=np.zeros((4,2),dtype="float32")
    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]
    diff=np.diff(pts,axis=1)
    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]
    return rect

def predict_digit(model: Model, images: list , image_size, device):
    transform = transforms.Compose([transforms.ToPILImage(), 
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))])
    #Prepare images for the recognition
    images_=[]
    for x,y,img_x, img_y, digit_img in images:
        w,h=digit_img.shape
        #Convert image to sqauare
        if w>h:
            img_square=cv2.copyMakeBorder(digit_img,10,10,10+(w-h)//2,10+w-h-(w-h)//2, cv2.BORDER_CONSTANT, value=(255,))
        else:
            img_square=cv2.copyMakeBorder(digit_img,10+(h-w)//2,10+h-w-(h-w)//2,10,10, cv2.BORDER_CONSTANT, value=(255,))
        data=transform(~img_square).unsqueeze(0)
        images_.append(data)
    
    if len(images_)==0:
        return []

    data = torch.cat(images_)
    model.eval()
    with torch.no_grad():
        out=model(data.to(device))
        p=out.data.max(1,keepdim=True)[1].reshape((len(images_),))
        return p.tolist()


def load_model(model_name,device):
    model=Model()
    model.load_state_dict(torch.load(model_name, map_location= torch.device(device)))
    model.eval()
    return model

def convert_to_grid(board:list):
    grid=[]
    index=0
    for i in range(0,9):
        cols=[]
        for j in range(0,9):
            cols.append(board[index])
            index+=1
        grid.append(cols)
    
    return np.asarray(grid).transpose().tolist()

def convert_to_list(grid):
    return np.asarray(grid).transpose().flatten().tolist()

def show_image(image_name, image):
    img=image.copy()
    cv2.imshow(image_name, img)
    cv2.waitKey(0)

def get_corners(contour, corner=4, max_iter=200):
    coefficient=1
    while max_iter>0 and coefficient >=0:
        max_iter -= -1
        epsilon= coefficient * cv2.arcLength(contour, True)
        poly_approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(poly_approx)
        if len(hull) == corner:
            return hull
        else:
            if len(hull) > corner:
                coefficient += .01
            else:
                coefficient -=.01
    return None

def sort_corners(corners):
    rect = np.zeros((4,2),dtype=np.float32)
    corners = corners.reshape(4,2)
    #Find top left
    tl= sorted(corners, key=lambda p: (p[0]+p[1]))[0]
    #Find top right
    tr= sorted(corners, key=lambda p: (p[0]-p[1]))[-1]
    rect[0]=tl
    rect[1]=tr
    corners=np.delete(corners, np.where(corners==tl)[0][0],0)
    corners=np.delete(corners, np.where(corners==tr)[0][0],0)
    #Sort remaining twi
    if(corners[0][0] > corners[1][0]):
        rect[2] = corners[0]
        rect[3] = corners[1]
    else:
        rect[2] = corners[1]
        rect[3] = corners[0]
    rect = rect.reshape(4,2)
    return rect

def predict_tesseract(images):
  results = []
  for x, y, img_x, img_y, digit_img in images:
    value = predict_digit_tesseract(digit_img, x, y)
    results.append(value)
  return results


def predict_digit_tesseract(digit_img, x, y):
  w, h = digit_img.shape
  if w > h:  # Convert image to square size
      digit_img = cv2.copyMakeBorder(digit_img, 0, 0, 
                       (w - h)//2, w - h - (w - h)//2,
                       cv2.BORDER_CONSTANT, value=(255,))
  digit_img = cv2.copyMakeBorder(digit_img, 
                       w//10, w//10, w//10, w//10, 
                       cv2.BORDER_CONSTANT, value=(255,))
  # Run OCR
  cf='-l eng --psm 8 --dpi 70 -c tessedit_char_whitelist=0123456789'    
  res = pytesseract.image_to_string(digit_img, 
                                    config=cf).strip()
  return int(res[0:1]) if len(res) > 0 else None

def format_grid(grid):
    table=''
    row_index=0
    for row in grid:
        if row_index==0:
            table+=("+-------+-------+-------+\n")
        str_row=""
        for i in range(0,9):
            if i==0:
                str_row+="| "
            if str(row[i]) != "0":
                str_row+=str(row[i])+" "
            else:
                str_row+="  "
            if (i+1)%3==0:
                str_row+="| "
        str_row+="\n"
        table+=str_row
        if (row_index+1)%3==0:
            table+=("+-------+-------+-------+\n")
        row_index+=1
    return table