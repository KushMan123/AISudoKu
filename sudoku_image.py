import cv2
import numpy as np
import matplotlib.pyplot as plt
from solve_sudoku import find_solution
from utilis import normalize_points, predict_digit, load_model, convert_to_grid, convert_to_list, show_image
import argparse

#https://blog.devgenius.io/solving-sudoku-in-real-time-using-a-convolutional-neural-network-and-opencv-e47a92478dce

#ArgParser
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="images/1.png")
ap.add_argument("-d","--debug", type=int, default=-1, help="debug")
args=vars(ap.parse_args())

IMG_SIZE=32
MODEL_FILE_NAME="output/ocr_model.pt"

# Setting Device and Debug
print("[INFO] Setting Up Device")
device="cpu"
debug=args["debug"] > 0
    
#Loading Model
print("[INFO] Loading OCR Model")
model=load_model(MODEL_FILE_NAME ,device)

#Loading the image and apply Canny Edge Detection
# Link: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
print("[INFO] Reading Image")
img=cv2.imread(args["image"])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edged=cv2.Canny(img_gray,100,400)
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, 
                               cv2.CHAIN_APPROX_SIMPLE)

#applying gaussain blur
blurred=cv2.GaussianBlur(img_gray,(11,11),0)
img_bw=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#Detecting Contour in the Image
img_out=img.copy()
w,h=img.shape[1],img.shape[0]

for cntr in contours:
    print("[INFO] Finding Suitable Contours")
    imgx,imgy,imgw,imgh=cv2.boundingRect(cntr)
    if imgw<(w/5) or imgw < (h/5) or imgw/imgh <0.25 or imgw/imgh>1.5:
        continue
    
    #Approximate the contours in 4 points
    peri=cv2.arcLength(cntr,True) # Calaculate the perimeter
    frm=cv2.approxPolyDP(cntr,0.1*peri, True) # Approximate the shape of the contour
    if len(frm)!=4:
        continue
    
    #Converted image should fit into the original size
    board_size=max(imgw,imgh)
    if imgx + board_size >= w or imgy + board_size >= h:
        continue

    #Points should not be too  close to eachother
    if cv2.norm(frm[0][0]-frm[1][0], cv2.NORM_L2)< 0.1*peri or \
    cv2.norm(frm[2][0]-frm[1][0], cv2.NORM_L2)< 0.1*peri or \
    cv2.norm(frm[3][0]-frm[1][0], cv2.NORM_L2)< 0.1*peri or \
    cv2.norm(frm[3][0]-frm[2][0], cv2.NORM_L2)< 0.1*peri :
        continue
    
    print("[INFO] Drawing sudoku contour using lines and points")
    # Draw sudoku contour using lines and points
    cv2.line(img_out, frm[0][0], frm[1][0], (0, 200, 0), thickness=3)
    cv2.line(img_out, frm[1][0], frm[2][0], (0, 200, 0), thickness=3)
    cv2.line(img_out, frm[2][0], frm[3][0], (0, 200, 0), thickness=3)
    cv2.line(img_out, frm[0][0], frm[3][0], (0, 200, 0), thickness=3)
    cv2.drawContours(img_out, frm, -1, (0, 255, 255), 10)

    if debug:
        show_image('image', img_out)
    
    print("[INFO] Perspective Transforming")
    #Source and destination points for perspective transform
    print(board_size)
    src_pts=normalize_points(frm.reshape(4,2))
    dst_pts=np.array([[0,0],[board_size,0],[board_size,board_size],[0,board_size]], dtype=np.float32)
    t_matrix=cv2.getPerspectiveTransform(src_pts,dst_pts)
    _, t_matrix_inv = cv2.invert(t_matrix)

    # Convert images, colored and monochrome
    warped_disp = cv2.warpPerspective(img, t_matrix, (board_size, board_size))
    warped_bw = cv2.warpPerspective(img_bw, t_matrix, (board_size, board_size))

    show_image('image', warped_disp)

    if debug:
        show_image('image', warped_bw)

    print("[INFO] Extracting Digits")
    #Sudoku Board Found, Now extracting the digits
    images=[]
    cell_w, cell_h= board_size//9,board_size//9
    for x in range(9):
        for y in range(9):
            x1, y1, x2, y2 = x*cell_w, y*cell_h, (x + 1)*cell_w, (y + 1)*cell_h
            cx, cy, w2, h2 = (x1 + x2)//2, (y1 + y2)//2, cell_w, cell_h
            # Find the contour of the digits
            crop=warped_bw[y1:y2,x1:x2]
            cntrs,_=cv2.findContours(crop,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

            for dc in cntrs:
                imgx2, imgy2, imgw2, imgh2=cv2.boundingRect(dc)
                if 0.2 * w2 < imgw2  < 0.8 * w2 and 0.4 * h2 < imgh2 < 0.8 * h2:
                    cv2.rectangle(warped_disp, (x1 + imgx2, y1 + imgy2), (x1 + imgx2 + imgw2, y1 + imgy2 + imgh2), (0, 255, 0), 1)
                    digit_img = crop[imgy2:imgy2 + imgh2, imgx2:imgx2 + imgw2]
                    images.append((x, y, cx, cy, digit_img))
                    if debug:
                        show_image("Digit Image",digit_img)
                    break
    
    print("[INFO] Predicting Digits")
    #Predict the result and Draw OCR results
    results=predict_digit(model,images, IMG_SIZE, device) 
    board=[0]*(9*9)
    for (x,y,img_x,img_y,digit_img), result in zip(images, results):
        board[9*x+y]=result
        #Calculate coordinates on the original image
        orig=cv2.perspectiveTransform(np.array([[[img_x,img_y]]],dtype=np.float32),t_matrix_inv).reshape((2,)).astype(np.int32)
        cv2.putText(img_out,str(result),orig,cv2.FONT_HERSHEY_SIMPLEX,1,(128,0,0),2,cv2.LINE_AA, False)

    if debug:
        show_image("Image with predicted Text", img_out)

    print("[INFO] Finding Solution")
    #Solve the OCR board
    grid_board=convert_to_grid(board)
    solved_grid_board, solution_found = find_solution(grid_board)
    board_orig= list(board)
    solved_board=convert_to_list(solved_grid_board)
    if solution_found:
        print("[INFO] Solution Found")
        for x in range(9):
            for y in range(9):
                if board_orig[9*x+y]==0:
                    pt_x, pt_y =x*cell_w+cell_w//2, y*cell_h+cell_h//2
                    pt_origin=cv2.perspectiveTransform(np.array([[[pt_x,pt_y]]], dtype=np.float32), t_matrix_inv). reshape((2,)).astype(np.int32)
                    cv2.putText(img_out,str(solved_board[9*x+y]),pt_origin,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,200),2,cv2.LINE_AA, False)
    else:
        print("[INFO] Solution Not Found")

    
    show_image("Solution Set", img_out)
    show_image("Edged", edged)
            
##Displaying Images and Edge Images and Gaussian Blur
# plt.imshow(img_gray,cmap=plt.get_cmap("gray"))
# plt.show()
# plt.imshow(edged,cmap=plt.get_cmap("gray"))
# plt.show()
# plt.imshow(img_bw,cmap=plt.get_cmap("gray"))
# plt.show()
